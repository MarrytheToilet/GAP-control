"""Inference (handbook 2.9) + baselines.

GAP-Control decode loop, per step:
    h_t, l0_t = BaseLM(prefix)
    r_t, V_hat = Controller(h_t, alpha)
    b_raw      = W_LM r_t
    b          = GapProject( center(b_raw|S_t), rho(g_t) )      # 0 outside S_t
    y_t        ~ softmax(l0_t + b)

Complexity: one base forward + one tiny controller forward per step. No expert LM, no
candidate-level rollout, no inference-time prefix scorer.

Also provides two baselines sharing the same sampler for fair comparison:
  * prompt_only  -- plain base sampling (optionally with an attribute instruction prefix)
  * fudge_like   -- a FUDGE-style discriminator reweighting of top-k using the reward as
                    a (slow) future predictor; included as a control-quality/efficiency point.
"""
from __future__ import annotations

import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from .base_lm import BaseLM
from .controller import GapController
from . import projection as proj


def _sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    logits = logits / max(temperature, 1e-6)
    probs = logits.softmax(-1)
    sorted_p, sorted_idx = probs.sort(descending=True)
    cum = sorted_p.cumsum(-1)
    mask = cum - sorted_p > top_p          # keep tokens up to and incl. the crossing one
    sorted_p[mask] = 0.0
    sorted_p /= sorted_p.sum().clamp_min(1e-12)
    choice = torch.multinomial(sorted_p, 1)
    return sorted_idx[choice].item()


@torch.no_grad()
def gap_decode(
    base: BaseLM,
    controller: GapController,
    prompt_ids: List[int],
    *,
    alpha: torch.Tensor,            # [A] reward-mixture / intensity vector
    cfg,
    condition=None,                 # ControlCondition: enables the live state signal
) -> dict:
    """Returns dict with text, token ids, and per-step diagnostics.

    The dynamic loop: each step we recompute the live running-state (length/keyword/
    structure satisfaction so far), feed it to the controller, and clamp the value gap with
    the monotone observed-satisfaction so the residual switches off once the constraint is met.
    """
    from . import state as state_mod
    device = base.device
    ids = list(prompt_ids)
    step_kl, step_bnorm, step_gap, step_vhat, step_rank, step_obs = [], [], [], [], [], []
    t0 = time.time()
    use_state = condition is not None and controller.state_dim > 0
    # tokens to force into the active set so the residual can surface a required keyword
    # or structural marker ('?', '!'); length is handled by an EOS budget bias below.
    kw_ids = condition.keyword_active_ids(base.tokenizer) if condition is not None else []
    struct_ids = condition.structure_active_ids(base.tokenizer) if condition is not None else []
    struct_marks = condition.structure_marks() if condition is not None else []
    len_spec = condition.length_spec() if condition is not None else None
    hard_ids = sorted(set(kw_ids) | set(struct_ids))

    past = None
    next_inp = torch.tensor([ids], device=device)                # first step: full prompt
    hfracs = cfg.hidden_fracs() if hasattr(cfg, "hidden_fracs") else (1.0,)
    for _ in range(cfg.gen_max_new):
        h_t, l0, past = base.step_cached(next_inp, past=past, hidden_fracs=hfracs)  # KV-cached: O(T)

        # live state from the continuation generated so far (the dynamic signal)
        gen_text = base.decode(ids[len(prompt_ids):])
        obs_sat = 0.0
        sf = None
        if use_state:
            sf = state_mod.running_state(gen_text, len(ids) - len(prompt_ids),
                                         condition, max_new=cfg.gen_max_new
                                         ).unsqueeze(0).to(device)
            obs_sat = state_mod.observed_satisfaction(gen_text, condition)

        r_t, v_hat = controller(h_t, alpha.unsqueeze(0), sf)
        l0 = l0.squeeze(0)
        u_t = (1.0 - l0.softmax(-1).max()).reshape(1)        # base uncertainty for the gate
        gate_mode = getattr(cfg, "projection", "l2") == "gate"
        if not gate_mode:
            # legacy path: pre-scale the residual (decode_strength x entropy_gate heuristic)
            strength = getattr(cfg, "decode_strength", 1.0)
            eg = getattr(cfg, "entropy_gate", 0.0)
            if eg > 0:    # steer more where the base is uncertain (preserves fluency on confident tokens)
                strength = strength * float(u_t.item() ** eg)
            r_t = r_t * strength
        # in gate mode the unified budget rho_t owns the magnitude (no pre-scaling)
        b_raw = controller.residual_from_r(r_t, base.lm_head).squeeze(0)  # [V]

        # active set from the base distribution, with required keyword tokens forced in
        K = min(cfg.decode_topk, l0.numel())
        base_logprob = l0.log_softmax(-1)
        _, top_ids = base_logprob.topk(K)
        if hard_ids:
            extra = [e for e in hard_ids if e not in set(top_ids.tolist())]
            if extra:
                top_ids = torch.cat([top_ids[: K - len(extra)],
                                     torch.tensor(extra, device=device)])
        top_lp = base_logprob[top_ids]
        p0_active = top_lp.exp()
        p0_active = (p0_active / p0_active.sum().clamp_min(1e-12)).unsqueeze(0)
        b_active = b_raw[top_ids].unsqueeze(0)

        # clamp the gap with observed satisfaction: monotone constraints already met -> gap 0
        v_eff = torch.clamp(v_hat, min=obs_sat) if use_state else v_hat
        b_proj = proj.gap_project(
            b_active, p0_active, v_eff,
            mode=cfg.projection, rho_min=cfg.rho_min, rho_max=cfg.rho_max,
            reward_target=cfg.reward_target, kl_budget_max=cfg.kl_budget_max,
            u_t=u_t, gate_gamma=getattr(cfg, "gate_gamma", 0.0),
        ).squeeze(0)

        # apply residual on active set only
        new_logits = l0.clone()
        new_logits[top_ids] = l0[top_ids] + b_proj

        # dynamic lexical term for pinpoint keyword control: push the forced keyword tokens,
        # gated by (1 - observed_satisfaction) so it vanishes once the keyword has appeared.
        lex = getattr(cfg, "lexical_strength", 0.0)
        if lex and kw_ids and obs_sat < 1.0:
            for e in kw_ids:
                new_logits[e] += lex * (1.0 - obs_sat)
        # structural-marker push ('?', '!'): same mechanism, gated by whether the marker
        # has appeared yet (a present marker makes the structure satisfied).
        if lex and struct_ids:
            need_mark = not any(m in gen_text for m in struct_marks)
            if need_mark:
                for e in struct_ids:
                    new_logits[e] += lex
        # length control via an EOS budget bias (a one-step residual cannot count tokens):
        # forbid EOS below the minimum, force it at the maximum, ramp in between.
        if len_spec is not None:
            lo, hi = len_spec
            n_gen = len(ids) - len(prompt_ids)
            eos_scale = getattr(cfg, "length_eos_scale", 12.0)
            if n_gen < lo:
                new_logits[base.eos_token_id] -= eos_scale
            elif n_gen >= hi:
                new_logits[base.eos_token_id] += eos_scale
            else:
                new_logits[base.eos_token_id] += eos_scale * (n_gen - lo) / max(hi - lo, 1)

        # diagnostics
        p_new = new_logits.log_softmax(-1).exp()
        kl = (p_new * (p_new.clamp_min(1e-12).log() - base_logprob)).sum().item()
        step_kl.append(kl)
        step_bnorm.append(b_proj.norm().item())
        step_gap.append(proj.value_gap(v_eff, cfg.reward_target).item())
        step_vhat.append(v_hat.item())
        step_obs.append(obs_sat)

        nxt = _sample(new_logits, cfg.gen_top_p, cfg.gen_temperature)
        # rank of chosen token under the *base* distribution (low-perturbation diagnostic)
        step_rank.append((base_logprob > base_logprob[nxt]).sum().item())
        ids.append(nxt)
        if nxt == base.eos_token_id:
            break
        next_inp = torch.tensor([[nxt]], device=device)          # only the new token next step

    elapsed = time.time() - t0
    n_new = len(ids) - len(prompt_ids)
    return {
        "ids": ids,
        "new_ids": ids[len(prompt_ids):],
        "text": base.decode(ids[len(prompt_ids):]),
        "full_text": base.decode(ids),
        "ms_per_token": 1000.0 * elapsed / max(n_new, 1),
        "mean_kl": float(sum(step_kl) / max(len(step_kl), 1)),
        "mean_bnorm": float(sum(step_bnorm) / max(len(step_bnorm), 1)),
        "trajectory": {"kl": step_kl, "bnorm": step_bnorm, "gap": step_gap,
                       "vhat": step_vhat, "obs_sat": step_obs, "base_rank": step_rank},
    }


def _finish(base, ids, prompt_len, elapsed) -> dict:
    n_new = len(ids) - prompt_len
    return {
        "ids": ids, "new_ids": ids[prompt_len:],
        "text": base.decode(ids[prompt_len:]),
        "full_text": base.decode(ids),
        "ms_per_token": 1000.0 * elapsed / max(n_new, 1),
        "mean_kl": 0.0, "mean_bnorm": 0.0,
    }


@torch.no_grad()
def _cached_sample(base: BaseLM, ids: list, cfg) -> list:
    """KV-cached plain top-p sampling continuation from `ids`. Returns extended ids."""
    device = base.device
    past = None
    next_inp = torch.tensor([ids], device=device)
    for _ in range(cfg.gen_max_new):
        _, l0, past = base.step_cached(next_inp, past=past, need_hidden=False)
        nxt = _sample(l0.squeeze(0), cfg.gen_top_p, cfg.gen_temperature)
        ids.append(nxt)
        if nxt == base.eos_token_id:
            break
        next_inp = torch.tensor([[nxt]], device=device)
    return ids


@torch.no_grad()
def instructed_decode(base: BaseLM, prompt_text: str, instruction: str, *,
                      cfg, fewshot: List[str] = None) -> dict:
    """Prompt/instruction baseline: prepend a control instruction and sample normally."""
    shots = ("\n".join(f"Example: {s}" for s in fewshot) + "\n") if fewshot else ""
    header = f"Write a continuation that is {instruction}.\n{shots}"
    ids = base.encode(header + prompt_text)
    prompt_len = len(ids)
    t0 = time.time()
    ids = _cached_sample(base, ids, cfg)
    return _finish(base, ids, prompt_len, time.time() - t0)


@torch.no_grad()
def contrastive_decode(base: BaseLM, views: List[dict], *, cfg) -> dict:
    """General logit-arithmetic decoder shared by CFG and PREADD.

    views = [{"ids": prefix_token_ids, "w": weight}, ...]. The same continuation tokens are
    appended to every view; the next-token logits are sum_i w_i * logits_i. The first view
    is the 'content' view whose decoded continuation is returned.
    """
    device = base.device
    views = [{"ids": list(v["ids"]), "w": v["w"], "past": None} for v in views]
    content = views[0]
    prompt_len = len(content["ids"])
    t0 = time.time()
    for step in range(cfg.gen_max_new):
        combined = None
        for v in views:
            inp = torch.tensor([v["ids"]], device=device) if step == 0 \
                else torch.tensor([[v["ids"][-1]]], device=device)   # KV-cached per view
            _, l0, v["past"] = base.step_cached(inp, past=v["past"], need_hidden=False)
            l0 = l0.squeeze(0)
            combined = v["w"] * l0 if combined is None else combined + v["w"] * l0
        nxt = _sample(combined, cfg.gen_top_p, cfg.gen_temperature)
        for v in views:
            v["ids"].append(nxt)
        if nxt == base.eos_token_id:
            break
    return _finish(base, content["ids"], prompt_len, time.time() - t0)


@torch.no_grad()
def cfg_decode(base: BaseLM, prompt_text: str, instruction: str, *, cfg, gamma: float = 1.5) -> dict:
    """Classifier-free guidance (Sanchez et al. 2023): l = l_uncond + γ(l_cond − l_uncond).
    cond = base | (instruction + prompt); uncond = base | prompt."""
    cond_ids = base.encode(f"Write a continuation that is {instruction}.\n{prompt_text}")
    uncond_ids = base.encode(prompt_text)
    # content view must decode cleanly -> use uncond prefix as content, but it needs the
    # combined logits; put cond first as content for stable decode, strip its header length.
    return contrastive_decode(base, [
        {"ids": cond_ids, "w": gamma},
        {"ids": uncond_ids, "w": 1.0 - gamma},
    ], cfg=cfg)


@torch.no_grad()
def preadd_decode(base: BaseLM, prompt_text: str, instruction: str,
                  anti_instruction: Optional[str], *, cfg, alpha: float = 2.0) -> dict:
    """PREADD / DExperts-style contrastive prompting (base-only): l = l_base + α(l_pos − l_neg).
    pos = base | (instruction + prompt); neg = base | (anti + prompt) or plain prompt."""
    base_ids = base.encode(prompt_text)
    pos_ids = base.encode(f"Write a continuation that is {instruction}.\n{prompt_text}")
    neg_text = (f"Write a continuation that is {anti_instruction}.\n{prompt_text}"
                if anti_instruction else prompt_text)
    neg_ids = base.encode(neg_text)
    return contrastive_decode(base, [
        {"ids": base_ids, "w": 1.0},
        {"ids": pos_ids, "w": alpha},
        {"ids": neg_ids, "w": -alpha},
    ], cfg=cfg)


@torch.no_grad()
def best_of_n_decode(base: BaseLM, reward_fn, prompt_ids: List[int], *, cfg, n: int = 8) -> dict:
    """Sample n base completions, return the one the reward scores highest. Strong test-time
    compute baseline (inference-time control via reranking)."""
    device = base.device
    inp = torch.tensor([prompt_ids] * n, device=device)
    t0 = time.time()
    gen = base.generate(inp, max_new_tokens=cfg.gen_max_new, do_sample=True,
                        top_p=cfg.gen_top_p, temperature=cfg.gen_temperature)
    new = gen[:, len(prompt_ids):]
    texts = [base.decode(row) for row in new]
    scores = reward_fn(texts)
    best = int(scores.argmax())
    elapsed = time.time() - t0
    ids = prompt_ids + new[best].tolist()
    out = _finish(base, ids, len(prompt_ids), elapsed)
    out["ms_per_token"] = 1000.0 * elapsed / max(n * cfg.gen_max_new, 1)  # amortized over all n
    return out


@torch.no_grad()
def prompt_only_decode(base: BaseLM, prompt_ids: List[int], *, cfg) -> dict:
    t0 = time.time()
    ids = _cached_sample(base, list(prompt_ids), cfg)
    return _finish(base, ids, len(prompt_ids), time.time() - t0)


@torch.no_grad()
def fudge_like_decode(base: BaseLM, reward_fn, prompt_ids: List[int], *,
                      cfg, lam: float = 4.0, fudge_topk: int = 20,
                      fudge_samples: int = 1, fudge_lookahead: int = 6) -> dict:
    """FUDGE-style baseline: reweight top-k candidates by an inference-time future
    predictor (here the reward applied to a short lookahead rollout). Speed: the
    expensive candidate-scoring runs only every ``cfg.fudge_interval`` tokens
    (plain sampling in between) -- a ~interval x speedup for the slow discriminator."""
    device = base.device
    interval = max(1, int(getattr(cfg, "fudge_interval", 1)))
    ids = list(prompt_ids)
    t0 = time.time()
    past = None
    next_inp = torch.tensor([ids], device=device)
    for step in range(cfg.gen_max_new):
        _, l0, past = base.step_cached(next_inp, past=past, need_hidden=False)  # cache main loop
        l0 = l0.squeeze(0)
        if step % interval != 0:                                 # plain sample between scored steps
            nxt = _sample(l0, cfg.gen_top_p, cfg.gen_temperature)
            ids.append(nxt)
            if nxt == base.eos_token_id:
                break
            next_inp = torch.tensor([[nxt]], device=device)
            continue
        base_logprob = l0.log_softmax(-1)
        K = min(fudge_topk, l0.numel())
        top_lp, top_ids = base_logprob.topk(K)
        # score each candidate by a short lookahead reward (the "future discriminator")
        cand = [ids + [v.item()] for v in top_ids]
        maxlen = len(ids) + 1
        pad = base.tokenizer.pad_token_id
        batch = torch.full((K, maxlen), pad, device=device, dtype=torch.long)
        attn = torch.zeros((K, maxlen), device=device, dtype=torch.long)
        for i, seq in enumerate(cand):
            batch[i, maxlen - len(seq):] = torch.tensor(seq, device=device)
            attn[i, maxlen - len(seq):] = 1
        gen = base.generate(batch, attention_mask=attn, max_new_tokens=fudge_lookahead,
                            do_sample=True, top_p=1.0, temperature=1.0)
        texts = [base.decode(row) for row in gen[:, maxlen:]]
        future = reward_fn(texts).to(device)                         # [K] in [0,1]
        adjusted = top_lp + lam * future.clamp_min(1e-6).log()       # reweighted logprobs
        probs = (adjusted / max(cfg.gen_temperature, 1e-6)).softmax(-1)
        nxt = top_ids[torch.multinomial(probs, 1)].item()
        ids.append(nxt)
        if nxt == base.eos_token_id:
            break
        next_inp = torch.tensor([[nxt]], device=device)
    elapsed = time.time() - t0
    n_new = len(ids) - len(prompt_ids)
    return {
        "ids": ids, "new_ids": ids[len(prompt_ids):],
        "text": base.decode(ids[len(prompt_ids):]),
        "full_text": base.decode(ids),
        "ms_per_token": 1000.0 * elapsed / max(n_new, 1),
        "mean_kl": 0.0, "mean_bnorm": 0.0,
    }
