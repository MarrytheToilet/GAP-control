"""Rollout teacher advantage (handbook 2.7 / 2.8).

For each prefix s_t we estimate, for every candidate token v in the active set
S_t = TopK(pi_0):
    Q_c(s_t, v) = E[ R_c(y) | prefix = s_t + v ]
by Monte-Carlo rollout: append v, sample `rollout_samples` completions with the frozen
base model, score them with the reward, average.

Then the centered advantage and KL-optimal teacher distribution:
    A_c(v)  = Q_c(v) - sum_{u in S_t} pi_0(u) Q_c(u)
    p*(v)  proportional to  pi_0(v) * exp( A_c(v) / tau )

We also record the value target  V* = sum_{u in S_t} pi_0(u) Q_c(u)  for the value head.

This is the expensive offline stage; everything is batched across candidate*rollout
sequences. Each record stores enough to train the controller without re-running the base
model (handbook 4.4 schema): hidden state, top-k ids, base logprobs, Q, A, V*.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch

from .base_lm import BaseLM
from . import state as state_mod


@torch.no_grad()
def estimate_prefix(
    base: BaseLM,
    reward_fn: Callable[[List[str]], torch.Tensor],
    prefix_ids: List[int],
    *,
    topk: int,
    rollout_samples: int,
    rollout_max_new: int,
    rollout_temperature: float,
    rollout_batch_size: int,
    tau: float,
    alpha: torch.Tensor,
    condition=None,
    max_new: int = 40,
    prompt_len: int = 0,
    extra_ids: Optional[List[int]] = None,
):
    """Return a teacher record dict for one prefix, or None if degenerate."""
    device = base.device
    ids = torch.tensor(prefix_ids, device=device).unsqueeze(0)          # [1, T]
    h_t, logits = base.step(ids)                                        # [1,H], [1,V]
    h_t = h_t.squeeze(0).half().cpu()                                   # store fp16

    logprobs = logits.log_softmax(-1).squeeze(0)                        # [V]
    topv = min(topk, logprobs.numel())
    _, topk_ids = logprobs.topk(topv)
    # active set = TopK(pi_0), with required keyword tokens forced in (uniform size K=topv)
    active = topk_ids.tolist()
    extra_new = [e for e in (extra_ids or []) if e not in set(active)]
    if extra_new:
        active = active[: topv - len(extra_new)] + extra_new
    active_ids = torch.tensor(active, device=device)
    base_logprob_topk = logprobs[active_ids]                            # [K]
    topk_ids_list = active

    # build candidate prefixes: s_t + v  for each v in S_t, repeated rollout_samples times
    cand = []
    for v in topk_ids_list:
        cand.extend([prefix_ids + [v]] * rollout_samples)
    # pad-left into a batch and generate in chunks
    rewards_per_cand = torch.zeros(len(topk_ids_list), device="cpu")
    counts = torch.zeros(len(topk_ids_list), device="cpu")
    pad = base.tokenizer.pad_token_id
    maxlen = len(prefix_ids) + 1

    for start in range(0, len(cand), rollout_batch_size):
        chunk = cand[start:start + rollout_batch_size]
        batch = torch.full((len(chunk), maxlen), pad, device=device, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), device=device, dtype=torch.long)
        for i, seq in enumerate(chunk):
            batch[i, maxlen - len(seq):] = torch.tensor(seq, device=device)
            attn[i, maxlen - len(seq):] = 1
        gen = base.generate(batch, attention_mask=attn,
                            max_new_tokens=rollout_max_new, do_sample=True,
                            top_p=1.0, temperature=rollout_temperature)
        new = gen[:, maxlen:]
        texts = [base.decode(row) for row in new]
        r = reward_fn(texts)                                            # [chunk]
        # scatter back to candidate index
        for i in range(len(chunk)):
            ci = (start + i) // rollout_samples
            rewards_per_cand[ci] += r[i].item()
            counts[ci] += 1

    # live running-state feature from the generated portion of this prefix (dynamic signal)
    gen_ids = prefix_ids[prompt_len:]
    state_feat = (state_mod.running_state(base.decode(gen_ids), len(gen_ids), condition,
                                          max_new=max_new)
                  if condition is not None else torch.zeros(state_mod.STATE_DIM))

    Q = rewards_per_cand / counts.clamp_min(1)                          # [K]
    p0 = base_logprob_topk.exp().cpu()                                  # [K]
    p0n = p0 / p0.sum().clamp_min(1e-12)                                # renorm over S_t
    v_star = (p0n * Q).sum().item()                                     # value target
    A = Q - v_star                                                      # centered advantage

    return {
        "prefix_token_ids": prefix_ids,
        "hidden": h_t,                                  # [H] fp16
        "topk_ids": torch.tensor(topk_ids_list),        # [K]
        "base_logprob_topk": base_logprob_topk.cpu(),   # [K]
        "Q_topk": Q,                                     # [K]
        "A_topk": A,                                     # [K]
        "value_target": v_star,                         # scalar
        "alpha": alpha.cpu(),                            # [NUM_SLOTS] control vector
        "state_feat": state_feat,                        # [STATE_DIM] live satisfaction
        "tau": tau,
    }


@torch.no_grad()
def estimate_prefix_multi(
    base: BaseLM,
    atoms,                       # list[catalog.Atomic]
    reward_fns,                  # list of callables texts->[N], aligned with atoms
    prefix_ids: List[int],
    *,
    topk: int,
    rollout_samples: int,
    rollout_max_new: int,
    rollout_temperature: float,
    rollout_batch_size: int,
    tau: float,
    prompt_len: int = 0,
    extra_ids: Optional[List[int]] = None,
    hidden_fracs=(1.0,),
):
    """Cache per-atomic advantage from ONE shared rollout set (compositional teacher).

    Because R_c = sum_i alpha_i R_i on the same rollout y, we estimate Q_i for every atomic
    on the shared completions; any composition's advantage is then sum_i alpha_i A_i.
    """
    device = base.device
    ids = torch.tensor(prefix_ids, device=device).unsqueeze(0)
    h_t, logits = base.step(ids, hidden_fracs=hidden_fracs)
    h_t = h_t.squeeze(0).half().cpu()
    logprobs = logits.log_softmax(-1).squeeze(0)
    topv = min(topk, logprobs.numel())
    _, topk_ids = logprobs.topk(topv)
    active = topk_ids.tolist()
    extra_new = [e for e in (extra_ids or []) if e not in set(active)]
    if extra_new:
        active = active[: topv - len(extra_new)] + extra_new
    active_ids = torch.tensor(active, device=device)
    base_logprob_topk = logprobs[active_ids].cpu()
    K = len(active)

    # shared rollouts: each candidate token x rollout_samples completions
    cand = []
    for v in active:
        cand.extend([prefix_ids + [v]] * rollout_samples)
    pad = base.tokenizer.pad_token_id
    maxlen = len(prefix_ids) + 1
    all_texts = [None] * len(cand)
    for start in range(0, len(cand), rollout_batch_size):
        chunk = cand[start:start + rollout_batch_size]
        batch = torch.full((len(chunk), maxlen), pad, device=device, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), device=device, dtype=torch.long)
        for i, seq in enumerate(chunk):
            batch[i, maxlen - len(seq):] = torch.tensor(seq, device=device)
            attn[i, maxlen - len(seq):] = 1
        gen = base.generate(batch, attention_mask=attn, max_new_tokens=rollout_max_new,
                            do_sample=True, top_p=1.0, temperature=rollout_temperature)
        for i, row in enumerate(gen[:, maxlen:]):
            all_texts[start + i] = base.decode(row)

    p0 = base_logprob_topk.exp()
    p0n = p0 / p0.sum().clamp_min(1e-12)

    atom_records = {}
    for atom, rfn in zip(atoms, reward_fns):
        r = rfn(all_texts).reshape(K, rollout_samples)         # [K, n]
        Q = r.mean(dim=1)                                       # [K]
        v_star = (p0n * Q).sum().item()
        A = Q - v_star
        atom_records[atom.id] = {"slot": atom.slot, "A": A.half(),
                                 "V": v_star, "keywords": atom.keywords}

    gen_ids = prefix_ids[prompt_len:]
    return {
        "prefix_token_ids": prefix_ids,
        "hidden": h_t,
        "topk_ids": torch.tensor(active),
        "base_logprob_topk": base_logprob_topk,
        "prompt_len": prompt_len,
        "gen_text": base.decode(gen_ids),
        "n_gen": len(gen_ids),
        "atoms": atom_records,
        "tau": tau,
    }
