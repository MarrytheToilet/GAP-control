"""Train the compositional controller (handbook 2.3 / 3.8).

Samples random attribute compositions (1..max_attrs, mixed soft/hard, random intensities)
and trains the controller to predict their advantage. The target is exact-linear:
    A_c = (sum_i alpha_i A_i) / (sum_i alpha_i)      (cached per-atomic advantages)
A set of combinations is *held out* from sampling and used to measure compositional
generalization. The interaction residual is regularized toward zero so single-attribute
behavior stays clean and composition defaults to the additive (provably-composing) pathway.

Usage:
    python scripts/train_compositional.py --config configs/compositional_demo.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import os
import random

import torch
import torch.nn.functional as F

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.controller import CompositionalController, build_controller
from gap_control.catalog import parse_atomics, condition_from_atomics
from gap_control.attributes import NUM_SLOTS
from gap_control import state as state_mod, metrics


def build_examples(records, atoms, cfg, rng, holdout):
    """Return (train_examples, heldout_examples). Each example: dict with rec_idx, alpha,
    A_c [K], V_c, state [STATE_DIM], ids(frozenset)."""
    by_id = {a.id: a for a in atoms}
    train, held = [], []
    from gap_control.attributes import SOFT_DIM_SET
    pos_choices = [0.5, 0.75, 1.0]
    # signed coefficients enable "advantage algebra": negative = suppress an attribute.
    # Soft dims can be negated (suppress sentiment/style/emotion); hard dims stay positive.
    signed_choices = [-1.0, -0.5, 0.5, 0.75, 1.0]
    for ri, rec in enumerate(records):
        avail = [by_id[i] for i in rec["atoms"] if i in by_id]
        # group by dim so a composition picks distinct dimensions
        for _ in range(cfg.compositions_per_prefix):
            m = rng.randint(1, min(cfg.max_attrs, len(avail)))
            # pick m atomics from distinct dims
            pool, picked, dims = list(avail), [], set()
            rng.shuffle(pool)
            for a in pool:
                if a.component.dim not in dims:
                    picked.append(a); dims.add(a.component.dim)
                if len(picked) == m:
                    break
            ids = frozenset(a.id for a in picked)
            inten = [rng.choice(signed_choices if a.component.dim in SOFT_DIM_SET else pos_choices)
                     for a in picked]
            alpha = torch.zeros(NUM_SLOTS)
            A_c = torch.zeros(rec["atoms"][picked[0].id]["A"].numel())
            V_c, wsum = 0.0, 0.0
            for a, w in zip(picked, inten):
                alpha[a.slot] = w
                A_c += w * rec["atoms"][a.id]["A"].float()
                V_c += w * rec["atoms"][a.id]["V"]
                wsum += abs(w)                                    # L1: signed-coefficient safe
            A_c /= max(wsum, 1e-6); V_c /= max(wsum, 1e-6)
            cond = condition_from_atomics(picked, inten)
            sf = state_mod.running_state(rec["gen_text"], rec["n_gen"], cond,
                                         max_new=cfg.gen_max_new)
            ex = {"rec_idx": ri, "alpha": alpha, "A_c": A_c, "V_c": V_c,
                  "state": sf, "ids": ids, "size": len(picked)}
            (held if ids in holdout else train).append(ex)
    return train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra-teacher", default=None,
                    help="additional teacher .pt to aggregate (DAgger: base + on-policy)")
    ap.add_argument("--out", default=None, help="checkpoint output path override")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = cfg.device

    blob = torch.load(cfg.teacher_path(), weights_only=False)
    records = blob["records"]
    if args.extra_teacher:
        extra = torch.load(args.extra_teacher, weights_only=False)["records"]
        records = records + extra
        print(f"[comp] aggregated DAgger teacher: {len(blob['records'])} + {len(extra)} = {len(records)}")
    atoms = parse_atomics(cfg.atomics)
    holdout = set(frozenset(c) for c in cfg.holdout_combos)
    print(f"[comp] {len(records)} prefixes, {len(atoms)} atomics, "
          f"{len(holdout)} held-out combos")

    # record-level tensors
    hidden = torch.stack([r["hidden"].float() for r in records]).to(device)
    topk_ids = torch.stack([r["topk_ids"] for r in records]).to(device)
    base_lp = torch.stack([r["base_logprob_topk"].float() for r in records]).to(device)
    H, K = hidden.size(1), topk_ids.size(1)

    train_ex, held_ex = build_examples(records, atoms, cfg, rng, holdout)
    print(f"[comp] train examples {len(train_ex)} | held-out examples {len(held_ex)}")

    def stack_ex(exs):
        return {
            "rec": torch.tensor([e["rec_idx"] for e in exs], device=device),
            "alpha": torch.stack([e["alpha"] for e in exs]).to(device),
            "A_c": torch.stack([e["A_c"] for e in exs]).to(device),
            "V_c": torch.tensor([e["V_c"] for e in exs], device=device),
            "state": torch.stack([e["state"] for e in exs]).to(device),
        }
    T = stack_ex(train_ex)
    Hh = stack_ex(held_ex) if held_ex else None

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    W = base.lm_head.weight.detach().float()
    state_dim = state_mod.STATE_DIM
    hidden_out = W.size(1)          # residual output dim = last-layer hidden (for tied W_LM)
    in_dim = H                      # controller INPUT dim (may concat multiple layers)

    caa_dirs = None
    if cfg.use_steering or cfg.controller_type == "static_caa":
        caa_dirs = torch.load(cfg.steering_path(), weights_only=False)["caa"].to(device)
        print(f"[comp] CAA directions loaded: {cfg.steering_path()}")
    controller = build_controller(
        cfg.controller_type, hidden_out, NUM_SLOTS, control_dim=cfg.control_dim,
        fuse_hidden=cfg.fuse_hidden, dropout=cfg.controller_dropout, state_dim=state_dim,
        caa_dirs=caa_dirs, linear_rank=cfg.linear_steer_rank, steer_rank=cfg.steer_rank,
        in_dim=in_dim, interaction=cfg.interaction
    ).to(device)
    print(f"[comp] in_dim={in_dim} hidden_out={hidden_out} "
          f"(mid_layer_frac={cfg.mid_layer_frac})")
    print(f"[comp] controller_type={cfg.controller_type} "
          f"params={sum(p.numel() for p in controller.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tau = cfg.tau

    def forward(idx, data, train=True):
        rec = data["rec"][idx]
        h = hidden[rec]; tk = topk_ids[rec]; blp = base_lp[rec]
        r_t, v_hat, int_norm = controller(h, data["alpha"][idx], data["state"][idx],
                                          return_aux=True)
        W_sel = W[tk]                                            # [B,K,H]
        b_active = (W_sel * r_t.unsqueeze(1)).sum(-1)            # [B,K]
        logp_theta = (blp + b_active).log_softmax(-1)
        logp_star = (blp + data["A_c"][idx] / tau).log_softmax(-1)
        p_star = logp_star.exp()
        kl_per = (p_star * (logp_star - logp_theta)).sum(-1)          # [B]
        if cfg.adv_weight:
            w = data["A_c"][idx].abs().amax(-1)                       # max|A| per example
            w = w / w.mean().clamp_min(1e-6)
            l_adv = (w * kl_per).mean()
        else:
            l_adv = kl_per.mean()
        l_val = F.mse_loss(v_hat, data["V_c"][idx])
        l_reg = (int_norm ** 2).mean()
        return l_adv, l_val, l_reg, b_active

    N = len(train_ex)
    os.makedirs(os.path.dirname(args.out or cfg.controller_path()), exist_ok=True)
    for epoch in range(cfg.epochs):
        controller.train()
        order = torch.randperm(N, device=device)
        tot = 0.0
        for s in range(0, N, cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            l_adv, l_val, l_reg, _ = forward(idx, T)
            loss = l_adv + cfg.value_lambda * l_val + cfg.interaction_lambda * l_reg
            gl = getattr(cfg, "gate_lambda", 0.0)
            if gl and getattr(controller, "caa_gate", None) is not None:
                loss = loss + gl * (controller.caa_gate ** 2).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
            opt.step(); tot += loss.item() * idx.numel()

        controller.eval()
        msg = f"epoch {epoch:02d} | train {tot/N:.4f}"
        if Hh is not None:
            with torch.no_grad():
                idx = torch.arange(len(held_ex), device=device)
                l_adv, l_val, l_reg, b = forward(idx, Hh, train=False)
                corr = sum(metrics.spearman(b[i].cpu(), (Hh["A_c"][i] / tau).cpu())
                           for i in range(len(held_ex))) / len(held_ex)
            msg += f" | HELD-OUT L_adv {l_adv:.4f} bias-adv corr {corr:.3f}"
        print(msg, flush=True)

    torch.save({"state_dict": controller.state_dict(), "config": cfg.__dict__,
                "hidden_size": hidden_out, "in_dim": in_dim, "num_slots": NUM_SLOTS, "state_dim": state_dim,
                "compositional": True, "use_steering": cfg.use_steering,
                "controller_type": cfg.controller_type, "interaction": cfg.interaction,
                "linear_steer_rank": cfg.linear_steer_rank, "steer_rank": cfg.steer_rank},
               args.out or cfg.controller_path())
    print(f"[comp] saved -> {args.out or cfg.controller_path()}")


if __name__ == "__main__":
    main()
