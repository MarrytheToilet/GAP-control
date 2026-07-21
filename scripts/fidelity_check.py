"""Distillation fidelity: how close is the controller residual b_theta to the cached
teacher target A_c/tau, for atomic / seen-composition / unseen (held-out) composition?

Directly probes the theorem-to-system bridge (Thm 1 / Prop 2): if the unseen-composition
fidelity matches the atomic fidelity, composition error is governed by per-atomic fit,
as Prop 2 predicts. Fully offline: base LM head + cached advantages + trained controller.

Usage:
    python scripts/fidelity_check.py --config configs/_gt_g2_rt1p0_rm25_tri.yaml \
        --ckpt models/controller/flc_multi/full.pt
"""
import _bootstrap  # noqa: F401
import argparse
import random

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.controller import build_controller
from gap_control.catalog import parse_atomics
from gap_control.attributes import NUM_SLOTS

import train_compositional as tc  # reuse build_examples (identical sampling)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = cfg.device

    blob = torch.load(cfg.teacher_path(), weights_only=False)
    records = blob["records"]
    atoms = parse_atomics(cfg.atomics)
    holdout = set(frozenset(c) for c in cfg.holdout_combos)
    train_ex, held_ex = tc.build_examples(records, atoms, cfg, rng, holdout)

    hidden = torch.stack([r["hidden"].float() for r in records]).to(device)
    topk_ids = torch.stack([r["topk_ids"] for r in records]).to(device)
    base_lp = torch.stack([r["base_logprob_topk"].float() for r in records]).to(device)

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    W = base.lm_head.weight.detach().float()

    ckpt = torch.load(args.ckpt or cfg.controller_path(), weights_only=False)
    kind = ckpt.get("controller_type", "compositional")
    caa = (torch.zeros(ckpt["num_slots"], ckpt["hidden_size"])
           if (ckpt.get("use_steering") or kind == "static_caa") else None)
    controller = build_controller(
        kind, ckpt["hidden_size"], ckpt["num_slots"], control_dim=cfg.control_dim,
        fuse_hidden=cfg.fuse_hidden, dropout=cfg.controller_dropout,
        state_dim=ckpt.get("state_dim", 0), caa_dirs=caa,
        linear_rank=ckpt.get("linear_steer_rank", 0), steer_rank=ckpt.get("steer_rank", 8),
        in_dim=ckpt.get("in_dim"), interaction=ckpt.get("interaction", True)).to(device)
    controller.load_state_dict(ckpt["state_dict"])
    controller.eval()
    tau = cfg.tau

    def metrics_for(exs):
        if not exs:
            return None
        rec = torch.tensor([e["rec_idx"] for e in exs], device=device)
        alpha = torch.stack([e["alpha"] for e in exs]).to(device)
        A_c = torch.stack([e["A_c"] for e in exs]).to(device)          # [B,K]
        state = torch.stack([e["state"] for e in exs]).to(device)
        h, tk, blp = hidden[rec], topk_ids[rec], base_lp[rec]
        r_t, _, _ = controller(h, alpha, state, return_aux=True)
        b_active = (W[tk] * r_t.unsqueeze(1)).sum(-1)                   # [B,K]
        target = A_c / tau
        # KL(p* || p_theta) over the active set
        logp_theta = (blp + b_active).log_softmax(-1)
        logp_star = (blp + target).log_softmax(-1)
        kl = (logp_star.exp() * (logp_star - logp_theta)).sum(-1)      # [B]
        # cosine between predicted residual and teacher target (direction fidelity)
        cos = torch.cosine_similarity(b_active, target, dim=-1)        # [B]
        # top-1 agreement: does the tilted argmax match?
        top1 = ((blp + b_active).argmax(-1) == (blp + target).argmax(-1)).float()
        return len(exs), kl.mean().item(), cos.mean().item(), top1.mean().item()

    atomic = [e for e in train_ex if e["size"] == 1]
    seen_comp = [e for e in train_ex if e["size"] >= 2]
    groups = [("atomic (trained)", atomic),
              ("seen composition (trained)", seen_comp),
              ("UNSEEN composition (held out)", held_ex)]
    print(f"{'group':32s} {'n':>6} {'KL(p*||pθ)':>12} {'cosine':>8} {'top1-agree':>11}")
    print("-" * 74)
    for name, exs in groups:
        r = metrics_for(exs)
        if r:
            n, kl, cos, t1 = r
            print(f"{name:32s} {n:6d} {kl:12.4f} {cos:8.3f} {t1:11.3f}")


if __name__ == "__main__":
    main()
