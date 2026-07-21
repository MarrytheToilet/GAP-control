"""Mechanism decomposition (review-v2 Q1): connect weak POINTWISE fidelity to large
DECODE-level gains.

Hypothesis: the decision-relevant signal is concentrated. On near-deterministic steps
(low base uncertainty u = 1 - max pi0) the teacher tilt rarely changes the argmax and
carries mostly Monte-Carlo noise -- these steps dilute global cosine/KL while being
behaviorally inert (the gate also suppresses intervention there, factor u^gamma). On
high-uncertainty steps the tilt DOES flip decisions, and what matters is whether the
controller captures those flips.

For each held-out prefix and atomic attribute (alpha=+1), stratify by u quartile:
  * flip rate      P(argmax(l0 + A/tau) != argmax(l0))          [teacher changes decision]
  * capture rate   P(ctrl argmax == teacher argmax | teacher flips)
  * base rate      P(ctrl argmax == teacher argmax | no flip)   [sanity, ~1 trivially]
  * mean |A|, mean cosine(b, A)                                  [signal strength / fit]

Usage:  python scripts/mechanism_analysis.py --ckpt models/controller/flc_multi_n48/full.pt
"""
import _bootstrap  # noqa: F401
import argparse

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.controller import build_controller
from gap_control.catalog import parse_atomics
from gap_control.attributes import NUM_SLOTS, SLOTS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flc_heldout.yaml")
    ap.add_argument("--ckpt", default="models/controller/flc_multi_n48/full.pt")
    ap.add_argument("--ref-config", default="configs/flc_heldout2.yaml",
                    help="independent teacher re-estimate for the measurement ceiling")
    ap.add_argument("--dump-json", default=None, help="write per-example records here")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    device = cfg.device

    blob = torch.load(cfg.teacher_path(), weights_only=False)
    records = blob["records"]
    ref_records = torch.load(Config.load(args.ref_config).teacher_path(),
                             weights_only=False)["records"] if args.ref_config else None
    atoms = parse_atomics(cfg.atomics)
    tau = cfg.tau

    base = BaseLM(cfg.base_model, device, cfg.dtype)
    W = base.lm_head.weight.detach().float()

    ckpt = torch.load(args.ckpt, weights_only=False)
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

    # one example per (record, atomic) with alpha = +1
    ex = []  # dicts: u, flip, captured (if flip), agree_noflip (if not), cos, absA
    for ridx, r in enumerate(records):
        h = r["hidden"].float().to(device).unsqueeze(0)
        tk = torch.as_tensor(r["topk_ids"]).long().to(device)
        blp = r["base_logprob_topk"].float().to(device)
        u = 1.0 - blp.exp().max().item()          # base uncertainty at this state
        base_arg = blp.argmax().item()
        for a in atoms:
            A = r["atoms"][a.id]["A"].float().to(device)
            alpha = torch.zeros(1, NUM_SLOTS, device=device)
            alpha[0, a.slot] = 1.0
            state = torch.zeros(1, ckpt.get("state_dim", 0), device=device)
            r_t, _, _ = controller(h, alpha, state, return_aux=True)
            b = (W[tk] * r_t.squeeze(0).unsqueeze(0)).sum(-1)
            t_arg = (blp + A / tau).argmax().item()
            c_arg = (blp + b).argmax().item()
            flip = (t_arg != base_arg)
            cA, cB = A - A.mean(), b - b.mean()
            d = dict(u=u, flip=flip, match=(c_arg == t_arg),
                     cos=torch.cosine_similarity(cA, cB, dim=0).item(),
                     absA=A.abs().mean().item())
            if ref_records is not None:
                A2 = ref_records[ridx]["atoms"][a.id]["A"].float().to(device)
                d["ref_match"] = ((blp + A2 / tau).argmax().item() == t_arg)
            ex.append(d)

    us_t = torch.tensor([e["u"] for e in ex])
    qs = torch.quantile(us_t, torch.tensor([0.25, 0.5, 0.75]))
    print(f"u quartile cuts: {[round(q.item(),3) for q in qs]}   n={len(ex)} (state,attr) pairs")
    print(f"{'u bucket':13s} {'n':>6} {'flip%':>7} {'capture%':>9} {'agree|noflip%':>14} "
          f"{'cos(b,A)':>9} {'mean|A|':>8}")
    print("-" * 72)
    bounds = [(-1e9, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]
    names = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    for nm, (lo, hi) in zip(names, bounds):
        b_ex = [e for e in ex if lo < e["u"] <= hi]
        fl = [e for e in b_ex if e["flip"]]
        nf = [e for e in b_ex if not e["flip"]]
        cap = 100 * sum(e["match"] for e in fl) / max(len(fl), 1)
        anf = 100 * sum(e["match"] for e in nf) / max(len(nf), 1)
        print(f"{nm:13s} {len(b_ex):6d} {100*len(fl)/max(len(b_ex),1):7.1f} {cap:9.1f} "
              f"{anf:14.1f} {sum(e['cos'] for e in b_ex)/max(len(b_ex),1):9.3f} "
              f"{sum(e['absA'] for e in b_ex)/max(len(b_ex),1):8.3f}")
    if args.dump_json:
        import json
        json.dump([{k: (bool(v) if isinstance(v, bool) else v) for k, v in e.items()}
                   for e in ex], open(args.dump_json, "w"))
        print(f"dumped {len(ex)} examples -> {args.dump_json}")
    fl = [e for e in ex if e["flip"]]
    nf = [e for e in ex if not e["flip"]]
    print(f"\noverall: teacher flips argmax on {100*len(fl)/len(ex):.1f}% of (state,attr); "
          f"controller captures the flipped argmax {100*sum(e['match'] for e in fl)/max(len(fl),1):.1f}% "
          f"(chance {100.0/32:.1f}%); agree|noflip {100*sum(e['match'] for e in nf)/max(len(nf),1):.1f}%")
    if ref_records is not None:
        ceil = 100 * sum(e['ref_match'] for e in fl) / max(len(fl), 1)
        print(f"MEASUREMENT CEILING: an independent teacher re-estimate captures the "
              f"flipped argmax {ceil:.1f}% of the time -> controller reaches "
              f"{100*sum(e['match'] for e in fl)/max(len(fl),1)/max(ceil,1e-9)*100:.0f}% of ceiling")


if __name__ == "__main__":
    main()
