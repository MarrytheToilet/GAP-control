"""MC-noise ceiling for the held-out-prefix fidelity measurement.

The 'ground truth' advantage on held-out prefixes is itself a Monte-Carlo estimate
(n=12 rollouts). Two independent estimates of the SAME quantity on the SAME prefixes
(seed 0 vs seed 1) bound what ANY predictor can achieve against one noisy reference:
their mutual cosine / top-1 agreement is the ceiling, and a controller's measured
cosine c against one estimate corresponds to a noise-corrected fidelity of roughly
c / sqrt(ceiling) against the true advantage.

Usage:
    python scripts/noise_ceiling.py --a configs/flc_heldout.yaml --b configs/flc_heldout2.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import random

import torch

from gap_control.config import Config
from gap_control.catalog import parse_atomics

import train_compositional as tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="configs/flc_heldout.yaml")
    ap.add_argument("--b", default="configs/flc_heldout2.yaml")
    args = ap.parse_args()
    cfg = Config.load(args.a)

    blob_a = torch.load(Config.load(args.a).teacher_path(), weights_only=False)
    blob_b = torch.load(Config.load(args.b).teacher_path(), weights_only=False)
    ra, rb = blob_a["records"], blob_b["records"]
    assert len(ra) == len(rb), "record count mismatch"
    # sanity: identical prefixes & active sets
    same_prefix = all(list(x["prefix_token_ids"]) == list(y["prefix_token_ids"]) for x, y in zip(ra, rb))
    same_topk = all(torch.as_tensor(x["topk_ids"]).equal(torch.as_tensor(y["topk_ids"])) for x, y in zip(ra, rb))
    print(f"[align] prefixes identical: {same_prefix}   active sets identical: {same_topk}")

    atoms = parse_atomics(cfg.atomics)
    holdout = set(frozenset(c) for c in cfg.holdout_combos)
    # identical composition sampling on both blobs (same rng seed)
    tr_a, ho_a = tc.build_examples(ra, atoms, cfg, random.Random(0), holdout)
    tr_b, ho_b = tc.build_examples(rb, atoms, cfg, random.Random(0), holdout)

    tau = cfg.tau

    def ceiling(ea, eb):
        if not ea:
            return None
        assert len(ea) == len(eb)
        A0 = torch.stack([e["A_c"] for e in ea]).float()
        A1 = torch.stack([e["A_c"] for e in eb]).float()
        blp = torch.stack([torch.tensor(0.0)])  # placeholder
        # use each example's own base logprobs (identical across seeds; take from ea rec)
        cos = torch.cosine_similarity(A0, A1, dim=-1)
        # tilted top-1 agreement needs base logprobs per record
        recs = [e["rec_idx"] for e in ea]
        blp = torch.stack([ra[i]["base_logprob_topk"].float() for i in recs])
        t0 = (blp + A0 / tau).argmax(-1)
        t1 = (blp + A1 / tau).argmax(-1)
        top1 = (t0 == t1).float().mean().item()
        # symmetric KL between the two tilted distributions
        lp0 = (blp + A0 / tau).log_softmax(-1)
        lp1 = (blp + A1 / tau).log_softmax(-1)
        kl = 0.5 * ((lp0.exp() * (lp0 - lp1)).sum(-1) + (lp1.exp() * (lp1 - lp0)).sum(-1))
        return len(ea), kl.mean().item(), cos.mean().item(), top1

    atomic_a = [e for e in tr_a if e["size"] == 1]; atomic_b = [e for e in tr_b if e["size"] == 1]
    comp_a = [e for e in tr_a if e["size"] >= 2];   comp_b = [e for e in tr_b if e["size"] >= 2]
    groups = [("atomic", atomic_a, atomic_b),
              ("composition (2-3 attrs)", comp_a, comp_b),
              ("held-out combos", ho_a, ho_b)]
    print(f"\nTEACHER-vs-TEACHER (independent MC estimates, n={cfg.rollout_samples} each)")
    print(f"{'group':26s} {'n':>6} {'symKL':>9} {'cosine':>8} {'top1-agree':>11}")
    print("-" * 66)
    for name, ea, eb in groups:
        r = ceiling(ea, eb)
        if r:
            n, kl, cos, t1 = r
            print(f"{name:26s} {n:6d} {kl:9.4f} {cos:8.3f} {t1:11.3f}")
    print("\nInterpretation: these values are the measurement ceiling. A controller's")
    print("cosine c against ONE noisy estimate ~ c/sqrt(ceiling_cos) against the truth.")


if __name__ == "__main__":
    main()
