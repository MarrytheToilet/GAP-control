"""Add-an-attribute wall-clock demo (paper claim: adaptation is O(m) once, not O(new
attribute) re-rollout).

The shared rollouts are the reusable artifact. Adding attribute m+1 to a deployed cache
means rescoring the ALREADY-GENERATED continuations with the new reward -- no new rollout.
We measure, on a subset of prefixes:
  * T_rollout+score(m) : build the shared-rollout cache once (the one-time cost)
  * T_score1           : rescore the stored continuations with ONE new reward
  * (report) controller fold-in is a short finetune, not shown here
and contrast with the cost other methods pay to add an attribute (new steer / new FUDGE
classifier / N x at every request for Best-of-N).

Usage:  USE_CUDA=1 python scripts/add_attr_demo.py --config configs/flc_multi.yaml --n 60
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.catalog import parse_atomics
from gap_control.attributes import ControlCondition, SOFT_DIM_SET
from gap_control.classifiers import SoftClassifierBank
from gap_control.teacher import estimate_prefix_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flc_multi.yaml")
    ap.add_argument("--n", type=int, default=60, help="prefixes to time on")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, cfg.device)

    atoms = parse_atomics(cfg.atomics)
    prefixes = [json.loads(l) for l in open(cfg.prefixes_path())][:args.n]

    # capture the shared rollout continuations via an instrumented reward_fn
    captured = {}
    def make_reward(a, idx):
        cond = ControlCondition([a.component])
        def fn(texts):
            captured[idx] = texts            # same texts for every atom (shared rollout)
            return cond.reward(texts, bank=bank, tokenizer=base.tokenizer)
        return fn
    reward_fns = [make_reward(a, i) for i, a in enumerate(atoms)]

    # ---- one-time cost: build the shared-rollout cache (rollout + score m atoms) ----
    torch.cuda.synchronize() if cfg.device == "cuda" else None
    t0 = time.time()
    all_texts = []
    for p in prefixes:
        captured.clear()
        rec = estimate_prefix_multi(
            base, atoms, reward_fns, p["prefix_token_ids"],
            topk=cfg.topk, rollout_samples=cfg.rollout_samples,
            rollout_max_new=cfg.rollout_max_new, rollout_temperature=cfg.rollout_temperature,
            rollout_batch_size=cfg.rollout_batch_size, tau=cfg.tau,
            prompt_len=p.get("prompt_len", 0), extra_ids=[], hidden_fracs=cfg.hidden_fracs())
        # stash this prefix's shared continuations (captured by reward_fn 0)
        all_texts.append(captured[0])
    torch.cuda.synchronize() if cfg.device == "cuda" else None
    t_cache = time.time() - t0

    # ---- marginal cost of adding attribute m+1: rescore the STORED continuations ----
    # new attribute = a soft class scored by an existing head (no re-rollout, no new gen)
    new_attr = parse_atomics([{"dim": "emotion", "value": "fear"}])[0]
    new_cond = ControlCondition([new_attr.component])
    torch.cuda.synchronize() if cfg.device == "cuda" else None
    t1 = time.time()
    for texts in all_texts:
        _ = new_cond.reward(texts, bank=bank, tokenizer=base.tokenizer)
    torch.cuda.synchronize() if cfg.device == "cuda" else None
    t_score1 = time.time() - t1

    per_prefix_cache = t_cache / len(prefixes)
    per_prefix_score = t_score1 / len(prefixes)
    full_240 = per_prefix_cache * 240
    add_240 = per_prefix_score * 240
    print(f"\n=== add-an-attribute wall-clock (Falcon, {args.n} prefixes, n={cfg.rollout_samples}) ===")
    print(f"one-time shared-rollout cache (rollout + score {len(atoms)} atoms):")
    print(f"    {per_prefix_cache:.2f} s/prefix  ->  {full_240/60:.1f} min for 240 prefixes")
    print(f"add attribute m+1 by rescoring STORED continuations (no re-rollout):")
    print(f"    {per_prefix_score:.3f} s/prefix  ->  {add_240:.1f} s for 240 prefixes")
    print(f"    speedup vs re-rolling the cache: {t_cache/max(t_score1,1e-6):.0f}x")
    print(f"    (the rollout is the reusable artifact; adding an attribute is a rescore +")
    print(f"     a short controller fold-in, vs a new steer / new FUDGE classifier / N x per request)")


if __name__ == "__main__":
    main()
