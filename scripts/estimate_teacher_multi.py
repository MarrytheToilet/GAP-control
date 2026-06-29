"""Compositional teacher: cache per-atomic advantages from one shared rollout set.

For every prefix, rolls out the shared completions once and evaluates every atomic
attribute's reward on them, storing per-atomic A/V. Any composition's teacher target is then
the linear combination of these (handbook 2.3; exact for base-policy advantage).

Usage:
    python scripts/estimate_teacher_multi.py --config configs/compositional_demo.yaml [--limit N]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import time

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.catalog import parse_atomics
from gap_control.attributes import ControlCondition, SOFT_DIM_SET
from gap_control.teacher import estimate_prefix_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)

    with open(cfg.prefixes_path()) as f:
        prefixes = [json.loads(l) for l in f]
    if args.limit:
        prefixes = prefixes[:args.limit]

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    atoms = parse_atomics(cfg.atomics)
    print(f"[teacher-multi] {len(atoms)} atomics: {[a.id for a in atoms]}")

    # bank must cover every soft dim in the ATOMIC POOL (not just the decode condition)
    soft_in_atoms = {a.component.dim for a in atoms if a.component.dim in SOFT_DIM_SET}
    bank = None
    if soft_in_atoms:
        if cfg.soft_reward_backend == "classifier":
            from gap_control.classifiers import SoftClassifierBank
            bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, cfg.device)
            missing = [d for d in soft_in_atoms if d not in bank.available()]
            if missing:
                from gap_control.rewards import JudgeBank
                print(f"[teacher-multi] no classifier for {missing}; using llm_judge")
                bank = JudgeBank(cfg.reward_judge_model, cfg.device)
        else:
            from gap_control.rewards import JudgeBank
            bank = JudgeBank(cfg.reward_judge_model, cfg.device)
        print(f"[teacher-multi] soft reward bank: {soft_in_atoms}")
    reward_fns = []
    for a in atoms:
        cond = ControlCondition([a.component])
        reward_fns.append(lambda texts, c=cond: c.reward(texts, bank=bank, tokenizer=base.tokenizer))

    # keyword tokens to force into the shared active set
    extra_ids = []
    for a in atoms:
        if a.keywords:
            extra_ids += ControlCondition([a.component]).keyword_active_ids(base.tokenizer)
    extra_ids = sorted(set(extra_ids))

    records, t0 = [], time.time()
    for i, p in enumerate(prefixes):
        rec = estimate_prefix_multi(
            base, atoms, reward_fns, p["prefix_token_ids"],
            topk=cfg.topk, rollout_samples=cfg.rollout_samples,
            rollout_max_new=cfg.rollout_max_new, rollout_temperature=cfg.rollout_temperature,
            rollout_batch_size=cfg.rollout_batch_size, tau=cfg.tau,
            prompt_len=p.get("prompt_len", 0), extra_ids=extra_ids,
            hidden_fracs=cfg.hidden_fracs())
        rec["prompt_id"] = p.get("prompt_id", -1)
        records.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(prefixes):
            print(f"  {i+1}/{len(prefixes)} ({(i+1)/(time.time()-t0):.2f} prefix/s)", flush=True)

    os.makedirs(os.path.dirname(cfg.teacher_path()), exist_ok=True)
    torch.save({"records": records, "atomics": cfg.atomics, "config": cfg.__dict__},
               cfg.teacher_path())
    # health: spread of each atomic's advantage
    print(f"[teacher-multi] saved {len(records)} records -> {cfg.teacher_path()}")
    for a in atoms:
        A = torch.cat([r["atoms"][a.id]["A"].float() for r in records])
        print(f"  {a.id:24s} |A|max={A.abs().max():.3f} std={A.std():.4f}")


if __name__ == "__main__":
    main()
