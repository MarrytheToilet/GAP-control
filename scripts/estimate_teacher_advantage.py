"""Stage 2: estimate teacher advantage (handbook 4.1 step 4).

For every prefix, rollout each top-k candidate token and build the centered advantage
A_c and the value target V*. Saves one tensor file (handbook 4.4 schema).

Usage:
    python scripts/estimate_teacher_advantage.py --config configs/sentiment_mvp.yaml [--limit N]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import time

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.rewards import build_condition_reward
from gap_control.teacher import estimate_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #prefixes (0 = all)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)

    with open(cfg.prefixes_path()) as f:
        prefixes = [json.loads(l) for l in f]
    if args.limit:
        prefixes = prefixes[:args.limit]

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    reward_fn, condition, _ = build_condition_reward(cfg, tokenizer=base.tokenizer)
    alpha = condition.to_alpha()
    extra_ids = condition.keyword_active_ids(base.tokenizer)   # force keyword into active set
    desc = ", ".join(f"{c.dim}:{c.value or c.keywords or c.length_target}"
                     for c in condition.components)
    print(f"[teacher] base={cfg.base_model} control=[{desc}] "
          f"prefixes={len(prefixes)} K={cfg.topk} n={cfg.rollout_samples}")

    records, t0 = [], time.time()
    for i, p in enumerate(prefixes):
        rec = estimate_prefix(
            base, reward_fn, p["prefix_token_ids"],
            topk=cfg.topk, rollout_samples=cfg.rollout_samples,
            rollout_max_new=cfg.rollout_max_new,
            rollout_temperature=cfg.rollout_temperature,
            rollout_batch_size=cfg.rollout_batch_size,
            tau=cfg.tau, alpha=alpha,
            condition=condition, max_new=cfg.gen_max_new,
            prompt_len=p.get("prompt_len", 0), extra_ids=extra_ids,
        )
        if rec is not None:
            rec["prompt_id"] = p.get("prompt_id", -1)
            records.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(prefixes):
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(prefixes)}  ({rate:.2f} prefix/s)")

    os.makedirs(os.path.dirname(cfg.teacher_path()), exist_ok=True)
    torch.save({"records": records, "config": cfg.__dict__}, cfg.teacher_path())

    # quick health check: spread of advantage = is there signal to learn?
    A = torch.cat([r["A_topk"] for r in records])
    print(f"[teacher] saved {len(records)} records -> {cfg.teacher_path()}")
    print(f"[teacher] A stats: mean={A.mean():.4f} std={A.std():.4f} "
          f"|A|max={A.abs().max():.4f}  (std>0 => there is advantage signal)")


if __name__ == "__main__":
    main()
