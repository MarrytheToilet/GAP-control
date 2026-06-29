"""Stage 5: evaluate (handbook 3.5 / 3.6).

Scores the decoded generations and prints the main control-quality-efficiency table:
    Control(reward) up | Success up | PPL down | KL-to-base down | ms/token down
plus distinct-2. Aggregated per method.

Usage:
    python scripts/evaluate.py --config configs/sentiment_mvp.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import json
from collections import defaultdict

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.rewards import build_condition_reward
from gap_control import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.load(args.config)

    path = f"{cfg.out_dir}/{cfg.task}_decode.jsonl"
    with open(path) as f:
        gens = [json.loads(l) for l in f]

    by_method = defaultdict(list)
    for g in gens:
        by_method[g["method"]].append(g)

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    reward_fn, condition, bank = build_condition_reward(cfg, tokenizer=base.tokenizer)

    rows = []
    for method, items in by_method.items():
        texts = [it["text"] for it in items]
        r = reward_fn(texts)
        joint = condition.joint_success(texts, bank=bank, tokenizer=base.tokenizer)
        rows.append({
            "method": method,
            "reward": metrics.mean_reward(r),
            "success": metrics.control_success(r, 0.5),
            "joint": joint.mean().item(),
            "ppl": metrics.perplexity(base, texts),
            "kl": sum(it["mean_kl"] for it in items) / len(items),
            "distinct2": metrics.distinct_n(texts, 2),
            "ms_tok": sum(it["ms_per_token"] for it in items) / len(items),
            "n": len(items),
        })

    # baselines first, GAP-Control last
    order = {"prompt": 0, "instruct": 1, "fewshot": 2, "cfg": 3, "preadd": 4,
             "fudge": 5, "bon": 6, "gap": 9}
    rows.sort(key=lambda x: order.get(x["method"], 8))

    desc = ", ".join(f"{c.dim}:{c.value or c.keywords or c.length_target}"
                     for c in condition.components)
    hdr = (f"{'method':<8} {'reward↑':>8} {'succ↑':>7} {'joint↑':>7} {'PPL↓':>8} "
           f"{'KL↓':>7} {'dist2↑':>7} {'ms/tok↓':>8} {'n':>4}")
    print("\n" + "=" * len(hdr))
    print(f"GAP-Control main results — task={cfg.task} control=[{desc}]")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        print(f"{x['method']:<8} {x['reward']:>8.3f} {x['success']:>7.3f} {x['joint']:>7.3f} "
              f"{x['ppl']:>8.2f} {x['kl']:>7.3f} {x['distinct2']:>7.3f} "
              f"{x['ms_tok']:>8.2f} {x['n']:>4d}")
    print("=" * len(hdr))

    with open(f"{cfg.out_dir}/{cfg.task}_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
