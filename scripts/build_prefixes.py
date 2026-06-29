"""Stage 1: build prefixes (handbook 4.1 step 3).

Sample base-model continuations from a set of neutral prompt stems and truncate them at
varied lengths to get prefixes covering early/mid/late generation. These are the s_t on
which we estimate teacher advantage.

Usage:
    python scripts/build_prefixes.py --config configs/sentiment_mvp.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM

# Neutral sentence-starter stems (PPLM/DExperts style): admit either sentiment.
DEFAULT_PROMPTS = [
    "The book", "The movie", "The food", "The city", "The weather",
    "The restaurant", "The hotel", "My experience", "The product", "The service",
    "The neighborhood", "The team", "The performance", "The meal", "The trip",
    "The painting", "The concert", "The class", "The phone", "The car",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)

    os.makedirs(os.path.dirname(cfg.prompts_path()), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.prefixes_path()), exist_ok=True)

    # use a pre-generated diverse prompt file if present (scripts/synth_prompts.py),
    # otherwise fall back to the built-in generic stems
    if os.path.exists(cfg.prompts_path()):
        prompts = [json.loads(l)["text"] for l in open(cfg.prompts_path())][:cfg.num_prompts]
        print(f"[build_prefixes] using {len(prompts)} prompts from {cfg.prompts_path()}")
    else:
        prompts = DEFAULT_PROMPTS[:cfg.num_prompts]
        with open(cfg.prompts_path(), "w") as f:
            for i, p in enumerate(prompts):
                f.write(json.dumps({"prompt_id": i, "text": p}) + "\n")

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    records = []
    for pid, prompt in enumerate(prompts):
        ids = base.encode(prompt)
        prompt_len = len(ids)
        inp = torch.tensor([ids] * cfg.prefixes_per_prompt, device=cfg.device)
        gen = base.generate(inp, max_new_tokens=cfg.prefix_gen_max_new,
                            do_sample=True, top_p=0.95, temperature=1.0)
        for row in gen:
            full = row.tolist()
            # truncate to a random prefix length within [min, max]
            lo, hi = cfg.prefix_min_len, min(cfg.prefix_max_len, len(full))
            if hi <= lo:
                cut = hi
            else:
                cut = int(torch.randint(lo, hi + 1, (1,)).item())
            prefix_ids = full[:cut]
            records.append({
                "prompt_id": pid,
                "prompt_len": prompt_len,
                "prefix_token_ids": prefix_ids,
                "prefix_text": base.decode(prefix_ids),
            })

    with open(cfg.prefixes_path(), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_prefixes] wrote {len(records)} prefixes -> {cfg.prefixes_path()}")


if __name__ == "__main__":
    main()
