"""Stage 4: decode with GAP-Control and baselines (handbook 4.1 step 6).

Generates continuations for the eval prompts under each requested method and writes one
JSONL with text + efficiency/perturbation diagnostics per sample.

Usage:
    python scripts/decode_gap_control.py --config configs/sentiment_mvp.yaml \
        --methods gap,prompt,fudge [--limit N]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.controller import GapController, CompositionalController
from gap_control import decoding
from gap_control.rewards import build_condition_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", default="gap,prompt", help="comma list: gap,prompt,fudge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompts", default=None, help="override prompts file (e.g. an unseen eval set)")
    ap.add_argument("--ckpt", default=None, help="controller checkpoint override")
    ap.add_argument("--out", default=None, help="output jsonl path override")
    ap.add_argument("--seed", type=int, default=None, help="decode seed override (for variance)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(args.seed if args.seed is not None else cfg.seed)
    methods = args.methods.split(",")

    with open(args.prompts or cfg.prompts_path()) as f:
        prompts = [json.loads(l) for l in f]
    if cfg.num_eval_prompts:
        prompts = prompts[:cfg.num_eval_prompts]
    if args.limit:
        prompts = prompts[:args.limit]

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)

    condition = cfg.condition()
    alpha = condition.to_alpha().to(cfg.device)

    controller = None
    if "gap" in methods:
        ckpt = torch.load(args.ckpt or cfg.controller_path(), weights_only=False)
        if ckpt.get("compositional"):
            kind = ckpt.get("controller_type", "compositional")
            caa = (torch.zeros(ckpt["num_slots"], ckpt["hidden_size"])
                   if (ckpt.get("use_steering") or kind == "static_caa") else None)
            from gap_control.controller import build_controller
            controller = build_controller(
                kind, ckpt["hidden_size"], ckpt["num_slots"], control_dim=cfg.control_dim,
                fuse_hidden=cfg.fuse_hidden, dropout=cfg.controller_dropout,
                state_dim=ckpt.get("state_dim", 0), caa_dirs=caa,
                linear_rank=ckpt.get("linear_steer_rank", 0),
                steer_rank=ckpt.get("steer_rank", 8),
                in_dim=ckpt.get("in_dim"),
                interaction=ckpt.get("interaction", True)).to(cfg.device)
        else:
            controller = GapController(ckpt["hidden_size"], ckpt["num_slots"],
                                       cfg.control_dim, cfg.fuse_hidden, cfg.controller_dropout,
                                       state_dim=ckpt.get("state_dim", 0)).to(cfg.device)
        controller.load_state_dict(ckpt["state_dict"])
        controller.eval()

    from gap_control import attributes as A
    instruction = A.describe(condition)
    anti = A.describe_anti(condition)

    reward_fn = None
    if {"fudge", "bon"} & set(methods):
        reward_fn, _, _ = build_condition_reward(cfg, tokenizer=base.tokenizer)

    out_path = args.out or f"{cfg.out_dir}/{cfg.task}_decode.jsonl"
    os.makedirs(cfg.out_dir, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for p in prompts:
            pid = base.encode(p["text"])
            ptext = p["text"]
            for s in range(cfg.samples_per_prompt):
                for m in methods:
                    if m == "gap":
                        out = decoding.gap_decode(base, controller, pid, alpha=alpha,
                                                  cfg=cfg, condition=condition)
                    elif m == "base":          # uncontrolled continuation (reference)
                        out = decoding.prompt_only_decode(base, pid, cfg=cfg)
                    elif m == "prompting":     # instruction-based control (merges old prompt/instruct)
                        out = decoding.instructed_decode(base, ptext, instruction, cfg=cfg)
                    elif m == "fudge":
                        out = decoding.fudge_like_decode(base, reward_fn, pid, cfg=cfg)
                    elif m == "cfg":
                        out = decoding.cfg_decode(base, ptext, instruction, cfg=cfg)
                    elif m == "preadd":
                        out = decoding.preadd_decode(base, ptext, instruction, anti, cfg=cfg)
                    elif m == "bon":
                        out = decoding.best_of_n_decode(base, reward_fn, pid, cfg=cfg)
                    else:
                        raise ValueError(f"unknown method {m}")
                    rec = {"method": m, "prompt_id": p["prompt_id"],
                           "prompt": p["text"], "sample": s,
                           "text": out["text"], "full_text": out["full_text"],
                           "ms_per_token": out["ms_per_token"],
                           "mean_kl": out["mean_kl"], "mean_bnorm": out["mean_bnorm"]}
                    if "trajectory" in out:
                        rec["trajectory"] = out["trajectory"]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            print(f"  prompt {p['prompt_id']} done")
    print(f"[decode] wrote {n} generations ({methods}) -> {out_path}")


if __name__ == "__main__":
    main()
