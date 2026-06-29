"""On-policy prefix construction (DAgger-style) — addresses the train/inference mismatch.

The base teacher rolls out under pi_0, but at inference we sample under the controlled
policy pi_theta, visiting different states. Here we decode WITH the current controller under
sampled conditions, take the controlled trajectories as new prefixes, and (re-)estimate the
teacher advantage on them. The controller architecture is unchanged — only the distribution
of prefix STATES the teacher sees moves toward what the controller actually visits.

By default APPENDS to the existing prefixes file (DAgger aggregation: base + on-policy).

Usage:
    python scripts/build_prefixes_onpolicy.py --config configs/compositional_v3.yaml \
        --conditions-per-prompt 2 --append
Then re-run estimate_teacher_multi.py and train_compositional.py.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import random

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.controller import CompositionalController, GapController
from gap_control.catalog import parse_atomics, condition_from_atomics
from gap_control import decoding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="controller checkpoint to decode with (default: this config's)")
    ap.add_argument("--conditions-per-prompt", type=int, default=2)
    ap.add_argument("--truncations", type=int, default=2)
    ap.add_argument("--append", action="store_true", help="aggregate with existing prefixes (DAgger)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed + 1)

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    ckpt = torch.load(args.ckpt or cfg.controller_path(), weights_only=False)
    Ctrl = CompositionalController if ckpt.get("compositional") else GapController
    controller = Ctrl(ckpt["hidden_size"], ckpt["num_slots"], cfg.control_dim,
                      cfg.fuse_hidden, cfg.controller_dropout,
                      state_dim=ckpt.get("state_dim", 0)).to(cfg.device)
    controller.load_state_dict(ckpt["state_dict"]); controller.eval()

    atoms = parse_atomics(cfg.atomics)
    by_dim = {}
    for a in atoms:
        by_dim.setdefault(a.component.dim, []).append(a)

    prompts = [json.loads(l) for l in open(cfg.prompts_path())][:cfg.num_prompts]
    if args.limit:
        prompts = prompts[:args.limit]

    def sample_condition():
        m = rng.randint(1, min(cfg.max_attrs, len(by_dim)))
        dims = rng.sample(list(by_dim), m)
        picked = [rng.choice(by_dim[d]) for d in dims]
        inten = [rng.choice([0.5, 0.75, 1.0]) for _ in picked]
        return condition_from_atomics(picked, inten)

    records = []
    for p in prompts:
        pid_ids = base.encode(p["text"])
        plen = len(pid_ids)
        for _ in range(args.conditions_per_prompt):
            cond = sample_condition()
            alpha = cond.to_alpha().to(cfg.device)
            out = decoding.gap_decode(base, controller, pid_ids, alpha=alpha, cfg=cfg,
                                      condition=cond)
            full = out["ids"]
            gen_len = len(full) - plen
            if gen_len < 2:
                continue
            for _ in range(args.truncations):
                cut = plen + rng.randint(1, gen_len)
                records.append({"prompt_id": p["prompt_id"], "prompt_len": plen,
                                "prefix_token_ids": full[:cut],
                                "prefix_text": base.decode(full[:cut])})
        print(f"  prompt {p['prompt_id']} done ({len(records)} prefixes)", flush=True)

    mode = "a" if args.append else "w"
    with open(cfg.prefixes_path(), mode) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_total = sum(1 for _ in open(cfg.prefixes_path()))
    print(f"[onpolicy] +{len(records)} on-policy prefixes "
          f"({'appended' if args.append else 'written'}) -> total {n_total}")


if __name__ == "__main__":
    main()
