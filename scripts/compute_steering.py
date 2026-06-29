"""Compute CAA-style steering directions per soft attribute in the BASE model's hidden space.

Two confound-reduction techniques (combinable):
  * --pairs      : use CONTRASTIVE PAIRS (data/pairs/{dim}.jsonl) so per-class hidden means
                   differ only in the attribute (cancels topic/length confounds). Direction
                   for class c = mean_i( h(c_i) - mean_{o!=c} h(o_i) ).
  * --nullspace R: AlphaSteer-style null-space projection (Sheng et al. 2026). Project each
                   direction orthogonal to the top-R principal "benign/fluency" subspace of
                   normal generation activations, so steering disturbs normal generation
                   minimally (preserves PPL) while keeping the attribute signal.

Without --pairs, falls back to unpaired class-mean differences (data/synth).
Saves models/steering/<base>.pt -> {"caa": [NUM_SLOTS, H], ...}.

Usage:
    python scripts/compute_steering.py --config configs/compositional_v3op.yaml --pairs --nullspace 64
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import re

import torch

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.attributes import SOFT_DIMS, SLOTS, NUM_SLOTS


@torch.no_grad()
def hidden_batch(base, texts, batch_size=32, max_len=64):
    """Last-token hidden state for each text. [N, H]."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t if t.strip() else " " for t in texts[i:i + batch_size]]
        enc = base.tokenizer(chunk, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_len).to(base.device)
        h, _ = base.step(enc.input_ids, enc.attention_mask)
        out.append(h.cpu())
    return torch.cat(out)


def paired_directions(base, dim, classes):
    """Directions from contrastive pairs: per item, h(c) - mean(others). [#cls, H] or None."""
    path = f"data/pairs/{dim}.jsonl"
    if not os.path.exists(path):
        return None
    items = [json.loads(l) for l in open(path)]
    items = [it for it in items if all(c in it["texts"] for c in classes)]
    if not items:
        return None
    flat, idx = [], {}
    for ii, it in enumerate(items):
        for c in classes:
            idx[(ii, c)] = len(flat)
            flat.append(it["texts"][c])
    H = hidden_batch(base, flat)
    dirs = {}
    for c in classes:
        per = []
        for ii in range(len(items)):
            hc = H[idx[(ii, c)]]
            others = torch.stack([H[idx[(ii, o)]] for o in classes if o != c]).mean(0)
            per.append(hc - others)
        dirs[c] = torch.stack(per).mean(0)
    return dirs


def unpaired_directions(base, dim, classes):
    path = f"data/synth/{dim}.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    means = {}
    for c in classes:
        txts = [r["text"] for r in rows if r["label"] == c]
        if txts:
            means[c] = hidden_batch(base, txts).mean(0)
    dirs = {}
    for c in classes:
        if c not in means:
            continue
        others = [means[o] for o in classes if o in means and o != c]
        ref = torch.stack(others).mean(0) if others else torch.zeros_like(means[c])
        dirs[c] = means[c] - ref
    return dirs


def benign_subspace(base, cfg, r):
    """Top-r principal directions of normal generation activations (for null-space proj)."""
    path = cfg.prefixes_path()
    if not os.path.exists(path):
        return None
    texts = [json.loads(l)["prefix_text"] for l in open(path)]
    X = hidden_batch(base, texts).float()                # [N, H]
    X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)  # right singular vecs = H-space PCs
    return Vh[:r]                                          # [r, H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", action="store_true", help="use contrastive pairs (data/pairs)")
    ap.add_argument("--nullspace", type=int, default=0, help="project orthogonal to top-R benign PCs")
    ap.add_argument("--out-dir", default="models/steering")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    H = base.hidden_size
    caa = torch.zeros(NUM_SLOTS, H)
    info = {}

    Vr = None
    if args.nullspace:
        Vr = benign_subspace(base, cfg, args.nullspace)
        if Vr is not None:
            Vr = Vr.to(base.device)
            print(f"[steering] benign subspace: top-{args.nullspace} PCs for null-space projection")

    for dim, classes in SOFT_DIMS.items():
        dirs = (paired_directions(base, dim, classes) if args.pairs else None)
        src = "pairs"
        if dirs is None:
            dirs = unpaired_directions(base, dim, classes); src = "unpaired"
        if not dirs:
            print(f"[steering] skip {dim} (no data)"); continue
        for c, d in dirs.items():
            d = d.float().to(base.device)
            if Vr is not None:                            # null-space projection (AlphaSteer)
                d = d - Vr.t() @ (Vr @ d)
            d = d / d.norm().clamp_min(1e-6)
            caa[SLOTS[(dim, c)]] = d.cpu()
            info[f"{dim}:{c}"] = SLOTS[(dim, c)]
        print(f"[steering] {dim}: {src} directions for {list(dirs)}"
              f"{' + null-space' if Vr is not None else ''}")

    os.makedirs(args.out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cfg.base_model)
    out = os.path.join(args.out_dir, f"{safe}.pt")
    torch.save({"caa": caa, "classes": info, "base_model": cfg.base_model,
                "hidden_size": H, "paired": args.pairs, "nullspace": args.nullspace}, out)
    print(f"[steering] saved [{NUM_SLOTS},{H}] -> {out}  ({len(info)} directions)")


if __name__ == "__main__":
    main()
