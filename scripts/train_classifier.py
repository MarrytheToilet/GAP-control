"""Fine-tune a soft-attribute classifier on synthetic data (bge-base-en + linear head).

Reads data/synth/{dim}.jsonl, trains, reports held-out accuracy + ECE, saves to
models/classifier/{dim}.pt for the SoftClassifierBank.

Usage:
    python scripts/train_classifier.py --dim sentiment [--backbone bge-base-en-v1.5]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import torch
import torch.nn.functional as F

from gap_control.env import load_env
from gap_control.attributes import SOFT_DIMS
from gap_control.classifiers import SoftClassifier


def expected_calibration_error(probs, labels, n_bins=10):
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    ece, edges = 0.0, torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return ece.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", required=True)
    ap.add_argument("--backbone", default="bge-base-en-v1.5")
    ap.add_argument("--data-dir", default="data/synth")
    ap.add_argument("--out-dir", default="models/classifier")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    load_env()
    torch.manual_seed(args.seed)

    classes = SOFT_DIMS[args.dim]
    cls2id = {c: i for i, c in enumerate(classes)}
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, f"{args.dim}.jsonl"))]
    texts = [r["text"] for r in rows]
    labels = torch.tensor([cls2id[r["label"]] for r in rows])
    print(f"[clf] dim={args.dim} classes={classes} n={len(texts)}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(texts), generator=g)
    n_val = max(1, int(len(texts) * args.val_frac))
    val_i, tr_i = perm[:n_val].tolist(), perm[n_val:].tolist()

    model = SoftClassifier(args.backbone, classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def encode(batch_texts):
        enc = model.tokenizer(batch_texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=128).to(args.device)
        return enc.input_ids, enc.attention_mask

    for epoch in range(args.epochs):
        model.train()
        order = [tr_i[k] for k in torch.randperm(len(tr_i)).tolist()]
        tot = 0.0
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            ii, am = encode([texts[k] for k in idx])
            y = labels[idx].to(args.device)
            logits = model(ii, am)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        # eval
        probs = model.predict_proba([texts[k] for k in val_i], args.device)
        yval = labels[val_i]
        acc = (probs.argmax(-1) == yval).float().mean().item()
        ece = expected_calibration_error(probs, yval)
        print(f"epoch {epoch} | train_loss {tot/len(tr_i):.4f} | val_acc {acc:.3f} | ECE {ece:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.dim}.pt")
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "backbone": args.backbone, "val_acc": acc, "ece": ece}, out)
    print(f"[clf] saved -> {out}  (val_acc {acc:.3f}, ECE {ece:.3f})")


if __name__ == "__main__":
    main()
