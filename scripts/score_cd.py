"""Score the controlled-decoding (CD) baseline outputs with the same classifier and
metrics as the main table: single-attribute relevance/success (positive) and
compositional joint success (unseen pair pos+formal; triple pos+formal+joy)."""
import _bootstrap  # noqa: F401
import json, os
import numpy as np
from gap_control.config import Config
from gap_control.classifiers import SoftClassifierBank

cfg = Config.load("configs/_gt_g2_rt1p0_rm25_single.yaml")
bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, "cpu")
POS=("sentiment","positive"); FRM=("style","formal"); JOY=("emotion","joy")

def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None

def prob(rows, d, v):
    return bank.prob(d, v, [r["text"] for r in rows]).cpu().numpy()

def joint(rows, attrs):
    ok = np.ones(len(rows), bool)
    for d, v in attrs:
        ok &= (bank.prob(d, v, [r["text"] for r in rows]).cpu().numpy() > 0.5)
    return ok.mean()

print(f"{'metric':<22}{'CD (n)':>16}")
print("-"*40)
r = rows("outputs/cd_single.jsonl")
if r:
    p = prob(r, *POS)
    print(f"{'rel P(positive)':<22}{p.mean():.3f} (n{len(r)})")
    print(f"{'succ P>0.5':<22}{(p>0.5).mean():.3f}")
r = rows("outputs/cd_ho2.jsonl")
if r:
    print(f"{'unseen pair joint':<22}{joint(r,[POS,FRM]):.3f} (n{len(r)})")
r = rows("outputs/cd_tri.jsonl")
if r:
    print(f"{'triple joint':<22}{joint(r,[POS,FRM,JOY]):.3f} (n{len(r)})")
print("\nref (main table): GAP 0.82/0.84/0.54/0.48 | LM-Steer 0.76/0.77/0.44/0.38 | BoN8 0.74/-/0.50/0.43")
