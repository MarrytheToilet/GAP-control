"""Residual-trajectory diagnostic (handbook 5: "证明闭环：未满足时强、满足后弱").

Visualizes the dynamic control loop over generation steps for the GAP-Control method:
per-step residual norm ||b_t||, value gap g_t, predicted value V_hat, observed
satisfaction, and KL to the base distribution. A healthy plot shows the residual strong
while the gap is open and collapsing toward zero once the constraint is satisfied.

Reads outputs/{task}_decode.jsonl (must include trajectories, i.e. decoded with `gap`).

Usage:
    python scripts/analyze_trajectory.py --config configs/keyword_demo.yaml [--index 0]
"""
import _bootstrap  # noqa: F401
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gap_control.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=-1, help="which gap sample (-1 = average)")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    path = f"{cfg.out_dir}/{cfg.task}_decode.jsonl"
    gaps = [json.loads(l) for l in open(path)
            if json.loads(l)["method"] == "gap" and "trajectory" in json.loads(l)]
    if not gaps:
        raise SystemExit("no gap trajectories found; decode with --methods gap first")

    keys = ["bnorm", "gap", "vhat", "obs_sat", "kl"]
    labels = {"bnorm": "||b_t|| residual norm", "gap": "g_t value gap",
              "vhat": "V_hat predicted value", "obs_sat": "observed satisfaction",
              "kl": "KL to base"}

    if args.index >= 0:
        series = {k: gaps[args.index]["trajectory"][k] for k in keys}
        title_n = f"sample {args.index}"
    else:  # average over samples, truncating to the shortest
        T = min(len(g["trajectory"]["bnorm"]) for g in gaps)
        series = {k: [sum(g["trajectory"][k][t] for g in gaps) / len(gaps)
                      for t in range(T)] for k in keys}
        title_n = f"mean of {len(gaps)} samples"

    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 1.7 * len(keys)), sharex=True)
    for ax, k in zip(axes, keys):
        ax.plot(series[k], marker=".", lw=1.4)
        ax.set_ylabel(labels[k], fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("generation step t")
    fig.suptitle(f"GAP-Control dynamic loop — {cfg.task} ({title_n})", fontsize=11)
    fig.tight_layout()
    out = f"{cfg.out_dir}/{cfg.task}_trajectory.png"
    fig.savefig(out, dpi=130)
    print(f"[trajectory] saved -> {out}")

    # also print a compact ASCII summary (works headless)
    print(f"\nstep |  ||b|| |   gap | V_hat | obs_sat |    KL")
    for t in range(min(len(series["bnorm"]), 30)):
        print(f"{t:4d} | {series['bnorm'][t]:6.3f} | {series['gap'][t]:5.3f} | "
              f"{series['vhat'][t]:5.3f} | {series['obs_sat'][t]:7.3f} | {series['kl'][t]:5.3f}")


if __name__ == "__main__":
    main()
