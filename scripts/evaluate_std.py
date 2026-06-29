"""Standardized CTG evaluation (recent practice: Air-Decoding + LLM-as-judge).

Metrics per method:
  * relevance (classifier)  -- mean P(target attribute) and success@0.5  [the control metric]
  * PPL  -- conditional perplexity of the GENERATED text only (prompt as context, not counted),
            under the model's own base (--ppl-model to override)
  * Dist-1/2/3  -- diversity
  * LLM-judge (--judge)  -- relevance / fluency / topicality in [0,1] (distinct judge model)

Reads one or more decode JSONLs (each line has method/prompt/text). Groups by method.

Usage:
    python scripts/evaluate_std.py --config configs/smol_sent.yaml \
        --files outputs/so_gap_4.jsonl outputs/so_ref.jsonl --judge
"""
import _bootstrap  # noqa: F401
import argparse
import json
from collections import defaultdict

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.classifiers import SoftClassifierBank
from gap_control import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--dim", default="sentiment")
    ap.add_argument("--value", default="positive")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    ap.add_argument("--ppl-model", default=None, help="override the LM used for PPL")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    by_method = defaultdict(list)
    for path in args.files:
        for line in open(path):
            r = json.loads(line)
            by_method[r["method"]].append(r)

    bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, cfg.device)
    base = BaseLM(args.ppl_model or cfg.base_model, cfg.device, cfg.dtype)

    hdr = (f"{'method':<11}{'relev':>7}{'succ':>7}{'PPL':>8}"
           f"{'D-1':>6}{'D-2':>6}{'D-3':>6}{'KL':>7}{'n':>5}")
    if args.judge:
        hdr += f"{'J-rel':>7}{'J-flu':>7}{'J-top':>7}"
    print("\n" + "=" * len(hdr))
    print(f"Standard CTG eval — dim={args.dim} target={args.value} (PPL on generated text only)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))

    order = {"base": 0, "prompting": 1, "static-CAA": 2, "LM-Steer": 3}
    rows = []
    for method, items in by_method.items():
        prompts = [it["prompt"] for it in items]
        texts = [it["text"] for it in items]
        rel = bank.prob(args.dim, args.value, texts)
        row = {"method": method, "relev": rel.mean().item(),
               "succ": (rel > 0.5).float().mean().item(),
               "ppl": metrics.conditional_perplexity(base, prompts, texts),
               "d1": metrics.distinct_n(texts, 1), "d2": metrics.distinct_n(texts, 2),
               "d3": metrics.distinct_n(texts, 3),
               "kl": sum(it.get("mean_kl", 0) for it in items) / len(items),
               "n": len(items)}
        if args.judge:
            from gap_control import judge as J
            jd = J.judge(f"{args.value} in {args.dim}", prompts, texts)
            row.update(jrel=jd["relevance"], jflu=jd["fluency"], jtop=jd["topicality"])
        rows.append(row)

    rows.sort(key=lambda x: order.get(x["method"], 8))
    for x in rows:
        line = (f"{x['method']:<11}{x['relev']:>7.3f}{x['succ']:>7.2f}{x['ppl']:>8.2f}"
                f"{x['d1']:>6.3f}{x['d2']:>6.3f}{x['d3']:>6.3f}{x['kl']:>7.3f}{x['n']:>5}")
        if args.judge:
            line += f"{x['jrel']:>7.2f}{x['jflu']:>7.2f}{x['jtop']:>7.2f}"
        print(line)
    print("=" * len(hdr))


if __name__ == "__main__":
    main()
