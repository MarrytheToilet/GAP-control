"""Independent multi-judge evaluation: re-score generations with several LLM judges
from model families distinct from the BGE reward/eval classifier, to test whether the
method ranking is an artifact of the classifier family.

Usage:
  python scripts/judge_eval.py --attribute "positive sentiment" --limit 120 \
      --judges glm-5.2,kimi-k2.6,deepseek-v4-flash \
      gap=outputs/con_gap_sentiment_positive.jsonl \
      lmsteer=outputs/con_lm_sentiment_positive.jsonl \
      prompting=outputs/con_pp_sentiment_positive.jsonl
"""
import _bootstrap  # noqa: F401
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from gap_control.env import get_client

RUBRIC = ('Rate the TEXT for the attribute "{attr}" on an integer scale 1-5 '
          '(5 = clearly {attr}). Reply ONLY as JSON: {{"relevance":x}}.')


def score_one(model, attr, prompt, text):
    extra = {"thinking": {"type": "disabled"}} if model.startswith("glm") else {}
    try:
        r = get_client().chat.completions.create(
            model=model, temperature=0.0, max_tokens=40, extra_body=extra,
            messages=[{"role": "user",
                       "content": f"Prompt: {prompt!r}\nText: {text!r}\n\n" + RUBRIC.format(attr=attr)}])
        m = re.search(r'"relevance"\s*:\s*([1-5])', r.choices[0].message.content or "")
        if m:
            return (int(m.group(1)) - 1) / 4.0
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--judges", default="glm-5.2,kimi-k2.6,deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=120, help="generations per method (1 per prompt)")
    ap.add_argument("pairs", nargs="+", help="name=path.jsonl")
    args = ap.parse_args()
    judges = args.judges.split(",")

    methods = {}
    for pr in args.pairs:
        name, path = pr.split("=", 1)
        seen, rows = set(), []
        for l in open(path):
            d = json.loads(l)
            if d["prompt_id"] in seen:    # 1 sample per prompt for a clean per-prompt set
                continue
            seen.add(d["prompt_id"])
            rows.append((d["prompt"], d["text"]))
            if len(rows) >= args.limit:
                break
        methods[name] = rows

    print(f"attribute: {args.attribute} | {args.limit}/method | judges: {judges}\n")
    header = f"{'method':12s}" + "".join(f"{j:>20s}" for j in judges)
    print(header); print("-" * len(header))
    table = {}
    for name, rows in methods.items():
        cells = []
        for j in judges:
            with ThreadPoolExecutor(max_workers=8) as ex:
                sc = list(ex.map(lambda pt: score_one(j, args.attribute, pt[0], pt[1]), rows))
            sc = [s for s in sc if s is not None]
            mean = sum(sc) / len(sc) if sc else float("nan")
            succ = sum(s > 0.5 for s in sc) / len(sc) if sc else float("nan")
            cells.append(f"{mean:.2f}/{succ:.2f}(n{len(sc)})")
            table[(name, j)] = (mean, succ)
        print(f"{name:12s}" + "".join(f"{c:>20s}" for c in cells))
    print("\n(cell = mean-relevance / success-rate(>0.5))")


if __name__ == "__main__":
    main()
