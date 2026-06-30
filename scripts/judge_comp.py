"""Composition judge: rate EACH target attribute of a generation with independent LLM
judges, and report JOINT success (all target attributes rated >= 3/5). Tests whether a
method's classifier-measured composition advantage survives independent evaluation.

Usage:
  python scripts/judge_comp.py --attrs "positive sentiment,formal style" --limit 100 \
      --judges glm-5.2,kimi-k2.6,deepseek-v4-flash \
      gap=outputs/rb_gap2_ho2.jsonl cd=outputs/cd_ho2.jsonl
"""
import _bootstrap  # noqa: F401
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from gap_control.env import get_client


def judge_all(model, attrs, prompt, text):
    lines = "\n".join(f"{i+1}) {a}" for i, a in enumerate(attrs))
    keys = ",".join(f'"{i+1}":x' for i in range(len(attrs)))
    msg = (f"Prompt: {prompt!r}\nText: {text!r}\n\n"
           f"Rate the TEXT on each attribute below, integer 1-5 (5 = strongly exhibits it):\n"
           f"{lines}\nReply ONLY as JSON: {{{keys}}}.")
    extra = {"thinking": {"type": "disabled"}} if model.startswith("glm") else {}
    try:
        r = get_client().chat.completions.create(
            model=model, temperature=0.0, max_tokens=60, extra_body=extra,
            messages=[{"role": "user", "content": msg}])
        out = r.choices[0].message.content or ""
        vals = {int(k): int(v) for k, v in re.findall(r'"(\d+)"\s*:\s*([1-5])', out)}
        if len(vals) == len(attrs):
            return all(vals[i + 1] >= 3 for i in range(len(attrs)))  # joint: all satisfied
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", required=True, help="comma-separated target attributes")
    ap.add_argument("--judges", default="glm-5.2,kimi-k2.6,deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("pairs", nargs="+")
    args = ap.parse_args()
    attrs = [a.strip() for a in args.attrs.split(",")]
    judges = args.judges.split(",")

    methods = {}
    for pr in args.pairs:
        name, path = pr.split("=", 1)
        seen, rows = set(), []
        for l in open(path):
            d = json.loads(l)
            if d["prompt_id"] in seen:
                continue
            seen.add(d["prompt_id"]); rows.append((d["prompt"], d["text"]))
            if len(rows) >= args.limit:
                break
        methods[name] = rows

    print(f"JOINT success on [{' + '.join(attrs)}] | {args.limit}/method | judges: {judges}\n")
    header = f"{'method':10s}" + "".join(f"{j:>16s}" for j in judges)
    print(header); print("-" * len(header))
    for name, rows in methods.items():
        cells = []
        for j in judges:
            with ThreadPoolExecutor(max_workers=8) as ex:
                v = list(ex.map(lambda pt: judge_all(j, attrs, pt[0], pt[1]), rows))
            v = [x for x in v if x is not None]
            cells.append(f"{(sum(v)/len(v) if v else float('nan')):.2f}(n{len(v)})")
        print(f"{name:10s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
