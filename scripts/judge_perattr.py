"""Independent-judge PER-ATTRIBUTE relevance over all seven soft classes (rebuttal:
extends the single-attribute judge column to the full attribute library).

For each of the 7 soft classes and each method (GAP / LM-Steer / prompting), three LLM
judges of distinct families rate 1-5 how strongly the text exhibits the attribute; we
report the mean rating rescaled to [0,1] ((r-1)/4), averaged over judges. Mirrors the
classifier per-attribute table (tab:perattr) so the two sit side by side.

Usage:  python scripts/judge_perattr.py [--limit 100]
"""
import _bootstrap  # noqa: F401
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from gap_control.env import get_client

JUDGES = ["kimi-k2.6", "deepseek-v4-flash", "MiniMax-M2.5"]
# (display, attribute phrase, file tag "<dim>_<value>")
ATTRS = [
    ("positive",  "positive sentiment",  "sentiment_positive"),
    ("negative",  "negative sentiment",  "sentiment_negative"),
    ("formal",    "formal style",        "style_formal"),
    ("informal",  "informal style",      "style_informal"),
    ("joy",       "joyful emotion",      "emotion_joy"),
    ("anger",     "angry emotion",       "emotion_anger"),
    ("sadness",   "sad emotion",         "emotion_sadness"),
]
METHODS = [("GAP", "con_gap_%s"), ("LM-Steer", "con_lm_%s"), ("prompting", "con_pp_%s")]


def load(fname, limit):
    seen, rows = set(), []
    for l in open(f"outputs/{fname}.jsonl"):
        d = json.loads(l)
        if d["prompt_id"] in seen:
            continue
        seen.add(d["prompt_id"]); rows.append((d["prompt"], d["text"]))
        if len(rows) >= limit:
            break
    return rows


def judge_one(model, attr, prompt, text):
    msg = (f"Prompt: {prompt!r}\nText: {text!r}\n\nRate the TEXT for the attribute "
           f'"{attr}" on an integer scale 1-5 (5 = strongly exhibits it). '
           'Reply ONLY as JSON: {"r":x}.')
    extra = ({"thinking": {"type": "disabled"}}
             if model.startswith(("glm", "deepseek")) else {})
    try:
        r = get_client().chat.completions.create(model=model, temperature=0.0, max_tokens=40,
              extra_body=extra, messages=[{"role": "user", "content": msg}])
        m = re.search(r'"r"\s*:\s*([1-5])', r.choices[0].message.content or "")
        if m:
            return (int(m.group(1)) - 1) / 4.0
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    print(f"{'attr':10s} " + " ".join(f"{m:>10s}" for m, _ in METHODS), flush=True)
    print("-" * 46, flush=True)
    col = {m: [] for m, _ in METHODS}
    for disp, phrase, tag in ATTRS:
        vals = {}
        for mname, pat in METHODS:
            rows = load(pat % tag, args.limit)
            per_judge = []
            for j in JUDGES:
                with ThreadPoolExecutor(max_workers=8) as ex:
                    v = [x for x in ex.map(lambda pt: judge_one(j, phrase, pt[0], pt[1]), rows)
                         if x is not None]
                if v:
                    per_judge.append(sum(v) / len(v))
            vals[mname] = sum(per_judge) / len(per_judge) if per_judge else None
            if vals[mname] is not None:
                col[mname].append(vals[mname])
        fmt = lambda x: f"{x:10.2f}" if x is not None else f"{'--':>10s}"
        print(f"{disp:10s} " + " ".join(fmt(vals[m]) for m, _ in METHODS), flush=True)
    print("-" * 46, flush=True)
    print(f"{'MEAN':10s} " + " ".join(f"{sum(col[m])/len(col[m]):10.2f}" for m, _ in METHODS),
          flush=True)


if __name__ == "__main__":
    main()
