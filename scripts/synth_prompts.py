"""Synthesize diverse prompt stems for prefix construction.

The default 16-20 generic stems give narrow, often negatively-biased contexts (base
P(positive)~0.05). This generates many varied, neutral, continuable openings across
domains via the API, so prefixes cover a broader, less-biased state distribution
(improves teacher coverage and the states seen at inference).

Writes data/prompts/{task}.jsonl (read by build_prefixes.py).

Usage:
    python scripts/synth_prompts.py --task compositional_full --n 160
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from gap_control.env import load_env, synth_settings
from gap_control.synth import _chat, _extract_list

DOMAINS = [
    "food and dining", "travel and places", "technology and gadgets", "movies and shows",
    "books", "work and career", "relationships", "health and fitness", "sports",
    "weather", "shopping", "education", "music", "cars", "home and living", "finance",
    "art", "nature", "city life", "everyday errands",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--per-domain", type=int, default=8)
    args = ap.parse_args()
    load_env()
    s = synth_settings()
    gen = s["gen_models"]

    def one(i_dom):
        i, dom = i_dom
        model = gen[i % len(gen)]
        user = (f"Write {args.per_domain} short, NEUTRAL sentence openings (2-6 words each) "
                f"about {dom}, each of which could be continued into either a positive or a "
                f"negative text (do not lean either way). Examples of the style: 'The new "
                f"cafe downtown', 'After the software update', 'The trail by the river'. "
                f"Return ONLY a JSON array of strings.")
        try:
            out = _chat(model, [{"role": "user", "content": user}],
                        temperature=1.0, max_tokens=300)
            return _extract_list(out)
        except Exception as e:
            print(f"  err {dom}: {str(e)[:60]}")
            return []

    prompts = []
    with ThreadPoolExecutor(max_workers=s["concurrency"]) as ex:
        for lst in ex.map(one, enumerate(DOMAINS)):
            prompts.extend(lst)
    # dedup + trim, cap to n
    seen, uniq = set(), []
    for p in prompts:
        p = re.sub(r'^["\'\-\d.\)\s]+', '', p).strip(' "\'')
        if 2 <= len(p.split()) <= 8 and p.lower() not in seen:
            seen.add(p.lower()); uniq.append(p)
    uniq = uniq[:args.n]

    path = f"data/prompts/{args.task}.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i, p in enumerate(uniq):
            f.write(json.dumps({"prompt_id": i, "text": p}, ensure_ascii=False) + "\n")
    print(f"[synth_prompts] wrote {len(uniq)} diverse prompts -> {path}")


if __name__ == "__main__":
    main()
