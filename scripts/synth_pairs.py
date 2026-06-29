"""Synthesize CONTRASTIVE PAIRS for clean steering directions (CAA best practice).

Naive CAA from unpaired data conflates the attribute with topic/length/style confounds.
Here each item is the SAME content written in every class of a dim, so the per-class hidden
means differ (mostly) only in the attribute -> mean(h_class - h_others) isolates it.

Writes data/pairs/{dim}.jsonl: {dim, topic, texts: {class: text}}.

Usage:
    python scripts/synth_pairs.py --dims sentiment,emotion,style --n 120
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from gap_control.env import load_env, synth_settings
from gap_control.attributes import SOFT_DIMS
from gap_control.synth import _chat
from gap_control.seeds import GENRES, TOPICS
from gap_control.synth import DESCRIPTIONS


def parse_obj(text, keys):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in keys:
        # tolerate case / spacing in keys
        for ok in obj:
            if ok.strip().lower() == k.lower():
                v = obj[ok]
                if isinstance(v, str) and v.strip():
                    out[k] = v.strip()
    return out if len(out) == len(keys) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="sentiment,emotion,style")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out-dir", default="data/pairs")
    args = ap.parse_args()
    load_env()
    s = synth_settings()
    gen = s["gen_models"]
    os.makedirs(args.out_dir, exist_ok=True)

    for dim in args.dims.split(","):
        classes = SOFT_DIMS[dim]
        descs = {c: DESCRIPTIONS.get((dim, c), c) for c in classes}

        def one(i):
            topic = TOPICS[i % len(TOPICS)]
            genre = GENRES[i % len(GENRES)]
            model = gen[i % len(gen)]
            spec = "; ".join(f'"{c}": a version that is {descs[c]}' for c in classes)
            user = (f"Write {genre} about {topic}, in {len(classes)} versions that keep the "
                    f"SAME content, length, and topic, changing ONLY the {dim}. Each 1-2 "
                    f"sentences. Versions: {spec}. Return ONLY a JSON object with keys "
                    f"{classes} mapping to the version strings.")
            try:
                out = _chat(model, [{"role": "user", "content": user}],
                            temperature=0.9, max_tokens=400)
                obj = parse_obj(out, classes)
                if obj:
                    return {"dim": dim, "topic": topic, "genre": genre, "texts": obj}
            except Exception as e:
                print(f"   err {dim} {topic}: {str(e)[:60]}")
            return None

        items = []
        with ThreadPoolExecutor(max_workers=s["concurrency"]) as ex:
            for r in ex.map(one, range(int(args.n * 1.4))):
                if r:
                    items.append(r)
                if len(items) >= args.n:
                    break
        path = os.path.join(args.out_dir, f"{dim}.jsonl")
        with open(path, "w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[pairs] {dim}: wrote {len(items)} contrastive items -> {path}", flush=True)


if __name__ == "__main__":
    main()
