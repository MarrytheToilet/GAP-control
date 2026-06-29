"""Generate + filter balanced synthetic labeled data for soft-attribute classifiers.

For each (dim, label): build a balanced seed grid (genres x topics x lengths x registers),
generate with a rotated pool of models, then filter with a distinct model. Writes
data/synth/{dim}.jsonl with rich metadata. Resumable and safe to run in the background.

Usage:
    python scripts/synth_data.py --dims sentiment,emotion,style --per-class 400
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from gap_control.env import load_env, synth_settings
from gap_control.attributes import SOFT_DIMS
from gap_control import synth, seeds


def existing_counts(path):
    counts = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                counts[json.loads(line)["label"]] = counts.get(json.loads(line)["label"], 0) + 1
            except Exception:
                pass
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="sentiment,emotion,style")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--per-cell", type=int, default=8, help="examples requested per seed cell")
    ap.add_argument("--oversample", type=float, default=1.8, help="generate this x per-class to survive filtering")
    ap.add_argument("--keep-threshold", type=float, default=0.6)
    ap.add_argument("--out-dir", default="data/synth")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    load_env()
    s = synth_settings()
    gen_models = s["gen_models"]
    print(f"[synth] generators={gen_models} filter={s['filter_model']} "
          f"per_class={args.per_class} keep>={args.keep_threshold}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for dim in args.dims.split(","):
        path = os.path.join(args.out_dir, f"{dim}.jsonl")
        done = existing_counts(path) if args.resume else {}
        if not args.resume and os.path.exists(path):
            os.remove(path)
        for label in SOFT_DIMS[dim]:
            have = done.get(label, 0)
            need = args.per_class - have
            if need <= 0:
                print(f"[synth] {dim}/{label}: already have {have}, skip", flush=True)
                continue
            target_gen = int(need * args.oversample)
            n_cells = max(1, target_gen // args.per_cell)
            cells = seeds.seed_cells(dim, label, n_cells, seed=args.seed)

            # generate across cells concurrently, rotating generator models
            def gen_one(i_cell):
                i, cell = i_cell
                model = gen_models[i % len(gen_models)]
                try:
                    texts = synth.generate_examples(
                        dim, label, args.per_cell, cell=cell, model=model,
                        temperature=s["temperature"], max_tokens=s["max_tokens"])
                except Exception as e:
                    print(f"   gen err [{dim}/{label}] {model}: {str(e)[:80]}", flush=True)
                    return []
                return [{"dim": dim, "label": label, "text": t, "genre": cell["genre"],
                         "topic": cell["topic"], "gen_model": model} for t in texts]

            cand = []
            with ThreadPoolExecutor(max_workers=s["concurrency"]) as ex:
                for fut in as_completed([ex.submit(gen_one, ic) for ic in enumerate(cells)]):
                    cand.extend(fut.result())
            # dedup exact texts
            seen, uniq = set(), []
            for c in cand:
                if c["text"] not in seen:
                    seen.add(c["text"]); uniq.append(c)

            kept = synth.filter_examples(dim, label, uniq, filter_model=s["filter_model"],
                                         keep_threshold=args.keep_threshold,
                                         concurrency=s["concurrency"])
            kept = kept[:need]
            with open(path, "a") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[synth] {dim}/{label}: generated {len(uniq)} -> kept {len(kept)} "
                  f"(total {have + len(kept)}/{args.per_class})", flush=True)
        print(f"[synth] {dim}: done -> {path}", flush=True)


if __name__ == "__main__":
    main()
