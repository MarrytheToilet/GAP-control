"""Prepare a CompMCTG dataset (Zhong et al., ACL 2024) for the GAP pipeline.

From third_party/CG4MCTG/data/<Dataset>/{cls,gen,unseen}.jsonl produces:
  1. data/compmctg/<ds>/{dim}.jsonl        classifier training data in our {dim,label,text}
     format (lowercased labels), for scripts/train_classifier.py --data-dir.
  2. data/prompts/<task>.jsonl (+ _eval)   disjoint prompt stems from the dataset's own
     domain (first words of gen.jsonl reviews) for cache prefixes / eval prompts.
  3. configs/<task>.yaml                   Original-protocol config (no held-out combos),
     configs/<task>_ho<i>.yaml / _acd<i>.yaml   one config per Hold-Out / ACD split, with
     holdout_combos filled from unseen.jsonl.

All GAP runs on these configs must set GAPCTRL_ATTRS=<yelp|fyelp|amazon>.

Usage:
    python scripts/compmctg_prep.py --dataset Yelp [--splits 2] [--n-train-prompts 120]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import random
import re

import yaml

ROOT = "third_party/CG4MCTG/data"
ASPECTS = {
    "Yelp": ["sentiment", "pronoun", "tense"],
    "Fyelp": ["sentiment", "gender", "cuisine", "tense"],
    "Amazon": ["sentiment", "topic"],
}
TEXT_KEY = "review"


def norm(v: str) -> str:
    return str(v).strip().lower()


def stems(rows, n_words=(6, 9), rng=None):
    """Clean, deduped opening stems from domain reviews."""
    seen, out = set(), []
    for r in rows:
        t = re.sub(r"\s+", " ", r[TEXT_KEY]).strip()
        words = t.split(" ")
        if len(words) < n_words[1] + 4 or not t[0].isalpha():
            continue
        k = rng.randint(*n_words)
        stem = " ".join(words[:k])
        key = stem.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(stem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ASPECTS))
    ap.add_argument("--splits", type=int, default=2, help="Hold-Out/ACD splits to emit")
    ap.add_argument("--n-train-prompts", type=int, default=120)
    ap.add_argument("--n-eval-prompts", type=int, default=60)
    ap.add_argument("--template", default="configs/flc_multi.yaml")
    args = ap.parse_args()
    ds = args.dataset
    task = f"{ds.lower()}_multi"
    dims = ASPECTS[ds]
    rng = random.Random(0)

    # ---- 1. classifier data ----
    cls_rows = [json.loads(l) for l in open(f"{ROOT}/{ds}/cls.jsonl")]
    os.makedirs(f"data/compmctg/{ds.lower()}", exist_ok=True)
    for dim in dims:
        with open(f"data/compmctg/{ds.lower()}/{dim}.jsonl", "w") as f:
            for r in cls_rows:
                f.write(json.dumps({"dim": dim, "label": norm(r[dim]),
                                    "text": r[TEXT_KEY]}, ensure_ascii=False) + "\n")
    print(f"[prep] classifier data: {len(cls_rows)} rows x {len(dims)} dims "
          f"-> data/compmctg/{ds.lower()}/")

    # ---- 2. prompt stems (train prefixes / eval, disjoint) ----
    gen_rows = [json.loads(l) for l in open(f"{ROOT}/{ds}/gen.jsonl")]
    rng.shuffle(gen_rows)
    all_stems = stems(gen_rows, rng=rng)
    need = args.n_train_prompts + args.n_eval_prompts
    assert len(all_stems) >= need, f"only {len(all_stems)} stems"
    with open(f"data/prompts/{task}.jsonl", "w") as f:
        for i, s in enumerate(all_stems[:args.n_train_prompts]):
            f.write(json.dumps({"prompt_id": i, "text": s}, ensure_ascii=False) + "\n")
    with open(f"data/prompts/{task}_eval.jsonl", "w") as f:
        for i, s in enumerate(all_stems[args.n_train_prompts:need]):
            f.write(json.dumps({"prompt_id": i, "text": s}, ensure_ascii=False) + "\n")
    print(f"[prep] prompts: {args.n_train_prompts} train + {args.n_eval_prompts} eval "
          f"-> data/prompts/{task}*.jsonl")

    # ---- 3. configs from protocol splits ----
    tpl = yaml.safe_load(open(args.template))
    from gap_control.attributes import _ATTR_PRESETS
    soft = _ATTR_PRESETS[ds.lower()]["soft"]
    atomics = [{"dim": d, "value": v} for d, vs in soft.items() for v in vs]
    base = dict(tpl)
    base.update(
        task=task, atomics=atomics, max_attrs=len(dims),
        num_prompts=args.n_train_prompts, prefixes_per_prompt=2,
        holdout_combos=[], use_steering=False,
        control=[{"dim": dims[0], "value": norm(soft[dims[0]][0]), "alpha": 1.0}],
    )

    def dump(cfg, name):
        with open(f"configs/{name}.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[prep] wrote configs/{name}.yaml "
              f"(holdout={len(cfg['holdout_combos'])} combos)")

    dump(base, task)  # Original protocol

    unseen = [json.loads(l) for l in open(f"{ROOT}/{ds}/unseen.jsonl")]
    modes = {}
    for r in unseen:
        modes.setdefault(r["mode"], []).append(r)
    print(f"[prep] protocol modes: { {m: len(v) for m, v in modes.items()} }")
    for mode, tag in [("Hold-Out", "ho"), ("ACD", "acd")]:
        for i, split in enumerate(modes.get(mode, [])[:args.splits]):
            combos = [[f"{d}:{norm(c[d])}" for d in dims] for c in split["unseen_combs"]]
            cfg = dict(base)
            cfg["holdout_combos"] = combos
            dump(cfg, f"{task}_{tag}{i}")

    print("[prep] done. Run everything with GAPCTRL_ATTRS=" + ds.lower())


if __name__ == "__main__":
    main()
