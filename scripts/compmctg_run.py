"""CompMCTG driver: decode every attribute combination under every protocol split.

Assumes compmctg_prep.py has run and (a) reward classifiers are trained into
models/classifier_<ds>/, (b) the cache + controllers exist (see --stage). Protocol
semantics: under split s, GAP decodes ALL combos with the controller trained on split
s's seen combos; prompting and LM-Steer are split-independent (per-attribute only) and
are decoded once. Outputs land in outputs/cmg_<ds>_<split>_<method>_<combo>.jsonl.

Stages (idempotent; each skips work whose output exists):
    classifiers  train bge reward heads on data/compmctg/<ds>/
    cache        build prefixes + shared-rollout cache (Original config)
    train        train GAP controller per split config + one LM-Steer
    decode       decode all combos x methods x splits on the eval prompts
Usage:
    GAPCTRL_ATTRS=yelp python scripts/compmctg_run.py --dataset Yelp --stage all
"""
import _bootstrap  # noqa: F401
import argparse
import itertools
import json
import os
import subprocess
import sys

import yaml

ASPECTS = {"Yelp": ["sentiment", "pronoun", "tense"],
           "Fyelp": ["sentiment", "gender", "cuisine", "tense"],
           "Amazon": ["sentiment", "topic"]}


def sh(cmd):
    print(f">> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ASPECTS))
    ap.add_argument("--stage", default="all",
                    choices=["all", "classifiers", "cache", "train", "decode"])
    ap.add_argument("--splits", default="orig,ho0,ho1,acd0,acd1")
    ap.add_argument("--samples", type=int, default=1, help="samples per prompt")
    args = ap.parse_args()
    ds = args.dataset
    dsl = ds.lower()
    task = f"{dsl}_multi"
    assert os.environ.get("GAPCTRL_ATTRS") == dsl, f"run with GAPCTRL_ATTRS={dsl}"
    from gap_control.attributes import SOFT_DIMS
    dims = ASPECTS[ds]
    combos = list(itertools.product(*[SOFT_DIMS[d] for d in dims]))
    splits = args.splits.split(",")
    stages = (["classifiers", "cache", "train", "decode"]
              if args.stage == "all" else [args.stage])
    clf_dir = f"models/classifier_{dsl}"
    py = [sys.executable]

    if "classifiers" in stages:
        os.makedirs(clf_dir, exist_ok=True)
        for d in dims:
            if os.path.exists(f"{clf_dir}/{d}.pt"):
                print(f"[skip] {clf_dir}/{d}.pt"); continue
            sh(py + ["scripts/train_classifier.py", "--dim", d,
                     "--data-dir", f"data/compmctg/{dsl}", "--out-dir", clf_dir])

    # all split configs share task/prefixes/cache; make sure classifier_dir points at ours
    for name in [task] + [f"{task}_{s}" for s in splits if s != "orig"]:
        p = f"configs/{name}.yaml"
        c = yaml.safe_load(open(p))
        if c.get("classifier_dir") != clf_dir:
            c["classifier_dir"] = clf_dir
            yaml.safe_dump(c, open(p, "w"), sort_keys=False)

    if "cache" in stages:
        if not os.path.exists(f"data/teacher/{task}/topk_advantage.pt"):
            sh(py + ["scripts/build_prefixes.py", "--config", f"configs/{task}.yaml"])
            sh(py + ["scripts/estimate_teacher_multi.py", "--config", f"configs/{task}.yaml"])
        else:
            print(f"[skip] cache data/teacher/{task}")

    if "train" in stages:
        for s in splits:
            cfgp = f"configs/{task}.yaml" if s == "orig" else f"configs/{task}_{s}.yaml"
            out = f"models/controller/{task}/{s}.pt"
            if os.path.exists(out):
                print(f"[skip] {out}"); continue
            sh(py + ["scripts/train_compositional.py", "--config", cfgp, "--out", out])
        # one LM-Steer (split-independent)
        lmcfg = f"configs/{task}_lmsteer.yaml"
        c = yaml.safe_load(open(f"configs/{task}.yaml"))
        c.update(controller_type="lmsteer", interaction=False, projection="l2",
                 rho_min=4.0, rho_max=4.0, decode_strength=1.0)
        c.pop("gate_gamma", None)
        yaml.safe_dump(c, open(lmcfg, "w"), sort_keys=False)
        out = f"models/controller/{task}/lmsteer.pt"
        if not os.path.exists(out):
            sh(py + ["scripts/train_compositional.py", "--config", lmcfg, "--out", out])

    if "decode" in stages:
        prompts = f"data/prompts/{task}_eval.jsonl"
        for combo in combos:
            tag = "".join(v[:3] for v in combo)
            control = [{"dim": d, "value": v, "alpha": 1.0} for d, v in zip(dims, combo)]
            ccfg = yaml.safe_load(open(f"configs/{task}.yaml"))
            ccfg.update(control=control, samples_per_prompt=args.samples)
            cpath = f"configs/_cmg_{task}_{tag}.yaml"
            yaml.safe_dump(ccfg, open(cpath, "w"), sort_keys=False)
            # LM-Steer decodes with ITS OWN projection recipe (fixed rho, l2), not the gate
            lcfg = yaml.safe_load(open(f"configs/{task}_lmsteer.yaml"))
            lcfg.update(control=control, samples_per_prompt=args.samples)
            lpath = f"configs/_cmg_{task}_lm_{tag}.yaml"
            yaml.safe_dump(lcfg, open(lpath, "w"), sort_keys=False)
            # split-independent baselines, decoded once per combo
            out = f"outputs/cmg_{dsl}_shared_pp_{tag}.jsonl"
            if not os.path.exists(out):
                sh(py + ["scripts/decode_gap_control.py", "--config", cpath,
                         "--ckpt", f"models/controller/{task}/orig.pt",
                         "--methods", "prompting", "--prompts", prompts, "--out", out])
            out = f"outputs/cmg_{dsl}_shared_lm_{tag}.jsonl"
            if not os.path.exists(out):
                sh(py + ["scripts/decode_gap_control.py", "--config", lpath,
                         "--ckpt", f"models/controller/{task}/lmsteer.pt",
                         "--methods", "gap", "--prompts", prompts, "--out", out])
            # GAP per split
            for s in splits:
                out = f"outputs/cmg_{dsl}_{s}_gap_{tag}.jsonl"
                if os.path.exists(out):
                    continue
                sh(py + ["scripts/decode_gap_control.py", "--config", cpath,
                         "--ckpt", f"models/controller/{task}/{s}.pt",
                         "--methods", "gap", "--prompts", prompts, "--out", out])
    print("[driver] done")


if __name__ == "__main__":
    main()
