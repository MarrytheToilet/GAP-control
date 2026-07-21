"""Score CompMCTG decodes into the protocol table.

For each protocol split: average attribute accuracy (every aspect correct per their
metric = mean over aspects of P(label correct)) and joint success over (a) the split's
UNSEEN combos — the compositional-generalization metric — and (b) all combos. GAP uses
the split's own controller outputs; prompting/LM-Steer are split-independent.

--evaluator bge       our reward classifiers (SANITY ONLY — same family as reward)
--evaluator official  the benchmark's RoBERTa checkpoints (paper numbers)

Usage:
    GAPCTRL_ATTRS=yelp python scripts/compmctg_score.py --dataset Yelp [--evaluator bge]
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import itertools
import json
import os

import numpy as np

ASPECTS = {"Yelp": ["sentiment", "pronoun", "tense"],
           "Fyelp": ["sentiment", "gender", "cuisine", "tense"],
           "Amazon": ["sentiment", "topic"]}


def load_rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


class BgeEval:
    def __init__(self, dsl, device):
        from gap_control.classifiers import SoftClassifierBank
        from gap_control.config import Config
        cfg = Config.load(f"configs/{dsl}_multi.yaml")
        self.bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, device)

    def correct(self, dim, value, texts):
        return self.bank.prob(dim, value, texts).cpu().numpy() > 0.5


class OfficialEval:
    """Benchmark RoBERTa classifiers (third_party/CG4MCTG/evaluation/classifiers)."""

    def __init__(self, ds, device):
        import sys
        import torch
        sys.path.insert(0, "third_party/CG4MCTG/evaluation/scripts")
        self.torch = torch
        self.device = device
        self._root = f"third_party/CG4MCTG/evaluation/classifiers_hf/classifiers/{ds}"
        self.models, self.toks = {}, {}
        mp = json.load(open("third_party/CG4MCTG/data/map_dict.json"))
        self.label_maps = {dim: {k.lower(): v for k, v in mp[dim].items()
                                 if isinstance(v, int) and k != "dim"}
                           for dim in ASPECTS[ds]}
        self.tar_dims = {dim: mp[dim]["dim"] for dim in ASPECTS[ds]}

    def _get(self, dim):
        if dim not in self.models:
            from model import RobertaForPreTraining
            from transformers import RobertaConfig, RobertaTokenizer
            d = os.path.join(self._root, dim)
            cfg = RobertaConfig.from_pretrained(d)
            cfg.tar_dim = self.tar_dims[dim]  # their custom head width (map_dict "dim")
            self.models[dim] = (RobertaForPreTraining.from_pretrained(d, config=cfg)
                                .to(self.device).eval())
            self.toks[dim] = RobertaTokenizer.from_pretrained(d)
        return self.models[dim], self.toks[dim]

    def correct(self, dim, value, texts):
        m, tok = self._get(dim)
        tgt = self.label_maps[dim][value.lower()]
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(self.device)
                # bypass their label-requiring forward: backbone + classification head.
                # their 3.4-era RobertaModel passes the mask straight to the modern
                # encoder, which wants the extended float mask -> pre-extend it here.
                am = enc.attention_mask[:, None, None, :].float()
                ext = (1.0 - am) * self.torch.finfo(self.torch.float32).min
                seq_out, pooled = m.roberta(input_ids=enc.input_ids,
                                            attention_mask=ext)
                _, logits = m.cls(seq_out, pooled)  # seq_relationship_score = classifier
                outs.append(logits.argmax(-1).cpu().numpy() == tgt)
        return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ASPECTS))
    ap.add_argument("--evaluator", default="bge", choices=["bge", "official"])
    ap.add_argument("--splits", default="orig,ho0,ho1,acd0,acd1")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ds, dsl = args.dataset, args.dataset.lower()
    from gap_control.attributes import SOFT_DIMS
    dims = ASPECTS[ds]
    combos = list(itertools.product(*[SOFT_DIMS[d] for d in dims]))
    ev = (BgeEval(dsl, args.device) if args.evaluator == "bge"
          else OfficialEval(ds, args.device))

    # unseen combos per split (from configs)
    import yaml
    unseen = {}
    for s in args.splits.split(","):
        if s == "orig":
            unseen[s] = []
            continue
        c = yaml.safe_load(open(f"configs/{dsl}_multi_{s}.yaml"))
        unseen[s] = [tuple(x.split(":")[1] for x in combo) for combo in c["holdout_combos"]]

    def combo_scores(method_file_fn):
        per_combo = {}
        for combo in combos:
            tag = "".join(v[:3] for v in combo)
            rows = load_rows(method_file_fn(tag))
            if not rows:
                continue
            texts = [r["text"] for r in rows]
            accs, joint = [], np.ones(len(texts), bool)
            for d, v in zip(dims, combo):
                c = ev.correct(d, v, texts)
                accs.append(c.mean())
                joint &= c
            per_combo[combo] = (float(np.mean(accs)), float(joint.mean()))
        return per_combo

    methods = {"prompting": lambda t: f"outputs/cmg_{dsl}_shared_pp_{t}.jsonl",
               "LM-Steer": lambda t: f"outputs/cmg_{dsl}_shared_lm_{t}.jsonl"}
    shared = {m: combo_scores(fn) for m, fn in methods.items()}

    print(f"=== {ds} ({args.evaluator} evaluator) ===")
    print(f"{'split':6s} {'method':10s} {'acc-all':>8} {'joint-all':>10} "
          f"{'acc-unseen':>11} {'joint-unseen':>13}")
    results = {}
    for s in args.splits.split(","):
        gap = combo_scores(lambda t, s=s: f"outputs/cmg_{dsl}_{s}_gap_{t}.jsonl")
        for name, pc in [("GAP", gap)] + list(shared.items()):
            if not pc:
                continue
            alls = list(pc.values())
            uns = [pc[c] for c in unseen[s] if c in pc]
            row = dict(acc_all=np.mean([a for a, _ in alls]),
                       joint_all=np.mean([j for _, j in alls]),
                       acc_uns=np.mean([a for a, _ in uns]) if uns else float("nan"),
                       joint_uns=np.mean([j for _, j in uns]) if uns else float("nan"))
            results[(s, name)] = row
            print(f"{s:6s} {name:10s} {row['acc_all']:8.3f} {row['joint_all']:10.3f} "
                  f"{row['acc_uns']:11.3f} {row['joint_uns']:13.3f}")
    out = f"paper/figures/compmctg_{dsl}_{args.evaluator}.json"
    json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
