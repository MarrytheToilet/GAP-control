"""Score the completed rebuttal outputs into paper-ready numbers (mean+/-std over seeds).
Classifier forced to CPU so it does not contend with the figdata GPU job.
Prints: main-table SmolLM2 block, hard-control, signed-control, matched-perturbation.
"""
import _bootstrap  # noqa: F401
import json, glob, os
from collections import defaultdict
import numpy as np
from gap_control.config import Config
from gap_control.classifiers import SoftClassifierBank
from gap_control import metrics

cfg = Config.load("configs/smol_multi.yaml")
bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, "cpu")
POS=("sentiment","positive"); INF=("style","informal"); FRM=("style","formal"); JOY=("emotion","joy")

def texts(path, method=None):
    if not os.path.exists(path): return None
    rows=[json.loads(l) for l in open(path)]
    if method: rows=[r for r in rows if r["method"]==method]
    return rows or None

def rel(rows, dim, val):
    return bank.prob(dim,val,[r["text"] for r in rows]).cpu().numpy()

def joint(rows, attrs):
    t=[r["text"] for r in rows]; ok=np.ones(len(t),bool)
    for d,v in attrs: ok&=(bank.prob(d,v,t).cpu().numpy()>0.5)
    return ok.mean()

def seeds_stat(fn, seeds=(1,2,3)):
    """fn(seed)->float or None; return mean,std over available seeds."""
    vals=[fn(s) for s in seeds]; vals=[v for v in vals if v is not None]
    if not vals: return None
    return float(np.mean(vals)), float(np.std(vals))

def fmt(ms):
    return "$-$" if ms is None else f"{ms[0]:.2f}\\,\\tiny$\\pm${ms[1]:.2f}"

# ---- file resolver per method/condition ----
def f(method, cond, seed):
    # method in {base,prompting,preadd,gap,lm,caa}; cond in {single,id2,ho2,tri}
    if method=="base": return f"outputs/rev_base_s{seed}.jsonl", "base"
    if method in ("prompting","preadd"): return f"outputs/rev_pp_{cond}_s{seed}.jsonl", method
    tag={"gap":"gap","lm":"lm","caa":"caa"}[method]
    return f"outputs/rev_{tag}_{cond}_s{seed}.jsonl", "gap"

def metric(method, cond, kind):
    def one(seed):
        path,m=f(method,cond,seed); rows=texts(path,m)
        if not rows: return None
        if kind=="rel":   return float(rel(rows,*POS).mean())
        if kind=="succ":  return float((rel(rows,*POS)>0.5).mean())
        if kind=="seen":  return joint(rows,[POS,INF])
        if kind=="unseen":return joint(rows,[POS,FRM])
        if kind=="tri":   return joint(rows,[POS,FRM,JOY])
        if kind=="d2":    return metrics.distinct_n([r["text"] for r in rows],2)
        if kind=="ms":    return float(np.mean([r.get("ms_per_token",0) for r in rows]))
    return seeds_stat(one)

print("\n================ MAIN TABLE — SmolLM2 block (mean+/-std, 3 seeds) ================")
print(f"{'method':<11}{'Rel':>14}{'Succ':>14}{'Seen':>14}{'Unseen':>14}{'Triple':>14}{'D2':>13}{'ms/tok':>15}")
for meth,label in [("base","base"),("prompting","prompting"),("preadd","PREADD"),
                   ("caa","static-CAA"),("lm","LM-Steer"),("gap","GAP")]:
    rel_=metric(meth,"single","rel"); suc=metric(meth,"single","succ")
    seen=metric(meth,"id2","seen"); uns=metric(meth,"ho2","unseen"); tri=metric(meth,"tri","tri")
    # base joint: score the uncontrolled single text for the combos
    if meth=="base":
        seen=seeds_stat(lambda s: (lambda r: joint(r,[POS,INF]) if r else None)(texts(f"outputs/rev_base_s{s}.jsonl","base")))
        uns =seeds_stat(lambda s: (lambda r: joint(r,[POS,FRM]) if r else None)(texts(f"outputs/rev_base_s{s}.jsonl","base")))
        tri =seeds_stat(lambda s: (lambda r: joint(r,[POS,FRM,JOY]) if r else None)(texts(f"outputs/rev_base_s{s}.jsonl","base")))
    d2=metric(meth,"single","d2"); ms=metric(meth,"single","ms")
    def c(x): return "  --  " if x is None else f"{x[0]:.2f}±{x[1]:.2f}"
    print(f"{label:<11}{c(rel_):>14}{c(suc):>14}{c(seen):>14}{c(uns):>14}{c(tri):>14}{c(d2):>13}{c(ms):>15}")

print("\n================ HARD CONTROL (success rate, simple verifiers) ================")
def verify(text, kind):
    t=text.lower()
    if kind=="keyword": return float("ocean" in t)
    if kind=="length":  return float(len(text.split())<=20)          # 'short'
    if kind=="structure": return float("?" in text)                   # interrogative
for ha in ["keyword","length","structure"]:
    for meth in ["base","prompting","gap"]:
        def one(s):
            r=texts(f"outputs/rev_hard_{ha}_s{s}.jsonl",meth)
            return None if not r else float(np.mean([verify(x["text"],ha) for x in r]))
        st=seeds_stat(one)
        print(f"  {ha:<10} {meth:<10} {'--' if st is None else f'{st[0]:.2f}±{st[1]:.2f}'}")

print("\n================ SIGNED CONTROL (P positive vs alpha) ================")
for a,tag in [(-1.0,"m1p0"),(-0.5,"m0p5"),(0.0,"0p0"),(0.5,"0p5"),(1.0,"1p0")]:
    st=seeds_stat(lambda s:(lambda r: float(rel(r,*POS).mean()) if r else None)(texts(f"outputs/rev_sgn{tag}_s{s}.jsonl","gap")))
    print(f"  alpha={a:+.1f}  {'--' if st is None else f'{st[0]:.2f}±{st[1]:.2f}'}")

print("\n================ MATCHED PERTURBATION (gap strength sweep) ================")
for s_,tag in [(1.0,"1p0"),(1.5,"1p5"),(2.0,"2p0"),(3.0,"3p0")]:
    rl=seeds_stat(lambda s:(lambda r: float(rel(r,*POS).mean()) if r else None)(texts(f"outputs/rev_str{tag}_s{s}.jsonl","gap")))
    kl=seeds_stat(lambda s:(lambda r: float(np.mean([x.get('mean_kl',0) for x in r])) if r else None)(texts(f"outputs/rev_str{tag}_s{s}.jsonl","gap")))
    print(f"  strength={s_}  rel={'--' if rl is None else f'{rl[0]:.2f}'}  KL={'--' if kl is None else f'{kl[0]:.2f}'}")
print("\n[done]")
