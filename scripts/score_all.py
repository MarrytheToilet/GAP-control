"""Score the full completed suite (SmolLM2 / Llama / Falcon main + Llama deep-dive)."""
import _bootstrap  # noqa
import json, glob, os, numpy as np
from gap_control.config import Config
from gap_control.classifiers import SoftClassifierBank
from gap_control import metrics
cfg = Config.load("configs/smol_multi.yaml")
dev = "cuda" if os.environ.get("USE_CUDA") else "cpu"
bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, dev)
POS=("sentiment","positive");NEG=("sentiment","negative");INF=("style","informal")
FRM=("style","formal");JOY=("emotion","joy");ANG=("emotion","anger");SAD=("emotion","sadness")

def rows(pat, method=None):
    fs=sorted(glob.glob(pat)); R=[]
    for f in fs: R+=[json.loads(l) for l in open(f)]
    if method: R=[r for r in R if r.get("method")==method]
    return R or None
def rel(R,d,v): return bank.prob(d,v,[r["text"] for r in R]).cpu().numpy()
def joint(R,attrs):
    ok=np.ones(len(R),bool)
    for d,v in attrs: ok&=(rel(R,d,v)>0.5)
    return round(float(ok.mean()),2)
def m1(R,d,v): return round(float(rel(R,*( (d,v) )).mean()),2)
def succ(R,d,v): return round(float((rel(R,d,v)>0.5).mean()),2)
def d2(R): return round(metrics.distinct_n([r["text"] for r in R],2),2)
def ms(R): return round(float(np.mean([r.get("ms_per_token",0) for r in R])),1)

# ---------- MAIN TABLE: 3 models ----------
MODELS={"SmolLM2":dict(base="outputs/rev_base_s*.jsonl",pp="outputs/rev_pp_%s_s*.jsonl",
                       gap="outputs/rev_gap_%s_s*.jsonl",lm="outputs/rev_lm_%s_s*.jsonl",caa="outputs/rev_caa_%s_s*.jsonl"),
        "Llama":dict(base="outputs/mm_llm_multi_base_s*.jsonl",pp="outputs/mm_llm_multi_pp_%s_s*.jsonl",
                     gap="outputs/mm_llm_multi_gap_%s_s*.jsonl",lm="outputs/mm_llm_multi_lm_%s_s*.jsonl",caa="outputs/mm_llm_multi_caa_%s_s*.jsonl"),
        "Falcon":dict(base="outputs/mm_flc_multi_base_s*.jsonl",pp="outputs/mm_flc_multi_pp_%s_s*.jsonl",
                      gap="outputs/mm_flc_multi_gap_%s_s*.jsonl",lm="outputs/mm_flc_multi_lm_%s_s*.jsonl",caa="outputs/mm_flc_multi_caa_%s_s*.jsonl")}
for mdl,P in MODELS.items():
    print(f"\n===== MAIN: {mdl} =====")
    print(f"{'method':<11}{'rel':>6}{'succ':>6}{'seen':>6}{'unseen':>7}{'triple':>7}{'d2':>6}{'ms':>7}")
    for name,key,cond_for in [("base","base",None),("prompting","pp","prompting"),("PREADD","pp","preadd"),
                              ("LM-Steer","lm","gap"),("static-CAA","caa","gap"),("GAP","gap","gap")]:
        if key=="base":
            R=rows(P["base"],"base")
            if not R: print(f"{name:<11} --"); continue
            r=dict(rel=m1(R,*POS),succ=succ(R,*POS),seen=joint(R,[POS,INF]),unseen=joint(R,[POS,FRM]),triple=joint(R,[POS,FRM,JOY]),d2=d2(R),ms=ms(R))
        else:
            S=rows(P[key]%"single",cond_for); ID=rows(P[key]%"id2",cond_for); HO=rows(P[key]%"ho2",cond_for); TR=rows(P[key]%"tri",cond_for)
            if not S: print(f"{name:<11} --"); continue
            r=dict(rel=m1(S,*POS),succ=succ(S,*POS),
                   seen=joint(ID,[POS,INF]) if ID else None,unseen=joint(HO,[POS,FRM]) if HO else None,
                   triple=joint(TR,[POS,FRM,JOY]) if TR else None,d2=d2(S),ms=ms(S))
        print(f"{name:<11}{r['rel']:>6}{r['succ']:>6}{str(r['seen']):>6}{str(r['unseen']):>7}{str(r['triple']):>7}{r['d2']:>6}{r['ms']:>7}")

# ---------- LLAMA per-attribute (gap=ll_single, baselines=fb_*) ----------
print("\n===== LLAMA per-attribute relevance =====")
attrs=[("sentiment","positive","Pos"),("sentiment","negative","Neg"),("emotion","joy","Joy"),
       ("emotion","anger","Ang"),("emotion","sadness","Sad"),("style","formal","Frm"),("style","informal","Inf")]
def perattr(label, patfn, method):
    vals=[]
    for d,v,_ in attrs:
        R=rows(patfn(d,v),method); vals.append(m1(R,d,v) if R else None)
    avg=round(np.mean([x for x in vals if x is not None]),2) if any(vals) else None
    print(f"{label:<11}"+"".join(f"{str(x):>6}" for x in vals)+f"{str(avg):>7}")
print(f"{'method':<11}"+"".join(f"{a[2]:>6}" for a in attrs)+f"{'Avg':>7}")
perattr("GAP", lambda d,v: f"outputs/ll_single_{d}_{v}_s*.jsonl","gap")
perattr("prompting", lambda d,v: f"outputs/fb_pp_{d}_{v}_s*.jsonl","prompting")
perattr("PREADD", lambda d,v: f"outputs/fb_pp_{d}_{v}_s*.jsonl","preadd")
perattr("LM-Steer", lambda d,v: f"outputs/fb_lm_{d}_{v}_s*.jsonl","gap")

# ---------- LLAMA ablation (matched strength) ----------
print("\n===== LLAMA ablation (matched strength 2) =====")
for av in ["core","int","caa","full"]:
    S=rows(f"outputs/ll_abl_{av}_single_s*.jsonl","gap"); HO=rows(f"outputs/ll_abl_{av}_ho2_s*.jsonl","gap")
    print(f"{av:<8} rel={m1(S,*POS) if S else '--'}  unseen={joint(HO,[POS,FRM]) if HO else '--'}")

# ---------- LLAMA hard ----------
print("\n===== LLAMA hard control =====")
def verify(t,k):
    t=t.lower()
    return float({"keyword":"ocean" in t,"length":len(t.split())<=20,"structure":"?" in t}[k])
for h in ["keyword","length","structure"]:
    for mth in ["base","prompting","gap"]:
        R=rows(f"outputs/ll_hard_{h}_s*.jsonl",mth)
        s=round(float(np.mean([verify(r["text"],h) for r in R])),2) if R else "--"
        print(f"  {h:<10}{mth:<10}{s}")

# ---------- LLAMA signed ----------
print("\n===== LLAMA signed =====")
for a,tag in [(-1.0,"m1p0"),(-0.5,"m0p5"),(0.0,"0p0"),(0.5,"0p5"),(1.0,"1p0")]:
    R=rows(f"outputs/ll_sgn_{tag}_s*.jsonl","gap"); print(f"  alpha={a:+.1f}  P(pos)={m1(R,*POS) if R else '--'}")
print("\n[done]")
