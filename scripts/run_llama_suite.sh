#!/usr/bin/env bash
# Full experimental suite on Llama-3.2-3B so it can be the PRIMARY model:
# trains the ablation (core/int) + hard controllers, then decodes the full eval matrix
# (per-attribute, signed, matched-perturbation, control-plane grid, ablation, hard).
# Llama already has: teacher cache (llm_multi), full/lmsteer/caa controllers, main conds.
# Outputs use the ll_* prefix. Serial; waits for multimodel + fill so GPU stays serial.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/llama_suite.log; : > "$LOG"
P=data/prompts/eval_std20.jsonl
M=Llama-3.2-3B; T=llm_multi
GAP=models/controller/$T/full.pt; LM=models/controller/$T/lmsteer.pt; CAA=models/controller/$T/caa.pt
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ echo ">> $*" >>"$LOG"; python "$@" >>"$LOG" 2>&1; }
dec(){ python scripts/decode_gap_control.py "$@" >>"$LOG" 2>&1; }

say "waiting for multimodel + fill to finish..."
while ! grep -q "ALL MULTIMODEL DONE" logs/multimodel.log 2>/dev/null; do sleep 60; done
while ! grep -q "ALL FILL DONE" logs/fill.log 2>/dev/null; do sleep 60; done
say "earlier queues done -> Llama full suite"

# ===== 1. train ablation controllers core + int (full/caa already trained) =====
python3 - "$M" "$T" <<'PY'
import sys,yaml
m,t=sys.argv[1],sys.argv[2]
for src,tag in [("_abl_core","core"),("_abl_int","int")]:
    c=yaml.safe_load(open(f"configs/{src}.yaml")); c.update(base_model=m,task=t)
    yaml.safe_dump(c,open(f"configs/{t}_{tag}.yaml","w"),sort_keys=False)
print("abl configs")
PY
say "[abl] train core"; run scripts/train_compositional.py --config configs/${T}_core.yaml --out models/controller/$T/core.pt
say "[abl] train int";  run scripts/train_compositional.py --config configs/${T}_int.yaml  --out models/controller/$T/int.pt

# ===== 2. hard controller on Llama (replicate smol_hard) =====
say "[hard] teacher + train"
python3 - "$M" <<'PY'
import yaml,shutil,os
m=yaml.safe_load(open("configs/smol_hard.yaml")); m["task"]="llm_hard"; m["base_model"]=__import__("sys").argv[1]
yaml.safe_dump(m,open("configs/llm_hard.yaml","w"),sort_keys=False)
for s in ["prefixes/smol_multi.jsonl","prompts/smol_multi.jsonl"]:
    dst=f"data/{s.replace('smol_multi','llm_hard')}"
    if not os.path.exists(dst): shutil.copy(f"data/{s}", dst)
hc=yaml.safe_load(open("configs/_cv_hard.yaml")); hc.update(base_model=__import__("sys").argv[1],task="llm_hard")
yaml.safe_dump(hc,open("configs/_cv_hard_llm.yaml","w"),sort_keys=False)
print("hard configs")
PY
run scripts/build_prefixes.py --config configs/llm_hard.yaml
run scripts/estimate_teacher_multi.py --config configs/llm_hard.yaml
run scripts/train_compositional.py --config configs/_cv_hard_llm.yaml --out models/controller/llm_hard/gap.pt

# ===== 3. all decode configs (derived from llm_multi_full / abl / hard) =====
python3 - "$T" <<'PY'
import sys,yaml
t=sys.argv[1]; g=yaml.safe_load(open(f"configs/{t}_full.yaml"))
def s(ctl,**ov):
    c=dict(g);c.update(control=ctl,decode_strength=2.0);c.update(ov);return c
singles={"sentiment_positive":("sentiment","positive"),"sentiment_negative":("sentiment","negative"),
 "emotion_joy":("emotion","joy"),"emotion_anger":("emotion","anger"),"emotion_sadness":("emotion","sadness"),
 "style_formal":("style","formal"),"style_informal":("style","informal")}
for k,(d,v) in singles.items():
    yaml.safe_dump(s([{"dim":d,"value":v,"alpha":1.0}]),open(f"configs/_ll_single_{k}.yaml","w"),sort_keys=False)
for a in [-1.0,-0.5,0.0,0.5,1.0]:
    yaml.safe_dump(s([{"dim":"sentiment","value":"positive","alpha":a}]),open(f"configs/_ll_sgn_{str(a).replace('.','p').replace('-','m')}.yaml","w"),sort_keys=False)
for st in [1.0,1.5,2.0,3.0]:
    yaml.safe_dump(s([{"dim":"sentiment","value":"positive","alpha":1.0}],decode_strength=st),open(f"configs/_ll_str_{str(st).replace('.','p')}.yaml","w"),sort_keys=False)
for a1 in [0.0,0.5,1.0,1.5]:
    for a2 in [0.0,0.5,1.0,1.5]:
        ctl=[{"dim":"sentiment","value":"positive","alpha":a1},{"dim":"style","value":"formal","alpha":a2}]
        yaml.safe_dump(s(ctl),open(f"configs/_ll_grid_a{str(a1).replace('.','p')}_b{str(a2).replace('.','p')}.yaml","w"),sort_keys=False)
# ablation matched-strength configs
conds={"single":[{"dim":"sentiment","value":"positive","alpha":1.0}],
 "id2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"informal","alpha":1.0}],
 "ho2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0}],
 "tri":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0},{"dim":"emotion","value":"joy","alpha":1.0}]}
for av,src in [("core",f"{t}_core"),("int",f"{t}_int"),("caa",f"{t}_caa"),("full",f"{t}_full")]:
    b=yaml.safe_load(open(f"configs/{src}.yaml"))
    for ck,ctl in conds.items():
        bb=dict(b);bb.update(control=ctl,decode_strength=2.0)
        yaml.safe_dump(bb,open(f"configs/_ll_abl_{av}_{ck}.yaml","w"),sort_keys=False)
# hard decode configs
hc=yaml.safe_load(open("configs/_cv_hard_llm.yaml"))
hard={"keyword":[{"dim":"keyword","alpha":1.0,"keywords":["ocean"]}],"length":[{"dim":"length","value":"short","alpha":1.0}],"structure":[{"dim":"structure","value":"interrogative","alpha":1.0}]}
for k,ctl in hard.items():
    cc=dict(hc);cc.update(control=ctl); yaml.safe_dump(cc,open(f"configs/_ll_hard_{k}.yaml","w"),sort_keys=False)
print("decode configs")
PY

# ===== 4. decodes (3 seeds for headline, 1 for grid) =====
for SEED in 1 2 3; do
  say "[Llama] per-attr seed=$SEED"
  for K in sentiment_positive sentiment_negative emotion_joy emotion_anger emotion_sadness style_formal style_informal; do
    dec --config configs/_ll_single_$K.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/ll_single_${K}_s$SEED.jsonl
  done
  say "[Llama] signed + matched seed=$SEED"
  for A in m1p0 m0p5 0p0 0p5 1p0; do dec --config configs/_ll_sgn_$A.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/ll_sgn_${A}_s$SEED.jsonl; done
  for S in 1p0 1p5 2p0 3p0; do dec --config configs/_ll_str_$S.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/ll_str_${S}_s$SEED.jsonl; done
  say "[Llama] ablation seed=$SEED"
  for AV in core int caa full; do
    CK=models/controller/$T/$AV.pt
    for C in single id2 ho2 tri; do dec --config configs/_ll_abl_${AV}_$C.yaml --ckpt $CK --methods gap --prompts $P --seed $SEED --out outputs/ll_abl_${AV}_${C}_s$SEED.jsonl; done
  done
  say "[Llama] hard seed=$SEED"
  for H in keyword length structure; do dec --config configs/_ll_hard_$H.yaml --ckpt models/controller/llm_hard/gap.pt --methods gap,base,prompting --prompts $P --seed $SEED --out outputs/ll_hard_${H}_s$SEED.jsonl; done
done
say "[Llama] control-plane grid (1 seed)"
for a1 in 0p0 0p5 1p0 1p5; do for a2 in 0p0 0p5 1p0 1p5; do
  dec --config configs/_ll_grid_a${a1}_b${a2}.yaml --ckpt $GAP --methods gap --prompts $P --seed 1 --out outputs/ll_grid_a${a1}_b${a2}_s1.jsonl
done; done

say "ALL LLAMA SUITE DONE"
