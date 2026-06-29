#!/usr/bin/env bash
# Decode-only jobs to replace the last paper placeholders with real data:
#  (A) per-attribute BASELINES (prompting/preadd/lmsteer) for Table 3
#  (B) matched-strength ablation (core/int/caa/full at decode_strength=2) for Table 2
# Reuses trained controllers; waits for multimodel so GPU stays serial.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/fill.log; : > "$LOG"
P=data/prompts/eval_std20.jsonl
GAP=models/controller/smol_multi/full.pt
LM=models/controller/smol_multi/lmsteer.pt
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
dec(){ python scripts/decode_gap_control.py "$@" >>"$LOG" 2>&1; }

say "waiting for multimodel to finish..."
while ! grep -q "ALL MULTIMODEL DONE" logs/multimodel.log 2>/dev/null; do sleep 60; done
say "multimodel done -> filling remaining real numbers"

# ---- build per-attribute configs (base GAP config, single attr each) ----
python3 - <<'PY'
import yaml
g=yaml.safe_load(open("configs/_abl_full.yaml")); lm=yaml.safe_load(open("configs/_mm_lmsteer.yaml"))
attrs={"sentiment_positive":("sentiment","positive"),"sentiment_negative":("sentiment","negative"),
 "emotion_joy":("emotion","joy"),"emotion_anger":("emotion","anger"),"emotion_sadness":("emotion","sadness"),
 "style_formal":("style","formal"),"style_informal":("style","informal")}
def s(b,ctl,**ov):
    c=dict(b);c.update(control=ctl);c.update(ov);return c
for k,(d,v) in attrs.items():
    ctl=[{"dim":d,"value":v,"alpha":1.0}]
    yaml.safe_dump(s(g,ctl),open(f"configs/_fb_pp_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(lm,ctl),open(f"configs/_fb_lm_{k}.yaml","w"),sort_keys=False)
# matched-strength ablation decode configs (strength 2, like full)
abl={"core":"_abl_core","int":"_abl_int","caa":"_abl_caa","full":"_abl_full"}
conds={"single":[{"dim":"sentiment","value":"positive","alpha":1.0}],
 "id2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"informal","alpha":1.0}],
 "ho2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0}],
 "tri":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0},{"dim":"emotion","value":"joy","alpha":1.0}]}
for a,src in abl.items():
    base=yaml.safe_load(open(f"configs/{src}.yaml"))
    for ck,ctl in conds.items():
        yaml.safe_dump(s(base,ctl,decode_strength=2.0),open(f"configs/_fa_{a}_{ck}.yaml","w"),sort_keys=False)
print("fill configs done")
PY

# ---- (A) per-attribute baselines (3 seeds) ----
for K in sentiment_positive sentiment_negative emotion_joy emotion_anger emotion_sadness style_formal style_informal; do
  for SEED in 1 2 3; do
    say "perattr baselines $K seed=$SEED"
    dec --config configs/_fb_pp_$K.yaml --ckpt $GAP --methods prompting,preadd --prompts $P --seed $SEED --out outputs/fb_pp_${K}_s$SEED.jsonl
    dec --config configs/_fb_lm_$K.yaml --ckpt $LM  --methods gap --prompts $P --seed $SEED --out outputs/fb_lm_${K}_s$SEED.jsonl
  done
done

# ---- (B) matched-strength ablation (3 seeds) ----
declare -A CK=( [core]=models/controller/smol_multi/core.pt [int]=models/controller/smol_multi/int.pt [caa]=models/controller/smol_multi/caa.pt [full]=models/controller/smol_multi/full.pt )
for A in core int caa full; do
  for C in single id2 ho2 tri; do
    for SEED in 1 2 3; do
      say "ablation $A $C seed=$SEED (strength 2)"
      dec --config configs/_fa_${A}_$C.yaml --ckpt ${CK[$A]} --methods gap --prompts $P --seed $SEED --out outputs/fa_${A}_${C}_s$SEED.jsonl
    done
  done
done

say "ALL FILL DONE"
