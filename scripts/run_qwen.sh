#!/usr/bin/env bash
# Third base model: gemma-2-2b-it. Full pipeline (plain full-CAA config, consistent
# with Falcon/SmolLM2): prefixes -> steering -> teacher -> train full/lmsteer/caa ->
# decode main conditions x methods x 3 seeds (for the cross-model main table).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/qwen.log; : > "$LOG"
P=data/prompts/eval_std20.jsonl
M=Qwen2.5-1.5B; T=qwn_multi
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ echo ">> $*" >>"$LOG"; python "$@" >>"$LOG" 2>&1; }
dec(){ python scripts/decode_gap_control.py "$@" >>"$LOG" 2>&1; }

say "================ QWEN pipeline ($M) ================"
python3 - "$M" "$T" <<'PY'
import sys,yaml,shutil,os
m,t=sys.argv[1],sys.argv[2]
base=yaml.safe_load(open("configs/smol_multi.yaml")); base.update(base_model=m,task=t)
yaml.safe_dump(base,open(f"configs/{t}.yaml","w"),sort_keys=False)
shutil.copy("data/prompts/smol_multi.jsonl", f"data/prompts/{t}.jsonl")
for src,tag in [("_abl_full","full"),("_mm_lmsteer","lmsteer"),("_abl_caa","caa")]:
    c=yaml.safe_load(open(f"configs/{src}.yaml")); c.update(base_model=m,task=t)
    yaml.safe_dump(c,open(f"configs/{t}_{tag}.yaml","w"),sort_keys=False)
print("configs ok")
PY

say "[qwen] build prefixes"; run scripts/build_prefixes.py --config configs/$T.yaml
say "[qwen] compute steering (CAA)"; run scripts/compute_steering.py --config configs/$T.yaml --pairs
say "[qwen] estimate teacher"; run scripts/estimate_teacher_multi.py --config configs/$T.yaml
mkdir -p models/controller/$T
say "[qwen] train full"; run scripts/train_compositional.py --config configs/${T}_full.yaml --out models/controller/$T/full.pt
say "[qwen] train lmsteer"; run scripts/train_compositional.py --config configs/${T}_lmsteer.yaml --out models/controller/$T/lmsteer.pt
say "[qwen] train caa"; run scripts/train_compositional.py --config configs/${T}_caa.yaml --out models/controller/$T/caa.pt
GAP=models/controller/$T/full.pt; LM=models/controller/$T/lmsteer.pt; CAA=models/controller/$T/caa.pt

python3 - "$T" <<'PY'
import sys,yaml
t=sys.argv[1]
gap=yaml.safe_load(open(f"configs/{t}_full.yaml")); lm=yaml.safe_load(open(f"configs/{t}_lmsteer.yaml")); caa=yaml.safe_load(open(f"configs/{t}_caa.yaml"))
conds={"single":[{"dim":"sentiment","value":"positive","alpha":1.0}],
 "id2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"informal","alpha":1.0}],
 "ho2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0}],
 "tri":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0},{"dim":"emotion","value":"joy","alpha":1.0}]}
def s(b,**ov):
    c=dict(b);c.update(ov);return c
for k,ctl in conds.items():
    yaml.safe_dump(s(gap,control=ctl,decode_strength=4.0,entropy_gate=2.0),open(f"configs/_gm_gap_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(lm,control=ctl),open(f"configs/_gm_lm_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(caa,control=ctl,rho_min=4.0,rho_max=4.0,decode_strength=4.0),open(f"configs/_gm_caa_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(gap,control=ctl),open(f"configs/_gm_pp_{k}.yaml","w"),sort_keys=False)
print("decode configs ok")
PY

for SEED in 1 2 3; do
  say "[qwen] decode seed=$SEED"
  dec --config configs/_gm_pp_single.yaml --ckpt $GAP --methods base --prompts $P --seed $SEED --out outputs/mm_qwn_multi_base_s$SEED.jsonl
  for C in single id2 ho2 tri; do
    dec --config configs/_gm_pp_$C.yaml  --ckpt $GAP --methods prompting,preadd --prompts $P --seed $SEED --out outputs/mm_qwn_multi_pp_${C}_s$SEED.jsonl
    dec --config configs/_gm_gap_$C.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/mm_qwn_multi_gap_${C}_s$SEED.jsonl
    dec --config configs/_gm_lm_$C.yaml  --ckpt $LM  --methods gap --prompts $P --seed $SEED --out outputs/mm_qwn_multi_lm_${C}_s$SEED.jsonl
  done
done
say "ALL QWEN DONE"
