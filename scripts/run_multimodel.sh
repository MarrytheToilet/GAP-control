#!/usr/bin/env bash
# Multi-model generality table: replicate the full SmolLM2 pipeline on
# Llama-3.2-3B and Falcon3-3B-Base. Per model: prefixes -> steering -> teacher
# -> train {GAP-full, LM-Steer, CAA} -> decode {single,id2,ho2,tri} x methods x seeds.
# Classifiers (bge) are shared/model-agnostic. Serial; waits for figdata first.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/multimodel.log; : > "$LOG"
P=data/prompts/eval_std20.jsonl
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ echo ">> $*" >>"$LOG"; python "$@" >>"$LOG" 2>&1; }
dec(){ python scripts/decode_gap_control.py "$@" >>"$LOG" 2>&1; }

# ---- stay serial on GPU: wait for the earlier queues ----
say "waiting for rebuttal + figdata to finish..."
while ! grep -q "ALL REBUTTAL EXPERIMENTS DONE" logs/rebuttal.log 2>/dev/null; do sleep 60; done
while ! grep -q "ALL FIGDATA DONE" logs/figdata.log 2>/dev/null; do sleep 60; done
say "earlier queues done -> starting multi-model pipeline"

MODELS=("Llama-3.2-3B:llm_multi" "Falcon3-3B-Base:flc_multi")

for ENTRY in "${MODELS[@]}"; do
  MODEL="${ENTRY%%:*}"; TASK="${ENTRY##*:}"
  say "================ MODEL=$MODEL TASK=$TASK ================"

  # ---- per-model configs (base + 3 training variants), derived from the SmolLM2 ones ----
  python3 - "$MODEL" "$TASK" <<'PY'
import sys,yaml,shutil,os
model,task=sys.argv[1],sys.argv[2]
base=yaml.safe_load(open("configs/smol_multi.yaml")); base.update(base_model=model,task=task)
yaml.safe_dump(base,open(f"configs/{task}.yaml","w"),sort_keys=False)
# reuse the same neutral prompt seeds; prefixes are regenerated per base
shutil.copy("data/prompts/smol_multi.jsonl", f"data/prompts/{task}.jsonl")
for src,tag in [("_abl_full","full"),("_mm_lmsteer","lmsteer"),("_abl_caa","caa")]:
    c=yaml.safe_load(open(f"configs/{src}.yaml")); c.update(base_model=model,task=task)
    yaml.safe_dump(c,open(f"configs/{task}_{tag}.yaml","w"),sort_keys=False)
print("configs:",task)
PY

  # ---- data + teacher (model-specific) ----
  say "[$TASK] build prefixes"; run scripts/build_prefixes.py --config configs/$TASK.yaml
  say "[$TASK] compute steering (CAA dirs)"; run scripts/compute_steering.py --config configs/$TASK.yaml --pairs
  say "[$TASK] estimate teacher"; run scripts/estimate_teacher_multi.py --config configs/$TASK.yaml

  # ---- train 3 controllers ----
  mkdir -p models/controller/$TASK
  say "[$TASK] train GAP-full"; run scripts/train_compositional.py --config configs/${TASK}_full.yaml    --out models/controller/$TASK/full.pt
  say "[$TASK] train LM-Steer"; run scripts/train_compositional.py --config configs/${TASK}_lmsteer.yaml --out models/controller/$TASK/lmsteer.pt
  say "[$TASK] train CAA";      run scripts/train_compositional.py --config configs/${TASK}_caa.yaml     --out models/controller/$TASK/caa.pt
  GAP=models/controller/$TASK/full.pt; LM=models/controller/$TASK/lmsteer.pt; CAA=models/controller/$TASK/caa.pt

  # ---- per-condition decode configs ----
  python3 - "$TASK" <<'PY'
import sys,yaml
task=sys.argv[1]
gap=yaml.safe_load(open(f"configs/{task}_full.yaml"))
lm =yaml.safe_load(open(f"configs/{task}_lmsteer.yaml"))
caa=yaml.safe_load(open(f"configs/{task}_caa.yaml"))
conds={"single":[{"dim":"sentiment","value":"positive","alpha":1.0}],
       "id2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"informal","alpha":1.0}],
       "ho2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0}],
       "tri":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0},{"dim":"emotion","value":"joy","alpha":1.0}]}
def s(b,**ov):
    c=dict(b); c.update(ov); return c
for k,ctl in conds.items():
    yaml.safe_dump(s(gap,control=ctl,decode_strength=2.0),open(f"configs/_mm_{task}_gap_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(lm ,control=ctl),open(f"configs/_mm_{task}_lm_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(caa,control=ctl,rho_min=4.0,rho_max=4.0,decode_strength=4.0),open(f"configs/_mm_{task}_caa_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(s(gap,control=ctl),open(f"configs/_mm_{task}_pp_{k}.yaml","w"),sort_keys=False)
print("decode configs:",task)
PY

  # ---- decode: all methods x conditions x 3 seeds ----
  for SEED in 1 2 3; do
    say "[$TASK] decode seed=$SEED"
    dec --config configs/_mm_${TASK}_pp_single.yaml --ckpt $GAP --methods base --prompts $P --seed $SEED --out outputs/mm_${TASK}_base_s$SEED.jsonl
    for C in single id2 ho2 tri; do
      dec --config configs/_mm_${TASK}_pp_$C.yaml  --ckpt $GAP --methods prompting,preadd --prompts $P --seed $SEED --out outputs/mm_${TASK}_pp_${C}_s$SEED.jsonl
      dec --config configs/_mm_${TASK}_gap_$C.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/mm_${TASK}_gap_${C}_s$SEED.jsonl
      dec --config configs/_mm_${TASK}_lm_$C.yaml  --ckpt $LM  --methods gap --prompts $P --seed $SEED --out outputs/mm_${TASK}_lm_${C}_s$SEED.jsonl
      dec --config configs/_mm_${TASK}_caa_$C.yaml --ckpt $CAA --methods gap --prompts $P --seed $SEED --out outputs/mm_${TASK}_caa_${C}_s$SEED.jsonl
    done
  done
  say "[$TASK] DONE"
done

say "ALL MULTIMODEL DONE"
