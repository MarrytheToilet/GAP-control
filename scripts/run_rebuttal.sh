#!/usr/bin/env bash
# Comprehensive rebuttal experiments (tmux). ALL baselines x ALL conditions x 3 seeds,
# matched-perturbation, signed control, hard constraints. Serial (controlled GPU util).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/rebuttal.log; : > "$LOG"
P=data/prompts/eval_std20.jsonl
GAP=models/controller/smol_multi/full.pt
LM=models/controller/smol_multi/lmsteer.pt
CAA=models/controller/smol_multi/caa.pt
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
dec(){ python scripts/decode_gap_control.py "$@" >>"$LOG" 2>&1; }

# ---- per-method, per-condition configs ----
python3 - <<'PY'
import yaml
gap=yaml.safe_load(open("configs/_abl_full.yaml"))      # full controller, rho50 str2
lm =yaml.safe_load(open("configs/_mm_lmsteer.yaml"))    # lmsteer rho4 str1
caa=yaml.safe_load(open("configs/_abl_caa.yaml"))       # static-CAA controller
conds={"single":[{"dim":"sentiment","value":"positive","alpha":1.0}],
       "id2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"informal","alpha":1.0}],
       "ho2":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0}],
       "tri":[{"dim":"sentiment","value":"positive","alpha":1.0},{"dim":"style","value":"formal","alpha":1.0},{"dim":"emotion","value":"joy","alpha":1.0}]}
def setup(base,**ov):
    c=dict(base); c.update(ov); return c
for k,ctl in conds.items():
    yaml.safe_dump(setup(gap,control=ctl,decode_strength=2.0),open(f"configs/_cv_gap_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(setup(lm ,control=ctl),open(f"configs/_cv_lm_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(setup(caa,control=ctl,rho_min=4.0,rho_max=4.0,decode_strength=4.0),open(f"configs/_cv_caa_{k}.yaml","w"),sort_keys=False)
    yaml.safe_dump(setup(gap,control=ctl),open(f"configs/_cv_pp_{k}.yaml","w"),sort_keys=False)         # for base/prompting/preadd/fudge
# matched-perturbation: gap single, strengths
for s in [1.0,1.5,2.0,3.0]:
    yaml.safe_dump(setup(gap,control=conds["single"],decode_strength=s),open(f"configs/_cv_str{str(s).replace('.','p')}.yaml","w"),sort_keys=False)
# signed control
for a in [-1.0,-0.5,0.0,0.5,1.0]:
    yaml.safe_dump(setup(gap,control=[{"dim":"sentiment","value":"positive","alpha":a}]),open(f"configs/_cv_sgn{str(a).replace('.','p').replace('-','m')}.yaml","w"),sort_keys=False)
print("configs generated")
PY
say "configs generated"

# ============ main table: all methods x all conditions x 3 seeds ============
for SEED in 1 2 3; do
  say "MAIN seed=$SEED"
  dec --config configs/_cv_pp_single.yaml --ckpt $GAP --methods base --prompts $P --seed $SEED --out outputs/rev_base_s$SEED.jsonl
  for C in single id2 ho2 tri; do
    dec --config configs/_cv_pp_$C.yaml  --ckpt $GAP --methods prompting,preadd --prompts $P --seed $SEED --out outputs/rev_pp_${C}_s$SEED.jsonl
    dec --config configs/_cv_gap_$C.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/rev_gap_${C}_s$SEED.jsonl
    dec --config configs/_cv_lm_$C.yaml  --ckpt $LM  --methods gap --prompts $P --seed $SEED --out outputs/rev_lm_${C}_s$SEED.jsonl
    dec --config configs/_cv_caa_$C.yaml --ckpt $CAA --methods gap --prompts $P --seed $SEED --out outputs/rev_caa_${C}_s$SEED.jsonl
  done
done

# ============ FUDGE (slow inference-time baseline): 2 seeds, 3 samples ============
for SEED in 1 2; do
  say "FUDGE seed=$SEED"
  for C in single ho2 tri; do
    python3 -c "import yaml;c=yaml.safe_load(open('configs/_cv_pp_$C.yaml'));c['samples_per_prompt']=3;yaml.safe_dump(c,open('configs/_cv_fudge_$C.yaml','w'),sort_keys=False)"
    dec --config configs/_cv_fudge_$C.yaml --methods fudge --prompts $P --seed $SEED --out outputs/rev_fudge_${C}_s$SEED.jsonl
  done
done

# ============ matched-perturbation (gap strength sweep, 3 seeds) ============
for SEED in 1 2 3; do
  say "matched-PPL seed=$SEED"
  for S in 1p0 1p5 2p0 3p0; do
    dec --config configs/_cv_str$S.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/rev_str${S}_s$SEED.jsonl
  done
done

# ============ signed control (3 seeds) ============
for SEED in 1 2 3; do
  say "signed seed=$SEED"
  for A in m1p0 m0p5 0p0 0p5 1p0; do
    dec --config configs/_cv_sgn$A.yaml --ckpt $GAP --methods gap --prompts $P --seed $SEED --out outputs/rev_sgn${A}_s$SEED.jsonl
  done
done

# ============ hard constraints: teacher + train + decode ============
say "HARD: teacher + train"
python3 - <<'PY'
import yaml,shutil,os
c=yaml.safe_load(open("configs/smol_multi.yaml")); c["task"]="smol_hard"
c["atomics"]=[{"dim":"keyword","keywords":["ocean"]},{"dim":"length","value":"short"},{"dim":"structure","value":"interrogative"}]
c["holdout_combos"]=[]; c["control"]=[{"dim":"keyword","alpha":1.0,"keywords":["ocean"]}]
yaml.safe_dump(c,open("configs/smol_hard.yaml","w"),sort_keys=False)
for s in ["prefixes/smol_multi.jsonl","prompts/smol_multi.jsonl"]:
    shutil.copy(f"data/{s}", f"data/{s.replace('smol_multi','smol_hard')}")
PY
python scripts/estimate_teacher_multi.py --config configs/smol_hard.yaml >>"$LOG" 2>&1
python3 -c "import yaml;c=yaml.safe_load(open('configs/smol_hard.yaml'));c.update(controller_type='compositional',use_steering=True,interaction=True,rho_min=0.5,rho_max=10.0,decode_strength=3.0,samples_per_prompt=6,epochs=40,lexical_strength=8.0);yaml.safe_dump(c,open('configs/_cv_hard.yaml','w'),sort_keys=False)"
python scripts/train_compositional.py --config configs/_cv_hard.yaml --out models/controller/smol_hard/gap.pt >>"$LOG" 2>&1
for HA in keyword length structure; do
  for SEED in 1 2 3; do
    python3 -c "
import yaml; c=yaml.safe_load(open('configs/_cv_hard.yaml'))
c['control']={'keyword':[{'dim':'keyword','alpha':1.0,'keywords':['ocean']}],'length':[{'dim':'length','value':'short','alpha':1.0}],'structure':[{'dim':'structure','value':'interrogative','alpha':1.0}]}['$HA']
yaml.safe_dump(c,open('configs/_cv_hard_$HA.yaml','w'),sort_keys=False)"
    dec --config configs/_cv_hard_$HA.yaml --ckpt models/controller/smol_hard/gap.pt --methods gap,base,prompting --prompts $P --seed $SEED --out outputs/rev_hard_${HA}_s$SEED.jsonl
  done
done

say "ALL REBUTTAL EXPERIMENTS DONE"
