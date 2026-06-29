#!/usr/bin/env bash
# Decode data for the two heatmap figures (control specificity + 2-D control plane).
# Doubles as the "more single attributes / more combinations" coverage.
# Reuses the trained GAP controller (smol_multi/full.pt) -- NO retraining.
# Waits for the main rebuttal to finish first so GPU stays serial.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/figdata.log; : > "$LOG"
P=data/prompts/eval_std20.jsonl
GAP=models/controller/smol_multi/full.pt
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
dec(){ python scripts/decode_gap_control.py "$@" >>"$LOG" 2>&1; }

# ---- wait for the main rebuttal to complete (serial GPU) ----
say "waiting for rebuttal to finish..."
while ! grep -q "ALL REBUTTAL EXPERIMENTS DONE" logs/rebuttal.log 2>/dev/null; do sleep 60; done
say "rebuttal done -> starting figure-data decodes"

# ---- build per-condition GAP configs (full controller, decode_strength=2.0) ----
python3 - <<'PY'
import yaml
gap=yaml.safe_load(open("configs/_abl_full.yaml"))
def setup(ctl,**ov):
    c=dict(gap); c.update(control=ctl,decode_strength=2.0); c.update(ov); return c
# (a) specificity: one single attribute at a time
singles={
 "sentiment_positive":[{"dim":"sentiment","value":"positive","alpha":1.0}],
 "sentiment_negative":[{"dim":"sentiment","value":"negative","alpha":1.0}],
 "emotion_joy":[{"dim":"emotion","value":"joy","alpha":1.0}],
 "emotion_anger":[{"dim":"emotion","value":"anger","alpha":1.0}],
 "emotion_sadness":[{"dim":"emotion","value":"sadness","alpha":1.0}],
 "style_formal":[{"dim":"style","value":"formal","alpha":1.0}],
 "style_informal":[{"dim":"style","value":"informal","alpha":1.0}],
}
for k,ctl in singles.items():
    yaml.safe_dump(setup(ctl),open(f"configs/_fd_single_{k}.yaml","w"),sort_keys=False)
# (b) control plane: alpha grid sentiment-positive x style-formal
for a1 in [0.0,0.5,1.0,1.5]:
    for a2 in [0.0,0.5,1.0,1.5]:
        ctl=[{"dim":"sentiment","value":"positive","alpha":a1},
             {"dim":"style","value":"formal","alpha":a2}]
        t=f"a{str(a1).replace('.','p')}_b{str(a2).replace('.','p')}"
        yaml.safe_dump(setup(ctl),open(f"configs/_fd_grid_{t}.yaml","w"),sort_keys=False)
print("figdata configs generated")
PY

# ---- (a) specificity decodes ----
for K in sentiment_positive sentiment_negative emotion_joy emotion_anger emotion_sadness style_formal style_informal; do
  say "specificity $K"
  dec --config configs/_fd_single_$K.yaml --ckpt $GAP --methods gap --prompts $P --seed 1 \
      --out outputs/rev_single_${K}_s1.jsonl
done

# ---- (b) control-plane decodes ----
for a1 in 0p0 0p5 1p0 1p5; do
  for a2 in 0p0 0p5 1p0 1p5; do
    say "grid a=$a1 b=$a2"
    dec --config configs/_fd_grid_a${a1}_b${a2}.yaml --ckpt $GAP --methods gap --prompts $P --seed 1 \
        --out outputs/rev_grid_a${a1}_b${a2}_s1.jsonl
  done
done

say "ALL FIGDATA DONE"
