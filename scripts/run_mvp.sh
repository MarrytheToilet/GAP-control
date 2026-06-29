#!/usr/bin/env bash
# End-to-end MVP-0 pipeline (handbook 4.1). Usage: bash scripts/run_mvp.sh [config]
set -e
CFG="${1:-configs/sentiment_mvp.yaml}"
cd "$(dirname "$0")/.."

echo "### 1/5 build prefixes"
python scripts/build_prefixes.py --config "$CFG"
echo "### 2/5 estimate teacher advantage"
python scripts/estimate_teacher_advantage.py --config "$CFG"
echo "### 3/5 train controller"
python scripts/train_controller.py --config "$CFG"
echo "### 4/5 decode (gap + baselines)"
python scripts/decode_gap_control.py --config "$CFG" \
  --methods prompt,instruct,cfg,preadd,fudge,bon,gap
echo "### 5/5 evaluate"
python scripts/evaluate.py --config "$CFG"
