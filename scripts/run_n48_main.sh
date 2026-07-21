#!/usr/bin/env bash
# n48-as-main-config: regenerate Falcon main-table GAP cells on the 116-prompt eval set.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_CUDA=1
CKPT=models/controller/flc_multi_n48/full.pt
P=data/prompts/eval_large.jsonl
for SEED in 1 2; do
  for C in single id2 ho2 tri; do
    OUT=outputs/main48_gap_${C}_s${SEED}.jsonl
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    python scripts/decode_gap_control.py --config configs/_gt_g2_rt1p0_rm25_${C}.yaml \
      --ckpt $CKPT --methods gap --prompts $P --seed $SEED --out $OUT || exit 1
  done
done
echo N48_MAIN_DONE
