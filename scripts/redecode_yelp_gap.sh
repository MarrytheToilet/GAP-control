#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_CUDA=1 GAPCTRL_ATTRS=yelp
# remove stale GAP outputs so the idempotent driver regenerates them with the fixed gate
rm -f outputs/cmg_yelp_orig_gap_*.jsonl outputs/cmg_yelp_ho0_gap_*.jsonl outputs/cmg_yelp_ho1_gap_*.jsonl outputs/cmg_yelp_acd0_gap_*.jsonl outputs/cmg_yelp_acd1_gap_*.jsonl
python scripts/compmctg_run.py --dataset Yelp --stage decode
