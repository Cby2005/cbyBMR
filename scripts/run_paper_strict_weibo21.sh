#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "${OUTPUT_DIR:-outputs/paper_strict}" logs

export BMR_PAPER_STRICT=1
python scripts/check_paper_assets.py --dataset Weibo_21

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./UAMFD.py \
  -train_dataset Weibo_21 \
  -test_dataset Weibo_21 \
  -batch_size 24 \
  -epochs 50 \
  -val 0 \
  -is_sample_positive 1.0 \
  -duplicate_fake_times 0 \
  -network_arch UAMFD \
  -is_filter 0 \
  -not_on_12 0 \
  -output_file "${OUTPUT_DIR:-outputs/paper_strict}" \
  2>&1 | tee logs/bmr_paper_strict_weibo21.log
