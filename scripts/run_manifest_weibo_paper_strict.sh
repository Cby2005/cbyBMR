#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MANIFEST_DIR="${MANIFEST_DIR:-/root/autodl-tmp/event-radar-repro/data/weibo_event_radar_json_full}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/event-radar-repro/data/weibo}"
TEXT_MODEL="${TEXT_MODEL:-/root/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/baseline_results/bmr/weibo}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${THREADS:-8}"
export MKL_NUM_THREADS="${THREADS:-8}"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

STABILITY_ARGS=()
if [[ "${BATCH_STATS_EVAL:-0}" == "1" ]]; then
  STABILITY_ARGS+=(--batch_stats_eval)
fi
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/bmr_manifest_weibo_paper_strict.log}"

"$PYTHON" scripts/check_manifest_assets.py --manifest_dir "$MANIFEST_DIR" --image_root "$IMAGE_ROOT"
BMR_PAPER_STRICT=1 BMR_BERT_CHINESE="$TEXT_MODEL" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON" train_manifest_paper_strict.py \
  --manifest_dir "$MANIFEST_DIR" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --text_model "$TEXT_MODEL" \
  --dataset_key weibo \
  --mlp_protocol "${MLP_PROTOCOL:-paper_elu}" \
  --batch_size "${BATCH_SIZE:-24}" \
  --epochs "${EPOCHS:-50}" \
  --learning_rate "${LEARNING_RATE:-0.0001}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --seed "${SEED:-42}" \
  --patience "${PATIENCE:-8}" \
  --threshold 0.5 \
  --real_label 0 \
  --fake_label 1 \
  --device cuda:0 \
  "${STABILITY_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
