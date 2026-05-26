#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MANIFEST_DIR="${MANIFEST_DIR:-/root/autodl-tmp/event-radar-repro/data/duola_title_body}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/recovery-project/data}"
TEXT_MODEL="${TEXT_MODEL:-/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/baseline_results/bmr/duola}"
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

DIAGNOSTIC_ARGS=()
DEFAULT_LOG_FILE="$PROJECT_ROOT/logs/bmr_manifest_duola_paper_strict.log"
if [[ "${DIAGNOSE_ACTIVATIONS:-0}" == "1" ]]; then
  if [[ "${BATCH_STATS_EVAL:-0}" == "1" ]]; then
    echo "DIAGNOSE_ACTIVATIONS=1 requires standard model.eval(); unset BATCH_STATS_EVAL." >&2
    exit 2
  fi
  DEFAULT_LOG_FILE="$PROJECT_ROOT/logs/bmr_manifest_duola_strict_activation_trace.log"
  DIAGNOSTIC_ARGS+=(
    --diagnose_activations
    --diagnostic_abs_threshold "${DIAGNOSTIC_ABS_THRESHOLD:-1000000}"
    --diagnostic_output "${DIAGNOSTIC_OUTPUT:-$OUTPUT_DIR/activation_diagnostics.json}"
  )
fi
LOG_FILE="${LOG_FILE:-$DEFAULT_LOG_FILE}"

"$PYTHON" scripts/check_manifest_assets.py --manifest_dir "$MANIFEST_DIR" --image_root "$IMAGE_ROOT"
BMR_PAPER_STRICT=1 BMR_PAPER_MLP="${BMR_PAPER_MLP:-1}" \
BMR_PATTERN_BACKBONE="${BMR_PATTERN_BACKBONE:-paper_inception_v3}" \
BMR_BERT_UNCASED="$TEXT_MODEL" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON" train_manifest_paper_strict.py \
  --manifest_dir "$MANIFEST_DIR" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --text_model "$TEXT_MODEL" \
  --dataset_key gossip \
  --mlp_protocol "${MLP_PROTOCOL:-paper_elu}" \
  --pattern_backbone "${PATTERN_BACKBONE:-paper_inception_v3}" \
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
  "${DIAGNOSTIC_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
