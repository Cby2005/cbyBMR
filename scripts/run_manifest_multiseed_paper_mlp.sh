#!/usr/bin/env bash
set -euo pipefail

# Run the paper-aligned MLP/frozen-encoder protocol for independent seeds.
# Best checkpoints are selected within each run using validation only.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATASET="${DATASET:-duola}"
SEEDS="${SEEDS:-13 21 42 87 100}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-8}"

case "$DATASET" in
  duola)
    RUNNER="$PROJECT_ROOT/scripts/run_manifest_duola_paper_strict.sh"
    BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/baseline_results/bmr/duola_paper_mlp_multiseed}"
    ;;
  weibo)
    RUNNER="$PROJECT_ROOT/scripts/run_manifest_weibo_paper_strict.sh"
    BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/baseline_results/bmr/weibo_paper_mlp_multiseed}"
    ;;
  *)
    echo "DATASET must be duola or weibo, got: $DATASET" >&2
    exit 2
    ;;
esac

mkdir -p "$BASE_OUTPUT" "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

completed=0
failed=0
for seed in $SEEDS; do
  output_dir="$BASE_OUTPUT/seed_$seed"
  log_file="$PROJECT_ROOT/logs/bmr_${DATASET}_paper_mlp_seed_${seed}.log"
  mkdir -p "$output_dir"
  echo "Running BMR paper-MLP protocol: dataset=$DATASET seed=$seed epochs=$EPOCHS output=$output_dir"
  if OUTPUT_DIR="$output_dir" LOG_FILE="$log_file" SEED="$seed" EPOCHS="$EPOCHS" \
       BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" PATIENCE="$PATIENCE" \
       MLP_PROTOCOL="paper_elu" \
       BATCH_STATS_EVAL="${BATCH_STATS_EVAL:-0}" \
       bash "$RUNNER"; then
    completed=$((completed + 1))
  else
    failed=$((failed + 1))
    printf 'seed=%s failed; inspect %s\n' "$seed" "$log_file" > "$output_dir/run_failed.txt"
  fi
done

"$PYTHON" scripts/summarize_multiseed_metrics.py \
  --root "$BASE_OUTPUT" \
  --seeds $SEEDS \
  --output "$BASE_OUTPUT/multiseed_summary.json" \
  | tee "$PROJECT_ROOT/logs/bmr_${DATASET}_paper_mlp_multiseed_summary.log"

echo "Completed runs: $completed; failed runs: $failed"
if [[ "$failed" -gt 0 ]]; then
  echo "Some strict runs failed. Their missing results are recorded in the summary; do not report them as metrics." >&2
fi
