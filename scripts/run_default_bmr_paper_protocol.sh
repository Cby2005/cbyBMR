#!/usr/bin/env bash
set -euo pipefail

# Default paper-protocol experiment suite: five validation-selected runs per dataset.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS="${DATASETS:-duola weibo}"
SEEDS="${SEEDS:-13 21 42 87 100}"

cd "$PROJECT_ROOT"
for dataset in $DATASETS; do
  echo "Starting default BMR paper protocol: dataset=$dataset seeds=$SEEDS"
  DATASET="$dataset" SEEDS="$SEEDS" \
  BMR_PAPER_MLP=1 PATTERN_BACKBONE=paper_inception_v3 \
  bash scripts/run_manifest_multiseed_paper_mlp.sh
done
