#!/usr/bin/env python3
"""Summarize independent BMR runs without tuning on the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ["accuracy", "macro_f1", "fake_precision", "fake_recall", "fake_f1", "auc"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="*", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = []
    missing = []
    for seed in args.seeds:
        metrics_path = args.root / f"seed_{seed}" / "test_metrics.json"
        if not metrics_path.exists():
            missing.append({"seed": seed, "expected_metrics": str(metrics_path)})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append({"seed": seed, "metrics_path": str(metrics_path), **metrics})

    summary = {
        "protocol": (
            "Independent seed repetitions. Each run selects its checkpoint using validation Macro-F1 only; "
            "test metrics are aggregated after training. Best test row is descriptive, not a model-selection rule."
        ),
        "root": str(args.root),
        "requested_seeds": args.seeds,
        "completed_runs": len(runs),
        "missing_runs": missing,
        "runs": runs,
    }
    if runs:
        summary["aggregate"] = {}
        for metric in METRICS:
            values = [float(run[metric]) for run in runs if run.get(metric) is not None]
            if values:
                summary["aggregate"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        summary["best_descriptive_by_macro_f1"] = max(runs, key=lambda run: run["macro_f1"])
        summary["best_descriptive_by_accuracy"] = max(runs, key=lambda run: run["accuracy"])

    output = args.output or args.root / "multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
