#!/usr/bin/env python3
"""Check BMR assets needed before a paper-protocol training launch."""

import argparse
import os
from pathlib import Path


def require(path: Path, description: str, problems: list[str]) -> None:
    marker = "OK" if path.exists() else "MISSING"
    print(f"[{marker}] {description}: {path}")
    if marker == "MISSING":
        problems.append(description)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["weibo", "Weibo_21", "gossip"], required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    problems: list[str] = []

    require(args.project_root / "mae_pretrain_vit_base.pth", "MAE ViT-base checkpoint", problems)
    chinese = Path(os.environ.get("BMR_BERT_CHINESE", "bert-base-chinese"))
    english = Path(os.environ.get("BMR_BERT_UNCASED", "bert-base-uncased"))
    if args.dataset in {"weibo", "Weibo_21"}:
        require(chinese, "Chinese BERT directory (set BMR_BERT_CHINESE)", problems)
        root = Path(os.environ.get("BMR_DATA_ROOT", "/home/groupshare")) / args.dataset
        suffix = "" if "21" in args.dataset else "_WWW_new"
        require(root / f"train_datasets{suffix}.xlsx", "training workbook", problems)
        require(root / f"test_datasets{suffix}.xlsx", "test workbook", problems)
        require(root / f"{args.dataset}_train_ambiguity_new.xlsx", "cross-modal consistency workbook", problems)
    else:
        require(english, "English BERT directory (set BMR_BERT_UNCASED)", problems)
        root = Path(os.environ.get("BMR_FAKENEWSNET_ROOT", "/home/groupshare/AAAI_dataset"))
        require(root / "gossip" / "gossip_train_no_filt.xlsx", "GossipCop training workbook", problems)
        require(root / "gossip" / "gossip_test_no_filt.xlsx", "GossipCop test workbook", problems)
        require(root / "gossip" / "gossip_train_ambiguity.xlsx", "GossipCop cross-modal consistency workbook", problems)

    if problems:
        raise SystemExit("Missing required BMR assets: " + ", ".join(problems))
    print("Paper-protocol asset check passed. Image paths are validated lazily by the released dataset loader.")


if __name__ == "__main__":
    main()
