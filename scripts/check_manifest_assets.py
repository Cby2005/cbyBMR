#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--mae_checkpoint", type=Path, default=Path("mae_pretrain_vit_base.pth"))
    args = parser.parse_args()
    missing_images = 0
    total = 0
    too_small = 0
    for split in ["train", "val", "test"]:
        path = args.manifest_dir / f"{split}.json"
        if not path.exists():
            raise SystemExit(f"Missing manifest: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            total += 1
            image_path = args.image_root / str(row.get("image_path", ""))
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    too_small += int(width < 64 or height < 64)
            except Exception:
                missing_images += 1
        print(f"{split}: {len(rows)} records")
    if not args.mae_checkpoint.exists():
        raise SystemExit(f"Missing MAE checkpoint: {args.mae_checkpoint}")
    if missing_images:
        raise SystemExit(f"{missing_images} manifest images are not readable.")
    print(f"Asset check passed: total={total}, image_placeholder_lt64={too_small}, MAE={args.mae_checkpoint}")


if __name__ == "__main__":
    main()
