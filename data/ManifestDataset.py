"""JSON-manifest loader for comparing BMR on fixed external splits.

This adapter keeps every manifest record in its original split. Per the BMR
paper, unreadable or smaller-than-64 images become zero image inputs and
texts shorter than five words become the placeholder text. For text without
whitespace word boundaries (for example Chinese), tokenizer units are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ManifestDataset(Dataset):
    def __init__(self, manifest_file, image_root, tokenizer, image_size=224):
        self.manifest_file = Path(manifest_file)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.rows = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        if not isinstance(self.rows, list):
            raise ValueError(f"Expected a JSON list: {self.manifest_file}")
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def _image(self, relative_path):
        path = self.image_root / str(relative_path)
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                if width < 64 or height < 64:
                    return torch.zeros((3, self.image_size, self.image_size)), "placeholder_small"
                return self.transform(image), "ok"
        except Exception:
            return torch.zeros((3, self.image_size, self.image_size)), "placeholder_unreadable"

    def _short_text(self, text):
        words = text.split()
        if text.isascii() or len(words) > 1:
            return len(words) < 5
        tokens = self.tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=5
        )["input_ids"]
        return len(tokens) < 5

    def __getitem__(self, index):
        row = self.rows[index]
        text = str(row.get("text") or f"Title: {row.get('title', '')}\nBody: {row.get('body', '')}").strip()
        text_state = "ok"
        if self._short_text(text):
            text, text_state = "No text provided", "placeholder_short"
        image, image_state = self._image(row.get("image_path", ""))
        return {
            "text": text,
            "image": image,
            "label": int(row["label"]),
            "id": str(row.get("id", index)),
            "image_state": image_state,
            "text_state": text_state,
        }
