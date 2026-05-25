"""Train released-paper BMR architecture on fixed JSON manifests.

The author training entry point is tied to Excel workbooks and performs
per-epoch evaluation selection on its configured test file. This entry point
imports the paper architecture, keeps BMR_PAPER_STRICT enabled, selects the
checkpoint on validation Macro-F1 only, and evaluates test exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os.environ["BMR_PAPER_STRICT"] = "1"

from data.ManifestDataset import ManifestDataset
from models.UAMFD_Net import UAMFD_Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--text_model", required=True)
    parser.add_argument("--dataset_key", choices=["weibo", "gossip"], required=True,
                        help="Selects Chinese (weibo) or English (gossip) BERT in the released network.")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cc_weight", type=float, default=4.0)
    parser.add_argument("--coarse_weight", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=197)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_stats_eval", action="store_true",
                        help="Numerical-stability fallback: keep BatchNorm layers in batch-stat mode during no-grad validation/test.")
    parser.add_argument("--diagnose_activations", action="store_true",
                        help="Trace abnormal activations during validation/test while preserving standard model.eval().")
    parser.add_argument("--diagnostic_abs_threshold", type=float, default=1e6,
                        help="Record module outputs whose finite absolute maximum reaches this value.")
    parser.add_argument("--diagnostic_output", type=Path, default=None,
                        help="JSON path for strict evaluation activation diagnostics.")
    parser.add_argument("--real_label", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fake_label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8,
                        help="Validation early stopping assumption; the paper does not report patience.")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ActivationTracer:
    """Record the first scale explosion/non-finite layer during strict evaluation."""

    def __init__(self, output_path, abs_threshold=1e6, max_events=20000):
        self.output_path = Path(output_path)
        self.abs_threshold = float(abs_threshold)
        self.max_events = int(max_events)
        self.events = []
        self.first_nonfinite = None
        self.handles = []
        self.enabled = False
        self.context = {}
        self.order = 0

    def register(self, model):
        for name, module in model.named_modules():
            if name:
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def begin_batch(self, split, batch_index, ids):
        self.context = {"split": split, "batch_index": int(batch_index), "ids": list(ids)[:5]}
        self.order = 0
        self.enabled = True

    def end_batch(self):
        self.enabled = False

    @staticmethod
    def _tensors(value):
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, (tuple, list)):
            tensors = []
            for item in value:
                tensors.extend(ActivationTracer._tensors(item))
            return tensors
        if isinstance(value, dict):
            tensors = []
            for item in value.values():
                tensors.extend(ActivationTracer._tensors(item))
            return tensors
        return []

    @staticmethod
    def _stats(tensors):
        count, nonfinite, absmax = 0, 0, 0.0
        shapes = []
        for tensor in tensors:
            if not (tensor.is_floating_point() or tensor.is_complex()):
                continue
            value = tensor.detach()
            shapes.append(list(value.shape))
            count += value.numel()
            finite = torch.isfinite(value)
            nonfinite += int((~finite).sum().item())
            if finite.any():
                absmax = max(absmax, float(value[finite].abs().max().item()))
        return {"num_values": count, "nonfinite": nonfinite, "absmax_finite": absmax, "shapes": shapes}

    def _hook(self, name):
        def capture(module, _inputs, output):
            if not self.enabled:
                return
            self.order += 1
            stats = self._stats(self._tensors(output))
            if stats["num_values"] == 0:
                return
            abnormal = stats["nonfinite"] > 0 or stats["absmax_finite"] >= self.abs_threshold
            if not abnormal:
                return
            event = {
                **self.context,
                "order": self.order,
                "module": name,
                "module_type": type(module).__name__,
                **stats,
            }
            if len(self.events) < self.max_events:
                self.events.append(event)
            if stats["nonfinite"] > 0 and self.first_nonfinite is None:
                self.first_nonfinite = event
        return capture

    def write(self, reason):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "purpose": "Localize the first abnormal/non-finite activation under standard model.eval().",
            "evaluation_mode": "standard model.eval(); no BatchNorm batch-stat fallback",
            "abs_threshold": self.abs_threshold,
            "reason": reason,
            "first_nonfinite": self.first_nonfinite,
            "abnormal_events": self.events,
            "interpretation": (
                "Forward hooks fire after child module execution, so the first non-finite event "
                "in execution order identifies the earliest recorded module output that became non-finite."
            ),
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
        return {
            "encoded": encoded,
            "images": torch.stack([item["image"] for item in batch]),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
            "ids": [item["id"] for item in batch],
            "image_states": [item["image_state"] for item in batch],
            "text_states": [item["text_state"] for item in batch],
        }
    return collate


def to_device(batch, device):
    return (
        batch["encoded"]["input_ids"].to(device),
        batch["encoded"]["attention_mask"].to(device),
        batch["encoded"]["token_type_ids"].to(device),
        batch["images"].to(device),
        batch["labels"].to(device),
    )


def forward_main(model, values):
    input_ids, attention_mask, token_type_ids, images, labels = values
    category = torch.zeros_like(labels, dtype=torch.long)
    output = model(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        image=images, no_ambiguity=False, category=category, calc_ambiguity=False
    )
    return output, labels


def cross_consistency_loss(model, values, criterion, real_label):
    input_ids, attention_mask, token_type_ids, images, labels = values
    indices = torch.nonzero(labels.long() == real_label, as_tuple=True)[0]
    if indices.numel() < 2:
        return torch.zeros((), device=labels.device)
    real_input = input_ids[indices]
    real_mask = attention_mask[indices]
    real_types = token_type_ids[indices]
    real_images = images[indices]
    aux_input = torch.cat([real_input, real_input], dim=0)
    aux_mask = torch.cat([real_mask, real_mask], dim=0)
    aux_types = torch.cat([real_types, real_types], dim=0)
    aux_images = torch.cat([real_images, real_images.roll(1, dims=0)], dim=0)
    aux_labels = torch.cat([torch.zeros(len(indices)), torch.ones(len(indices))]).to(labels.device)
    aux_logits, *_ = model(
        input_ids=aux_input, attention_mask=aux_mask, token_type_ids=aux_types,
        image=aux_images, no_ambiguity=False, category=torch.zeros_like(aux_labels, dtype=torch.long),
        calc_ambiguity=True
    )
    return criterion(aux_logits.squeeze(1), aux_labels)


def metric_values(labels, scores, threshold, fake_label):
    y_true = np.asarray(labels, dtype=np.int64)
    score_fake = np.asarray(scores, dtype=np.float64)
    pred_fake = (score_fake >= threshold).astype(np.int64)
    y_pred = pred_fake if fake_label == 1 else 1 - pred_fake
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    fake_index = fake_label
    auc_labels = (y_true == fake_label).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fake_precision": float(precision[fake_index]),
        "fake_recall": float(recall[fake_index]),
        "fake_f1": float(f1[fake_index]),
        "auc": float(roc_auc_score(auc_labels, score_fake)) if len(np.unique(auc_labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "class_0_support": int(support[0]),
        "class_1_support": int(support[1]),
        "threshold": threshold,
    }


def set_evaluation_mode(model, batch_stats_eval):
    model.eval()
    if batch_stats_eval:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold, fake_label, batch_stats_eval=False, save_predictions=None,
             tracer=None, trace_split="evaluation"):
    set_evaluation_mode(model, batch_stats_eval)
    labels, scores, rows = [], [], []
    losses, states_image, states_text = [], Counter(), Counter()
    completed = False
    try:
        for batch_index, batch in enumerate(loader):
            if tracer:
                tracer.begin_batch(trace_split, batch_index, batch["ids"])
            try:
                values = to_device(batch, device)
                output, y = forward_main(model, values)
            finally:
                if tracer:
                    tracer.end_batch()
            mix_logits = output[0].squeeze(1)
            if not torch.isfinite(mix_logits).all():
                bad_indexes = torch.nonzero(~torch.isfinite(mix_logits), as_tuple=True)[0].cpu().tolist()
                bad_ids = [batch["ids"][i] for i in bad_indexes]
                mode = "batch-stat fallback" if batch_stats_eval else "released eval mode"
                raise FloatingPointError(
                    f"Non-finite BMR logits during {mode}; example ids={bad_ids[:5]}. "
                    "The released network can overflow on external image distributions. "
                    "For a disclosed numerical-stability fallback, rerun with --batch_stats_eval."
                )
            losses.append(criterion(mix_logits, y).item() * len(y))
            probs = torch.sigmoid(mix_logits).cpu().tolist()
            labels.extend(y.long().cpu().tolist())
            scores.extend(probs)
            states_image.update(batch["image_states"])
            states_text.update(batch["text_states"])
            rows.extend({"id": item_id, "label": int(label), "fake_score": float(score)}
                        for item_id, label, score in zip(batch["ids"], y.cpu().tolist(), probs))
        completed = True
    finally:
        if tracer:
            tracer.write(f"{trace_split}_{'completed' if completed else 'aborted'}")
    result = metric_values(labels, scores, threshold, fake_label)
    result["loss"] = float(sum(losses) / max(1, len(labels)))
    result["image_states"] = dict(states_image)
    result["text_states"] = dict(states_text)
    if save_predictions:
        save_predictions.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    args = parse_args()
    if args.real_label == args.fake_label:
        raise ValueError("real_label and fake_label must differ.")
    if args.diagnose_activations and args.batch_stats_eval:
        raise ValueError("--diagnose_activations is for strict standard model.eval(); do not combine it with --batch_stats_eval.")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("The released BMR network constructs CUDA modules; run this baseline with a CUDA GPU.")
    set_seed(args.seed)
    if args.dataset_key == "weibo":
        os.environ["BMR_BERT_CHINESE"] = args.text_model
    else:
        os.environ["BMR_BERT_UNCASED"] = args.text_model
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    collate = make_collate(tokenizer, args.max_length)
    datasets = {
        split: ManifestDataset(args.manifest_dir / f"{split}.json", args.image_root, tokenizer)
        for split in ["train", "val", "test"]
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, drop_last=False,
                           num_workers=args.num_workers, pin_memory=True, collate_fn=collate),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model = UAMFD_Net(
        dataset=args.dataset_key, text_token_len=args.max_length, image_token_len=197,
        is_use_bce=True, batch_size=args.batch_size, thresh=args.threshold
    ).to(device)
    tracer = None
    if args.diagnose_activations:
        diagnostic_output = args.diagnostic_output or (args.output_dir / "activation_diagnostics.json")
        tracer = ActivationTracer(diagnostic_output, args.diagnostic_abs_threshold)
        tracer.register(model)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    total_steps = max(1, args.epochs * len(loaders["train"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["protocol"] = {
        "selection": "best validation Macro-F1; test evaluated once after model selection",
        "cc_pairs": "generated online from real-label training examples only; aligned=0, rolled mismatch=1",
        "invalid_modalities": "image <64x64 or unreadable -> zero image; <5 tokenizer pieces -> No text provided",
        "input": "title + body only",
        "evaluation_batchnorm": "batch statistics stability fallback" if args.batch_stats_eval else "released running statistics",
        "activation_diagnostics": (
            "forward-hook trace during standard model.eval(); no evaluation behavior change"
            if args.diagnose_activations else "disabled"
        ),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best, stale, history, start = -1.0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "main": 0.0, "coarse": 0.0, "cc": 0.0, "n": 0}
        for batch in loaders["train"]:
            values = to_device(batch, device)
            output, labels = forward_main(model, values)
            mix, image_only, text_only, pattern_only = [part.squeeze(1) for part in output[:4]]
            main_loss = criterion(mix, labels)
            coarse_loss = (
                criterion(image_only, labels) + criterion(text_only, labels) + criterion(pattern_only, labels)
            ) / 3.0
            cc_loss = cross_consistency_loss(model, values, criterion, args.real_label)
            loss = main_loss + args.coarse_weight * coarse_loss + args.cc_weight * cc_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}; rerun with a reduced learning rate.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            n = len(labels)
            totals["loss"] += loss.item() * n
            totals["main"] += main_loss.item() * n
            totals["coarse"] += coarse_loss.item() * n
            totals["cc"] += cc_loss.item() * n
            totals["n"] += n
        val = evaluate(model, loaders["val"], criterion, device, args.threshold, args.fake_label, args.batch_stats_eval,
                       tracer=tracer, trace_split=f"validation_epoch_{epoch}")
        record = {"epoch": epoch, "train": {k: totals[k] / max(1, totals["n"]) for k in ["loss", "main", "coarse", "cc"]},
                  "validation": val}
        history.append(record)
        print(json.dumps(record))
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": val}, checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation macro-F1={best:.6f}.")
                break
    torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"]}, checkpoint_dir / "last.pt")
    best_state = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state["model"])
    test = evaluate(model, loaders["test"], criterion, device, args.threshold, args.fake_label, args.batch_stats_eval,
                    args.output_dir / "test_predictions.json", tracer=tracer, trace_split="test")
    test.update({"checkpoint_epoch": int(best_state["epoch"]), "runtime_seconds": time.time() - start})
    (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"final_test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
