# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : threshold_test.py
# Project     : Medical-Segmentation-IA-Leakage-Audit
# Author      : Nicolas Dumetz
#
# Created     : Friday April 24 2026
#
# =============================================================

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .indepedant import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEVICE,
    IMG_SIZE,
    NUM_CLASSES,
    SimpleDataset,
    USE_CUDA,
    build_model,
)

"""
Launch script example:
python3 src/robust_method/threshold_test.py \
  --model-path best_model.pth \
  --dataset-root /tmp/Dataset_Comparison/frame_mix
"""


# **************************************************************************** #
#                                                                              #
#                                 SETTINGS                                     #
#                                                                              #
# **************************************************************************** #


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset_Lesion_Comparison"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "threshold_test_results.json"
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

IMAGE_DIRNAME = "images"
MASK_DIRNAME = "masks"
TEST_DIRNAME = "test"


# **************************************************************************** #
#                                                                              #
#                              THRESHOLD HELPERS                               #
#                                                                              #
# **************************************************************************** #


def parse_thresholds(raw_value):
    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid threshold `{value}`. Expected a value between 0 and 1.")
        values.append(value)
    if not values:
        raise ValueError("No thresholds provided.")
    return sorted(set(values))


def resolve_test_dir(dataset_root):
    dataset_root = Path(dataset_root)
    candidate = dataset_root / TEST_DIRNAME
    if (candidate / IMAGE_DIRNAME).is_dir() and (candidate / MASK_DIRNAME).is_dir():
        return candidate
    if (dataset_root / IMAGE_DIRNAME).is_dir() and (dataset_root / MASK_DIRNAME).is_dir():
        return dataset_root
    raise FileNotFoundError(
        f"Unable to find a test split in `{dataset_root}`. "
        f"Expected either `{candidate / IMAGE_DIRNAME}` or `{dataset_root / IMAGE_DIRNAME}`."
    )


# **************************************************************************** #
#                                                                              #
#                              DATASET DEFINITION                              #
#                                                                              #
# **************************************************************************** #


class TestDataset(SimpleDataset):
    def __init__(self, test_root, img_size=IMG_SIZE):
        self.root = Path(test_root)
        super().__init__(
            img_dir=self.root / IMAGE_DIRNAME,
            mask_dir=self.root / MASK_DIRNAME,
            img_size=img_size,
            transform=None,
        )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        stem = Path(self.filenames[idx]).stem
        return sample["image"], sample["label"], stem


def positive_probabilities_from_logits(logits):
    if logits.shape[1] == 1:
        return torch.sigmoid(logits).squeeze(1)
    return torch.softmax(logits, dim=1)[:, 1]


def compute_metrics(pred_bin, gt_bin):
    tp = torch.sum(pred_bin * gt_bin).item()
    fp = torch.sum(pred_bin * (1.0 - gt_bin)).item()
    fn = torch.sum((1.0 - pred_bin) * gt_bin).item()

    union = torch.sum(pred_bin).item() + torch.sum(gt_bin).item()
    intersection = tp

    dice = (2.0 * intersection) / union if union > 0 else 1.0
    iou = intersection / (union - intersection) if (union - intersection) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return dice, iou, precision, recall


# **************************************************************************** #
#                                                                              #
#                                EVALUATION                                    #
#                                                                              #
# **************************************************************************** #


def evaluate_thresholds(model, loader, thresholds):
    model.eval()
    results = {
        threshold: {"dice": [], "iou": [], "precision": [], "recall": []}
        for threshold in thresholds
    }

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Inference test"):
            images = images.to(DEVICE, non_blocking=USE_CUDA)
            labels = labels.to(DEVICE, non_blocking=USE_CUDA)
            probabilities = positive_probabilities_from_logits(model(images))
            target = (labels == 1).float()

            for threshold in thresholds:
                predictions = (probabilities > threshold).float()
                for pred_bin, gt_bin in zip(predictions, target):
                    dice, iou, precision, recall = compute_metrics(pred_bin, gt_bin)
                    results[threshold]["dice"].append(dice)
                    results[threshold]["iou"].append(iou)
                    results[threshold]["precision"].append(precision)
                    results[threshold]["recall"].append(recall)

    return results


def summarize_results(results):
    summary = {}
    best_threshold = None
    best_dice = -1.0

    for threshold, metrics in results.items():
        entry = {
            "dice": float(np.mean(metrics["dice"])) if metrics["dice"] else 0.0,
            "iou": float(np.mean(metrics["iou"])) if metrics["iou"] else 0.0,
            "precision": float(np.mean(metrics["precision"])) if metrics["precision"] else 0.0,
            "recall": float(np.mean(metrics["recall"])) if metrics["recall"] else 0.0,
            "images": len(metrics["dice"]),
        }
        summary[threshold] = entry

        if entry["dice"] > best_dice:
            best_dice = entry["dice"]
            best_threshold = threshold

    return summary, best_threshold


# **************************************************************************** #
#                                                                              #
#                                 REPORTING                                    #
#                                                                              #
# **************************************************************************** #


def print_report(summary, best_threshold):
    print("\n" + "=" * 65)
    print("RESULTATS PAR SEUIL SUR LE SET DE TEST")
    print("=" * 65)
    print(f"{'Seuil':<8} | {'Dice':<8} | {'IoU':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 65)

    for threshold in sorted(summary):
        metrics = summary[threshold]
        print(
            f"{threshold:<8.1f} | "
            f"{metrics['dice'] * 100:<8.2f} | "
            f"{metrics['iou'] * 100:<8.2f} | "
            f"{metrics['precision'] * 100:<10.2f} | "
            f"{metrics['recall'] * 100:<8.2f}"
        )

    print("=" * 65)
    print(
        f"FINAL VERDICT: threshold {best_threshold:.1f} gives "
        f"{summary[best_threshold]['dice'] * 100:.2f}% Dice."
    )
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained paper TransUNet checkpoint on the test split for multiple thresholds."
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint path (.pth)")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated thresholds, e.g. 0.1,0.2,0.3,0.4,0.5",
    )
    parser.add_argument("--results-json", default=str(DEFAULT_RESULTS_JSON))
    return parser.parse_args()


# **************************************************************************** #
#                                                                              #
#                                  MAIN LOOP                                   #
#                                                                              #
# **************************************************************************** #


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    test_root = resolve_test_dir(args.dataset_root)
    dataset = TestDataset(test_root, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=USE_CUDA,
    )

    print(f"Device: {DEVICE}")
    print(f"Test dataset: {test_root}")
    print(f"Images: {len(dataset)}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Thresholds: {', '.join(f'{value:.1f}' for value in thresholds)}")

    model, model_num_classes = build_model(
        args.model_path,
        img_size=args.img_size,
        fallback_num_classes=args.num_classes,
    )
    print(f"Model classes: {model_num_classes}")

    raw_results = evaluate_thresholds(model, loader, thresholds)
    summary, best_threshold = summarize_results(raw_results)

    print_report(summary, best_threshold)

    payload = {
        "config": {
            "device": str(DEVICE),
            "model_path": str(Path(args.model_path).resolve()),
            "dataset_root": str(test_root.resolve()),
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_classes": model_num_classes,
            "thresholds": thresholds,
        },
        "best_threshold": best_threshold,
        "best_dice": summary[best_threshold]["dice"],
        "threshold_results": {f"{threshold:.1f}": metrics for threshold, metrics in summary.items()},
    }

    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
