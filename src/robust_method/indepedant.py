# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : indepedant.py
# Project     : Medical-Segmentation-IA-Leakage-Audit
# Author      : Nicolas Dumetz
#
# Created     : Friday April 24 2026
#
# =============================================================
import argparse
import cv2
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .model import build_paper_transunet
from .train import IMG_SIZE, NUM_CLASSES, SimpleDataset

"""
Launch script example:
python3 src/robust_method/indepedant.py \
  --model-path best_model.pth \
  --dataset-root /tmp/Dataset_Comparison/frame_mix
"""


# **************************************************************************** #
#                                                                              #
#                                 SETTINGS                                     #
#                                                                              #
# **************************************************************************** #


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset_Lesion_Comparison"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "independent_eval_results.json"
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 2
DEFAULT_THRESHOLD = 0.5

IMAGE_DIRNAME = "images"
MASK_DIRNAME = "masks"
INDEPENDENT_DIRNAME = "independent"
MASK_EXT = ".png"


# **************************************************************************** #
#                                                                              #
#                             DATASET UTILITIES                                #
#                                                                              #
# **************************************************************************** #


def patient_id_from_name(filename):
    stem = Path(filename).stem
    return stem.split("_", 1)[0]


def resolve_independent_dir(dataset_root):
    dataset_root = Path(dataset_root)
    candidate = dataset_root / INDEPENDENT_DIRNAME
    if (candidate / IMAGE_DIRNAME).is_dir() and (candidate / MASK_DIRNAME).is_dir():
        return candidate
    if (dataset_root / IMAGE_DIRNAME).is_dir() and (dataset_root / MASK_DIRNAME).is_dir():
        return dataset_root
    raise FileNotFoundError(
        f"Unable to find an independent dataset in `{dataset_root}`. "
        f"Expected either `{candidate / IMAGE_DIRNAME}` or `{dataset_root / IMAGE_DIRNAME}`."
    )


# **************************************************************************** #
#                                                                              #
#                              DATASET DEFINITION                              #
#                                                                              #
# **************************************************************************** #


class IndependentDataset(SimpleDataset):
    def __init__(self, independent_root, img_size=IMG_SIZE):
        self.root = Path(independent_root)
        super().__init__(
            img_dir=self.root / IMAGE_DIRNAME,
            mask_dir=self.root / MASK_DIRNAME,
            img_size=img_size,
            transform=None,
        )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        filename = self.filenames[idx]
        stem = Path(filename).stem
        patient_id = patient_id_from_name(filename)
        return sample["image"], sample["label"], stem, patient_id


# **************************************************************************** #
#                                                                              #
#                            CHECKPOINT LOADING                                #
#                                                                              #
# **************************************************************************** #


def build_model(model_path, img_size, fallback_num_classes):
    model, num_classes = build_paper_transunet(
        model_path=model_path,
        img_size=img_size,
        device=DEVICE,
        num_classes=fallback_num_classes,
    )
    return model, num_classes


# **************************************************************************** #
#                                                                              #
#                           PREDICTION AND METRICS                             #
#                                                                              #
# **************************************************************************** #


def logits_to_binary_prediction(logits, threshold):
    if logits.shape[1] == 1:
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > threshold).long().squeeze(1)
        positive_probs = probabilities.squeeze(1)
        return prediction, positive_probs

    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    positive_probs = probabilities[:, 1]
    return prediction, positive_probs


def binary_confusion(prediction, target):
    pred_fg = prediction == 1
    target_fg = target == 1
    tp = torch.sum(pred_fg & target_fg).item()
    tn = torch.sum(~pred_fg & ~target_fg).item()
    fp = torch.sum(pred_fg & ~target_fg).item()
    fn = torch.sum(~pred_fg & target_fg).item()
    return tp, tn, fp, fn


def dice_from_masks(prediction, target, smooth=1e-5):
    pred_fg = (prediction == 1).float()
    target_fg = (target == 1).float()
    intersection = torch.sum(pred_fg * target_fg).item()
    denominator = torch.sum(pred_fg).item() + torch.sum(target_fg).item()
    return float((2.0 * intersection + smooth) / (denominator + smooth))


def metrics_from_confusion(tp, tn, fp, fn):
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn + epsilon) if tp + fn > 0 else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn + epsilon) if (tp + fp + fn) > 0 else 1.0
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def save_prediction_mask(destination_dir, stem, probs, threshold):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    mask = (probs > threshold).astype(np.uint8) * 255
    cv2.imwrite(str(destination_dir / f"{stem}{MASK_EXT}"), mask)


# **************************************************************************** #
#                                                                              #
#                                EVALUATION                                    #
#                                                                              #
# **************************************************************************** #


def aggregate_patient_report(patient_stats):
    report = {}
    for patient_id, stats in sorted(patient_stats.items()):
        entry = {
            "images": stats["images"],
            "positive_images": stats["positive_images"],
            "predicted_positive_images": stats["predicted_positive_images"],
            "mean_dice_all": stats["dice_sum"] / max(stats["images"], 1),
            "mean_dice_positive_only": None,
        }
        if stats["positive_images"] > 0:
            entry["mean_dice_positive_only"] = (
                stats["dice_positive_sum"] / stats["positive_images"]
            )
        entry.update(metrics_from_confusion(stats["tp"], stats["tn"], stats["fp"], stats["fn"]))
        report[patient_id] = entry
    return report


def evaluate(model, loader, threshold, save_predictions_dir=None):
    model.eval()
    global_confusion = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    patient_stats = defaultdict(
        lambda: {
            "images": 0,
            "positive_images": 0,
            "predicted_positive_images": 0,
            "dice_sum": 0.0,
            "dice_positive_sum": 0.0,
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }
    )
    summary = {
        "images": 0,
        "positive_images": 0,
        "predicted_positive_images": 0,
        "mean_dice_all": 0.0,
        "mean_dice_positive_only": None,
        "total_predicted_positive_pixels": 0.0,
        "total_target_positive_pixels": 0.0,
    }
    positive_dice_sum = 0.0
    image_results = []

    with torch.no_grad():
        for images, labels, stems, patient_ids in tqdm(loader, desc="Inference independent"):
            images = images.to(DEVICE, non_blocking=USE_CUDA)
            labels = labels.to(DEVICE, non_blocking=USE_CUDA)
            logits = model(images)
            predictions, probabilities = logits_to_binary_prediction(logits, threshold)

            for index, stem in enumerate(stems):
                prediction = predictions[index]
                target = labels[index]
                patient_id = patient_ids[index]

                tp, tn, fp, fn = binary_confusion(prediction, target)
                image_dice = dice_from_masks(prediction, target)
                has_target_positive = bool(torch.any(target == 1).item())
                has_pred_positive = bool(torch.any(prediction == 1).item())

                global_confusion["tp"] += tp
                global_confusion["tn"] += tn
                global_confusion["fp"] += fp
                global_confusion["fn"] += fn

                stats = patient_stats[patient_id]
                stats["images"] += 1
                stats["dice_sum"] += image_dice
                stats["tp"] += tp
                stats["tn"] += tn
                stats["fp"] += fp
                stats["fn"] += fn
                if has_target_positive:
                    stats["positive_images"] += 1
                    stats["dice_positive_sum"] += image_dice
                if has_pred_positive:
                    stats["predicted_positive_images"] += 1

                summary["images"] += 1
                summary["mean_dice_all"] += image_dice
                summary["total_predicted_positive_pixels"] += torch.sum(prediction == 1).item()
                summary["total_target_positive_pixels"] += torch.sum(target == 1).item()
                if has_target_positive:
                    summary["positive_images"] += 1
                    positive_dice_sum += image_dice
                if has_pred_positive:
                    summary["predicted_positive_images"] += 1

                if save_predictions_dir:
                    save_prediction_mask(
                        save_predictions_dir,
                        stem,
                        probabilities[index].detach().cpu().numpy(),
                        threshold,
                    )

                image_results.append(
                    {
                        "stem": stem,
                        "patient_id": patient_id,
                        "dice": image_dice,
                        "target_positive": has_target_positive,
                        "predicted_positive": has_pred_positive,
                        "tp": tp,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                    }
                )

    summary["mean_dice_all"] /= max(summary["images"], 1)
    if summary["positive_images"] > 0:
        summary["mean_dice_positive_only"] = positive_dice_sum / summary["positive_images"]

    return {
        "global": metrics_from_confusion(**global_confusion),
        "summary": summary,
        "patients": aggregate_patient_report(patient_stats),
        "images": image_results,
    }


# **************************************************************************** #
#                                                                              #
#                                 REPORTING                                    #
#                                                                              #
# **************************************************************************** #


def print_report(results):
    global_metrics = results["global"]
    summary = results["summary"]

    print("\n=== Independent Holdout Evaluation ===")
    print(f"Pixel Dice:               {global_metrics['dice'] * 100:.2f}%")
    print(f"Pixel IoU:                {global_metrics['iou'] * 100:.2f}%")
    print(f"Pixel Precision:          {global_metrics['precision'] * 100:.2f}%")
    print(f"Pixel Recall:             {global_metrics['recall'] * 100:.2f}%")
    print(f"Pixel Accuracy:           {global_metrics['accuracy'] * 100:.2f}%")
    print(f"Mean Image Dice:          {summary['mean_dice_all'] * 100:.2f}%")
    if summary["mean_dice_positive_only"] is None:
        print("Mean Dice Positive Only:  n/a")
    else:
        print(f"Mean Dice Positive Only:  {summary['mean_dice_positive_only'] * 100:.2f}%")
    print(f"Predicted Positive Images:{summary['predicted_positive_images']} / {summary['images']}")
    print(f"Target Positive Images:   {summary['positive_images']} / {summary['images']}")
    print(f"Predicted Positive Pixels:{int(summary['total_predicted_positive_pixels'])}")
    print(f"Target Positive Pixels:   {int(summary['total_target_positive_pixels'])}")

    print("\nPer patient:")
    print("Patient     | Images | GT+ Img | Pred+ Img | Mean Dice | Dice GT+")
    print("-" * 72)
    for patient_id, metrics in results["patients"].items():
        dice_pos = "n/a"
        if metrics["mean_dice_positive_only"] is not None:
            dice_pos = f"{metrics['mean_dice_positive_only'] * 100:6.2f}%"
        print(
            f"{patient_id:<11} | "
            f"{metrics['images']:>6} | "
            f"{metrics['positive_images']:>7} | "
            f"{metrics['predicted_positive_images']:>9} | "
            f"{metrics['mean_dice_all'] * 100:>8.2f}% | "
            f"{dice_pos:>8}"
        )


# **************************************************************************** #
#                                                                              #
#                                  ARGUMENTS                                   #
#                                                                              #
# **************************************************************************** #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained paper TransUNet checkpoint on the independent split."
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint path (.pth)")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--results-json", default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument(
        "--save-predictions-dir",
        default="",
        help="Optional directory where predicted binary masks will be written.",
    )
    return parser.parse_args()


# **************************************************************************** #
#                                                                              #
#                                  MAIN LOOP                                   #
#                                                                              #
# **************************************************************************** #


def main():
    args = parse_args()
    independent_root = resolve_independent_dir(args.dataset_root)
    dataset = IndependentDataset(independent_root, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=USE_CUDA,
    )

    print(f"Device: {DEVICE}")
    print(f"Independent dataset: {independent_root}")
    print(f"Images: {len(dataset)}")
    print(f"Checkpoint: {args.model_path}")

    model, model_num_classes = build_model(
        args.model_path,
        img_size=args.img_size,
        fallback_num_classes=args.num_classes,
    )
    print(f"Model classes: {model_num_classes}")

    results = evaluate(
        model,
        loader,
        threshold=args.threshold,
        save_predictions_dir=args.save_predictions_dir or None,
    )
    results["config"] = {
        "device": str(DEVICE),
        "model_path": str(Path(args.model_path).resolve()),
        "dataset_root": str(independent_root.resolve()),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_classes": model_num_classes,
        "threshold": args.threshold,
        "save_predictions_dir": args.save_predictions_dir,
    }

    print_report(results)

    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
