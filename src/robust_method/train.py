
# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : train.py
# Project     : Medical-Segmentation-IA-Leakage-Audit
# Author      : Nicolas Dumetz
#
# Created     : Friday April 24 2026
#
# =============================================================
import cv2
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from .model import PaperTransUNet, build_paper_config

"""
Launch script example:
DATASET_ROOT=/tmp/Dataset_Comparison/frame_mix python3 src/robust_method/train.py
"""


# **************************************************************************** #
#                                                                              #
#                                 SETTINGS                                     #
#                                                                              #
# **************************************************************************** #


IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SAVE_DIR = "checkpoints"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/tmp/Dataset_Lesion_FrameMix"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(8, os.cpu_count() or 1)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
NUM_CLASSES = 2
SEED = int(os.environ.get("SEED", 42))

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

os.makedirs(SAVE_DIR, exist_ok=True)


# **************************************************************************** #
#                                                                              #
#                             REPRODUCIBILITY                                  #
#                                                                              #
# **************************************************************************** #


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    random.seed(SEED + worker_id)
    np.random.seed(SEED + worker_id)


# **************************************************************************** #
#                                                                              #
#                               AUGMENTATIONS                                  #
#                                                                              #
# **************************************************************************** #


def random_rot_flip(image, mask):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return image, mask


def random_rotate(image, mask, angle_range=(-20, 20)):
    angle = np.random.uniform(*angle_range)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image, mask


class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, image, mask):
        if random.random() > 0.5:
            image, mask = random_rot_flip(image, mask)
        elif random.random() > 0.5:
            image, mask = random_rotate(image, mask)

        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        return image, mask


# **************************************************************************** #
#                                                                              #
#                              DATASET DEFINITION                              #
#                                                                              #
# **************************************************************************** #


class SimpleDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=IMG_SIZE, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.filenames = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png"))
        )
        if not self.filenames:
            raise RuntimeError(f"No images found in: {self.img_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = self.img_dir / name
        mask_name = Path(name).with_suffix(".png").name
        mask_path = self.mask_dir / mask_name

        image = cv2.imread(os.fspath(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.fspath(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.int64)

        image = image.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(mask).long(),
        }


# **************************************************************************** #
#                                                                              #
#                             LOSSES AND METRICS                               #
#                                                                              #
# **************************************************************************** #


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for class_idx in range(self.num_classes):
            tensor_list.append((input_tensor == class_idx).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * target, dims)
        denominator = torch.sum(inputs * inputs, dims) + torch.sum(target * target, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


def foreground_dice(logits, target, smooth=1e-5):
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    pred_fg = (pred == 1).float()
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    denominator = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.mean()


# **************************************************************************** #
#                                                                              #
#                            TRAINING UTILITIES                                #
#                                                                              #
# **************************************************************************** #


def build_dataloader(dataset, shuffle, batch_size):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": USE_CUDA,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(dataset, **loader_kwargs)


def evaluate(model, dataloader):
    model.eval()
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        val_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in val_bar:
            images = batch["image"].to(DEVICE, non_blocking=USE_CUDA)
            labels = batch["label"].to(DEVICE, non_blocking=USE_CUDA)
            outputs = model(images)
            batch_dice = foreground_dice(outputs, labels).item()
            total_dice += batch_dice
            num_batches += 1
            val_bar.set_postfix(dice=f"{batch_dice:.4f}")

    return total_dice / max(num_batches, 1)


# **************************************************************************** #
#                                                                              #
#                                  MAIN LOOP                                   #
#                                                                              #
# **************************************************************************** #


def train():
    set_seed(SEED)

    train_dir = DATASET_ROOT / "train"
    val_dir = DATASET_ROOT / "val"
    if not val_dir.exists():
        val_dir = DATASET_ROOT / "test"

    train_dataset = SimpleDataset(
        train_dir / "images",
        train_dir / "masks",
        img_size=IMG_SIZE,
        transform=RandomGenerator(output_size=(IMG_SIZE, IMG_SIZE)),
    )
    val_dataset = SimpleDataset(
        val_dir / "images",
        val_dir / "masks",
        img_size=IMG_SIZE,
        transform=None,
    )

    n_gpu = max(torch.cuda.device_count(), 1)
    effective_batch_size = BATCH_SIZE * n_gpu
    train_loader = build_dataloader(train_dataset, shuffle=True, batch_size=effective_batch_size)
    val_loader = build_dataloader(val_dataset, shuffle=False, batch_size=effective_batch_size)

    model = PaperTransUNet(build_paper_config(IMG_SIZE, NUM_CLASSES), img_size=IMG_SIZE).to(DEVICE)
    if USE_CUDA and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(NUM_CLASSES)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    max_iterations = EPOCHS * len(train_loader)
    iter_num = 0

    print(
        f"Training on {DEVICE} | train={len(train_dataset)} | val={len(val_dataset)} "
        f"| batch={effective_batch_size} | workers={NUM_WORKERS} | val_split={val_dir.name}"
    )

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", ncols=100)
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_batches = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [train]",
            leave=False,
        )
        for batch in batch_bar:
            images = batch["image"].to(DEVICE, non_blocking=USE_CUDA)
            labels = batch["label"].to(DEVICE, non_blocking=USE_CUDA)

            outputs = model(images)
            loss_ce = ce_loss(outputs, labels)
            loss_dice = dice_loss(outputs, labels, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = LR * math.pow(1.0 - iter_num / max(max_iterations, 1), 0.9)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            batch_dice = foreground_dice(outputs.detach(), labels).item()
            train_loss += loss.item()
            train_dice += batch_dice
            train_batches += 1

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_dice:.4f}",
                lr=f"{lr_:.6f}",
            )

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_dice = train_dice / max(train_batches, 1)
        avg_val_dice = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Dice: {avg_train_dice:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            train_dice=f"{avg_train_dice:.4f}",
            val_dice=f"{avg_val_dice:.4f}",
        )

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)

    final_path = os.path.join(SAVE_DIR, f"epoch_{EPOCHS}.pth")
    torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    train()
