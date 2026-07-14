#!/usr/bin/env python3
"""
Fine-tune a Vision Transformer (ViT) on EuroSAT for Land Cover Classification
==============================================================================

Fixes applied vs original:
  1. Removed deprecated `from torch.cuda.amp import GradScaler, autocast` import
  2. GradScaler() → torch.amp.GradScaler('cuda')
  3. autocast(device_type=...) → torch.amp.autocast('cuda') — compatible with
     both older and newer PyTorch (2.x) on Windows
  4. total_mem → total_memory  (CUDA device properties attribute name)
  5. evaluate() autocast also fixed (was causing crash on val loop)
  6. CPU fallback: autocast disabled gracefully when no CUDA available
  7. Added missing GradScaler / autocast imports guard at top

Runs on: RTX 3050 6 GB (local), Kaggle T4 16 GB, Colab T4 15 GB
"""

import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import timm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

class Config:
    seed: int = 42
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    best_model_path: str = os.path.join(checkpoint_dir, "best_vit_eurosat.pth")

    model_name: str = "vit_base_patch16_224"
    num_classes: int = 10
    pretrained: bool = True

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    epochs: int = 20
    batch_size: int = 16
    accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    num_workers: int = 2

    img_size: int = 224
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)


cfg = Config()

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]


# ============================================================================
# 1. REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 2. AMP HELPERS  ← FIX: unified autocast + scaler that works on any PyTorch 2.x
# ============================================================================

def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    """
    FIX 1 & 2: Use torch.amp.GradScaler (not torch.cuda.amp.GradScaler).
    Disabled automatically when running on CPU so no crash without GPU.
    """
    return torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))


def autocast_ctx(device: torch.device):
    """
    FIX 3: Use torch.amp.autocast with the device string.
    Falls back to a no-op context manager on CPU.
    """
    if device.type == 'cuda':
        return torch.amp.autocast('cuda')
    # On CPU torch.amp.autocast('cpu') exists in PyTorch >= 1.10
    # but fp16 on CPU is unsupported; use bfloat16 or just disable
    return torch.amp.autocast('cpu', enabled=False)


# ============================================================================
# 3. DATA AUGMENTATION
# ============================================================================

def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def get_eval_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(cfg.img_size + 32),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


# ============================================================================
# 4. DATASET
# ============================================================================

def get_dataloaders():
    full_train_ds = torchvision.datasets.EuroSAT(
        root=cfg.data_root, download=True, transform=get_train_transforms()
    )
    full_eval_ds = torchvision.datasets.EuroSAT(
        root=cfg.data_root, download=True, transform=get_eval_transforms()
    )

    n = len(full_train_ds)
    print(f"[DATA] EuroSAT total samples: {n}")

    indices = list(range(n))
    rng = random.Random(cfg.seed)
    rng.shuffle(indices)

    n_train = int(cfg.train_ratio * n)
    n_val   = int(cfg.val_ratio * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[DATA] Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    train_loader = DataLoader(
        Subset(full_train_ds, train_idx),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_eval_ds, val_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(full_eval_ds, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ============================================================================
# 5. MODEL
# ============================================================================

def build_model(device: torch.device) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )

    # Freeze backbone — only the classification head will be trained
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    # Gradient checkpointing saves ~30% VRAM at a small speed cost
    model.set_grad_checkpointing(enable=True)

    model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {cfg.model_name}")
    print(f"[MODEL] Total params:     {total:,}")
    print(f"[MODEL] Trainable params: {trainable:,}  (head only)")
    return model


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    num_batches  = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  train", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # FIX 3: use our helper so autocast works on any PyTorch 2.x build
        with autocast_ctx(device):
            logits = model(images)
            loss   = criterion(logits, labels) / cfg.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * cfg.accum_steps
        num_batches  += 1
        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    # Flush any remaining gradients at end of epoch
    if num_batches % cfg.accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()
    return running_loss / max(num_batches, 1)


# ============================================================================
# 7. VALIDATION / TEST LOOP
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # FIX 5: val loop uses same fixed autocast helper
        with autocast_ctx(device):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(len(loader), 1)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# ============================================================================
# 8. TEST EVALUATION + PLOTS
# ============================================================================

def full_test_evaluation(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    macro_f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Loss:       {test_loss:.4f}")
    print(f"  Accuracy:   {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  Macro F1:   {macro_f1:.4f}")
    print()
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix — Acc: {test_acc*100:.1f}%  F1: {macro_f1:.4f}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = os.path.join(cfg.checkpoint_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    print(f"[EVAL] Confusion matrix saved to {out_path}")
    plt.close(fig)


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    seed_everything(cfg.seed)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", end="")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        # FIX 4: total_mem → total_memory
        print(f"  —  {torch.cuda.get_device_name(0)}  "
              f"({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("  ⚠  No GPU detected; training will be slow.")

    # ---- Data ----
    train_loader, val_loader, test_loader = get_dataloaders()

    # ---- Model ----
    model = build_model(device)

    # ---- Loss / Optimizer / Scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    # FIX 1 & 2: use torch.amp.GradScaler (not deprecated torch.cuda.amp version)
    scaler = make_scaler(device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- Training loop ----
    best_val_acc   = 0.0
    best_model_wts = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg.epochs} epochs")
    print(f"Effective batch size: {cfg.batch_size} × {cfg.accum_steps} = "
          f"{cfg.batch_size * cfg.accum_steps}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  │  "
            f"train_loss: {train_loss:.4f}  │  "
            f"val_loss: {val_loss:.4f}  │  "
            f"val_acc: {val_acc:.4f}  │  "
            f"lr: {lr_now:.2e}  │  "
            f"time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, cfg.best_model_path)
            print(f"  ↑ New best val_acc: {best_val_acc:.4f}  — checkpoint saved")

    print(f"\n{'='*60}")
    print(f"Training complete.  Best val accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")

    # ---- Training curves ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, cfg.epochs + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="green")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    curves_path = os.path.join(cfg.checkpoint_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    print(f"[PLOT] Training curves saved to {curves_path}")
    plt.close(fig)

    # ---- Test evaluation ----
    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(
        torch.load(cfg.best_model_path, map_location=device, weights_only=True)
    )
    full_test_evaluation(model, test_loader, device)


if __name__ == "__main__":
    main()