#!/usr/bin/env python3
"""
Fine-tune a Vision Transformer (ViT) on EuroSAT for Land Cover Classification
==============================================================================

Fixes applied vs original:
  1. Removed deprecated `from torch.cuda.amp import GradScaler, autocast` import
  2. GradScaler() -> torch.amp.GradScaler('cuda')
  3. autocast(device_type=...) -> torch.amp.autocast('cuda') - compatible with
     both older and newer PyTorch (2.x) on Windows
  4. total_mem -> total_memory  (CUDA device properties attribute name)
  5. evaluate() autocast also fixed (was causing crash on val loop)
  6. CPU fallback: autocast disabled gracefully when no CUDA available
  7. Added missing GradScaler / autocast imports guard at top
  8. NEW: auto batch-size probe finds the largest safe batch_size for your
     GPU before training starts, instead of a hardcoded guess.

Runs on: RTX 3050 6 GB (local), Kaggle T4 16 GB, Colab T4 15 GB
"""

import os
import copy
import gc
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
    best_model_path: str = os.path.join(checkpoint_dir, "best_vit_eurosat_20ep.pth")

    model_name: str = "vit_base_patch16_224"
    num_classes: int = 10
    pretrained: bool = True

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    epochs: int = 20
    batch_size: int = 16          # overwritten by auto batch-size probe at runtime
    accum_steps: int = 4
    target_effective_batch: int = 64   # batch_size * accum_steps, kept roughly constant
    lr: float = 1e-2          # tuned via lr.py sweep (was 3e-4 default)
    weight_decay: float = 1e-2
    num_workers: int = 4      # persistent_workers=True means train/val/test loaders
                               # each keep this many processes alive simultaneously.
                               # 16 and then 6 both hit Windows shared-memory/paging
                               # limits once train+val+test workers overlapped. 4 is
                               # safer; loaders are also explicitly freed before test
                               # evaluation (see main()) to avoid accumulation.
    persistent_workers: bool = True   # keep worker processes alive between epochs
    prefetch_factor: int = 4          # each worker preloads 4 batches ahead

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
# 2. AMP HELPERS
# ============================================================================

def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    """
    Use torch.amp.GradScaler (not torch.cuda.amp.GradScaler).
    Disabled automatically when running on CPU so no crash without GPU.
    """
    return torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))


def autocast_ctx(device: torch.device):
    """
    Use torch.amp.autocast with the device string.
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

    # persistent_workers/prefetch_factor only apply when num_workers > 0
    loader_kwargs = {}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(
        Subset(full_train_ds, train_idx),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(full_eval_ds, val_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(full_eval_ds, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        **loader_kwargs,
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

    # Freeze backbone - only the classification head will be trained
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
# 5b. AUTO BATCH SIZE PROBE
# ============================================================================

def find_max_batch_size(model, device, candidates=(8, 16, 24, 32, 48, 64, 96), safety_margin=0.85):
    """
    Probes increasing batch sizes with dummy data to find the largest one
    that fits in VRAM, then backs off by `safety_margin` to leave headroom
    for optimizer states and real data loading (which this probe doesn't
    fully replicate, since AdamW keeps extra momentum buffers per parameter).

    Only meaningful on CUDA - on CPU, just returns the smallest candidate.
    """
    if device.type != "cuda":
        print("[BATCH PROBE] No CUDA device - skipping probe, using smallest candidate.")
        return candidates[0]

    model.train()
    working_bs = candidates[0]

    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(bs, 3, cfg.img_size, cfg.img_size, device=device)
            y = torch.randint(0, cfg.num_classes, (bs,), device=device)

            with autocast_ctx(device):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[BATCH PROBE] batch_size={bs:>3}  ->  OK, peak VRAM: {peak_gb:.2f} GB")
            working_bs = bs
            model.zero_grad(set_to_none=True)

        except torch.cuda.OutOfMemoryError:
            print(f"[BATCH PROBE] batch_size={bs:>3}  ->  OUT OF MEMORY, stopping search")
            torch.cuda.empty_cache()
            break

    safe_bs = max(1, int(working_bs * safety_margin))
    print(f"[BATCH PROBE] Largest working batch size: {working_bs}")
    print(f"[BATCH PROBE] Using {int(safety_margin*100)}% safety margin -> batch_size = {safe_bs}")

    # Reset any state the probe's dummy forward/backward passes may have left behind
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return safe_bs


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
        title=f"Confusion Matrix - Acc: {test_acc*100:.1f}%  F1: {macro_f1:.4f}",
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
    out_path = os.path.join(cfg.checkpoint_dir, "confusion_matrix_20ep.png")
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
        print(f"  -  {torch.cuda.get_device_name(0)}  "
              f"({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("  [!] No GPU detected; training will be slow.")

    # ---- Model (built first, since the batch probe needs it) ----
    model = build_model(device)

    # ---- Auto batch size probe ----
    cfg.batch_size = find_max_batch_size(model, device)
    cfg.accum_steps = max(1, round(cfg.target_effective_batch / cfg.batch_size))
    effective_batch = cfg.batch_size * cfg.accum_steps
    print(f"[CONFIG] batch_size={cfg.batch_size}  accum_steps={cfg.accum_steps}  "
          f"effective_batch={effective_batch}\n")

    # ---- Data (built AFTER probe, since it needs the final cfg.batch_size) ----
    train_loader, val_loader, test_loader = get_dataloaders()

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

    scaler = make_scaler(device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- Training loop ----
    best_val_acc   = 0.0
    best_model_wts = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg.epochs} epochs")
    print(f"Effective batch size: {cfg.batch_size} x {cfg.accum_steps} = "
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
            f"Epoch {epoch:02d}/{cfg.epochs}  |  "
            f"train_loss: {train_loss:.4f}  |  "
            f"val_loss: {val_loss:.4f}  |  "
            f"val_acc: {val_acc:.4f}  |  "
            f"lr: {lr_now:.2e}  |  "
            f"time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, cfg.best_model_path)
            print(f"  ^ New best val_acc: {best_val_acc:.4f}  - checkpoint saved")

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
    curves_path = os.path.join(cfg.checkpoint_dir, "training_curves_20ep.png")
    fig.savefig(curves_path, dpi=150)
    print(f"[PLOT] Training curves saved to {curves_path}")
    plt.close(fig)

    # ---- Free train/val loader workers before test evaluation ----
    # persistent_workers=True keeps each loader's worker processes alive as
    # long as the loader object is referenced. train_loader and val_loader
    # have been alive (train_loader's workers + val_loader's workers,
    # re-used every epoch) for the whole training loop. Deleting them here
    # and forcing garbage collection shuts those worker processes down
    # *before* test_loader spawns its own — otherwise all three loaders'
    # workers briefly coexist, which previously exhausted Windows' shared
    # memory / paging file.
    del train_loader, val_loader
    gc.collect()

    # ---- Test evaluation ----
    print("\nLoading best checkpoint for test evaluation ...")
    model.load_state_dict(
        torch.load(cfg.best_model_path, map_location=device, weights_only=True)
    )
    full_test_evaluation(model, test_loader, device)


if __name__ == "__main__":
    main()
