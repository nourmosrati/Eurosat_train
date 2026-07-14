#!/usr/bin/env python3
"""
ViT-B/16 Fine-tuning on EuroSAT  —  Version 2  (Week 1 Upgrades)
==================================================================

NEW in v2 vs v1:
  [SPEED-1]  RAM Dataset Cache     — loads all 27k images into RAM once,
                                     zero disk reads after startup
  [SPEED-2]  DataLoader workers    — increased 2 → 4, persistent_workers=True
                                     (workers stay alive between epochs, no
                                     thread-spawn overhead each epoch)
  [SPEED-3]  prefetch_factor=2     — each worker pre-loads 2 batches ahead
                                     so GPU never waits for data
  [SPEED-4]  torch.compile()       — compiles model to optimized CUDA kernels
                                     (PyTorch 2.x only, ~10-30% speedup)
  [SPEED-5]  cudnn.benchmark=True  — lets cuDNN auto-tune convolution kernels
                                     for your exact input size
  [QUALITY-1] Progressive unfreezing — 3 training stages:
                Stage 1 (20 ep): head only          lr=3e-4  (restores v1)
                Stage 2 (10 ep): last 4 blocks      lr=1e-5
                Stage 3 (10 ep): last 8 blocks      lr=5e-6
  [QUALITY-2] Stage checkpointing  — each stage saves its own best checkpoint
                                     so you can resume from any point
  [QUALITY-3] Per-stage test eval  — full confusion matrix printed after
                                     every stage, not just at the end

Install (same as before, nothing new needed):
  pip install torch torchvision timm scikit-learn matplotlib tqdm

Runs on: RTX 3050 6 GB (local), Kaggle T4, Colab T4
"""

import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T
import timm
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    seed:           int   = 42
    data_root:      str   = "./data"
    checkpoint_dir: str   = "./checkpoints"

    # ── Model ──────────────────────────────────────────────────────────────
    model_name:   str  = "vit_base_patch16_224"
    num_classes:  int  = 10
    pretrained:   bool = True
    use_compile:  bool = False   # [SPEED-4] torch.compile() — set False if
                                #  you get errors on older Windows builds

    # ── Data splits ────────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio:   float = 0.15

    # ── DataLoader ─────────────────────────────────────────────────────────
    batch_size:       int  = 16   # per micro-step
    accum_steps:      int  = 4    # effective batch = 16 × 4 = 64
    num_workers:      int  = 4    # [SPEED-2] was 2
    prefetch_factor:  int  = 2    # [SPEED-3] batches preloaded per worker
    use_ram_cache:    bool = True  # [SPEED-1] load all images into RAM

    # ── Image ──────────────────────────────────────────────────────────────
    img_size:      int   = 224
    imagenet_mean        = (0.485, 0.456, 0.406)
    imagenet_std         = (0.229, 0.224, 0.225)

    # ── Training stages ────────────────────────────────────────────────────
    # Each stage: (epochs, lr, weight_decay, n_blocks_to_unfreeze)
    # n_blocks=0 → head only   n_blocks=4 → head + last 4 ViT blocks
    stages = [
        {"name": "Stage-1  head only",          "epochs": 20, "lr": 3e-4,  "wd": 1e-2, "unfreeze_blocks": 0},
        {"name": "Stage-2  last 4 blocks",       "epochs": 10, "lr": 1e-5,  "wd": 1e-2, "unfreeze_blocks": 4},
        {"name": "Stage-3  last 8 blocks",       "epochs": 10, "lr": 5e-6,  "wd": 1e-2, "unfreeze_blocks": 8},
    ]

    # ── Resume control ─────────────────────────────────────────────────────
    # Set to 0 to train all stages from scratch.
    # Set to 1 to skip Stage-1 and load its checkpoint (you already did it).
    start_from_stage: int = 0

    # Path to a checkpoint to resume from (only used when start_from_stage > 0)
    resume_checkpoint: str = "./checkpoints/stage1_best.pth"


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
    # [SPEED-5] benchmark=True lets cuDNN pick the fastest conv algorithm
    # for your exact input size — small one-time cost, faster every epoch
    torch.backends.cudnn.benchmark   = True
    torch.backends.cudnn.deterministic = False  # must be False with benchmark


# ============================================================================
# 2. AMP HELPERS
# ============================================================================

def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))


def autocast_ctx(device: torch.device):
    if device.type == 'cuda':
        return torch.amp.autocast('cuda')
    return torch.amp.autocast('cpu', enabled=False)


# ============================================================================
# 3. DATA AUGMENTATION
# ============================================================================

def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
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
# 4. RAM-CACHED DATASET  [SPEED-1]
# ============================================================================

class CachedEuroSAT(Dataset):
    """
    [SPEED-1] Loads ALL 27,000 EuroSAT images into RAM on startup (~200 MB).
    After that, every epoch reads from RAM instead of disk.
    On a laptop with 8+ GB RAM this eliminates disk I/O completely.

    Images are stored as PIL objects (no transform applied yet).
    The transform is applied per __getitem__ so augmentation is still random.
    """
    def __init__(self, root: str, transform=None, download: bool = True):
        base = torchvision.datasets.EuroSAT(
            root=root, download=download, transform=None
        )
        self.transform = transform
        self.classes   = base.classes
        self.images: list = []
        self.labels:  list = []

        print(f"[CACHE] Loading {len(base)} images into RAM ...", end="", flush=True)
        t0 = time.time()
        for img, label in base:
            # img is a PIL Image (EuroSAT uses default_loader)
            self.images.append(img.copy())   # .copy() detaches from file handle
            self.labels.append(label)
        elapsed = time.time() - t0
        ram_mb  = sum(
            i.size[0] * i.size[1] * len(i.getbands())
            for i in self.images[:100]
        ) * len(self.images) / 100 / 1024 / 1024
        print(f" done in {elapsed:.1f}s  (~{ram_mb:.0f} MB in RAM)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# 5. DATALOADERS
# ============================================================================

def get_dataloaders():
    """
    Build train / val / test DataLoaders.
    Uses CachedEuroSAT when cfg.use_ram_cache=True (recommended).
    """
    DatasetClass = CachedEuroSAT if cfg.use_ram_cache else torchvision.datasets.EuroSAT

    # We need the full dataset twice: once with train augmentation, once eval
    full_train_ds = DatasetClass(
        root=cfg.data_root, download=True, transform=get_train_transforms()
    )
    full_eval_ds = DatasetClass(
        root=cfg.data_root, download=False, transform=get_eval_transforms()
    )

    n = len(full_train_ds)
    print(f"[DATA] Total samples: {n}")

    indices  = list(range(n))
    rng      = random.Random(cfg.seed)
    rng.shuffle(indices)

    n_train  = int(cfg.train_ratio * n)
    n_val    = int(cfg.val_ratio   * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[DATA] Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # [SPEED-2] persistent_workers=True — worker processes stay alive between
    # epochs so Python doesn't pay thread-spawn cost at the start of each epoch
    # [SPEED-3] prefetch_factor=2 — each worker prefetches 2 batches ahead
    # so the GPU never waits on the CPU to finish augmentation
    loader_kwargs = dict(
        num_workers      = cfg.num_workers,
        pin_memory       = True,          # pinned RAM → faster CPU→GPU transfer
        persistent_workers = cfg.num_workers > 0,   # [SPEED-2]
        prefetch_factor  = cfg.prefetch_factor if cfg.num_workers > 0 else None,  # [SPEED-3]
    )

    train_loader = DataLoader(
        Subset(full_train_ds, train_idx),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(full_eval_ds, val_idx),
        batch_size=cfg.batch_size * 2, shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        Subset(full_eval_ds, test_idx),
        batch_size=cfg.batch_size * 2, shuffle=False,
        **loader_kwargs
    )
    return train_loader, val_loader, test_loader


# ============================================================================
# 6. MODEL + PROGRESSIVE UNFREEZING  [QUALITY-1]
# ============================================================================

def build_model(device: torch.device) -> nn.Module:
    """Load ViT-B/16 with ImageNet weights, freeze everything to start."""
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )
    # Start fully frozen — unfreezing is done per stage
    for param in model.parameters():
        param.requires_grad = False

    # Head always trains
    for param in model.head.parameters():
        param.requires_grad = True

    model.set_grad_checkpointing(enable=True)
    model.to(device)
    return model


def unfreeze_blocks(model: nn.Module, n_blocks: int) -> None:
    """
    [QUALITY-1] Progressive unfreezing:
    Unfreeze the last `n_blocks` transformer blocks + the head.
    n_blocks=0 → head only (Stage 1, same as v1)
    n_blocks=4 → head + blocks 8,9,10,11
    n_blocks=8 → head + blocks 4,5,6,7,8,9,10,11
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze head
    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze last n_blocks transformer blocks
    if n_blocks > 0:
        total = len(model.blocks)       # 12 for ViT-B/16
        start = max(0, total - n_blocks)
        for i, block in enumerate(model.blocks):
            if i >= start:
                for param in block.parameters():
                    param.requires_grad = True

        # Also unfreeze the final layer norm (sits after the last block)
        if hasattr(model, 'norm'):
            for param in model.norm.parameters():
                param.requires_grad = True

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct         = 100 * trainable_p / total_p
    print(f"[UNFREEZE] Trainable: {trainable_p:,} / {total_p:,}  ({pct:.3f}%)")


# ============================================================================
# 7. TRAINING LOOP
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
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

    # Flush remaining gradients
    if (num_batches % cfg.accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(num_batches, 1)


# ============================================================================
# 8. EVALUATION LOOP
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
# 9. TEST EVALUATION + PLOTS  [QUALITY-3]
# ============================================================================

def full_test_evaluation(model, test_loader, device, stage_name: str, save_prefix: str):
    """Run full test evaluation and save confusion matrix + classification report."""
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    macro_f1 = f1_score(labels, preds, average="macro")
    cm        = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS  —  {stage_name}")
    print(f"{'='*60}")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print()
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="True",
        title=f"{stage_name}  —  Acc: {test_acc*100:.2f}%  F1: {macro_f1:.4f}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = os.path.join(cfg.checkpoint_dir, f"{save_prefix}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confusion matrix → {cm_path}")

    return test_acc, macro_f1


def plot_history(history: dict, stage_name: str, save_prefix: str):
    """Plot and save loss + accuracy curves for one stage."""
    n = len(history["train_loss"])
    if n == 0:
        return
    epochs_range = range(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title=f"{stage_name} — Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["val_acc"], color="green", label="Val Accuracy")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title=f"{stage_name} — Val Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(cfg.checkpoint_dir, f"{save_prefix}_training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Training curves → {out}")


# ============================================================================
# 10. ONE FULL STAGE
# ============================================================================

def run_stage(
    stage_cfg: dict,
    stage_idx: int,
    model: nn.Module,
    train_loader, val_loader, test_loader,
    device: torch.device,
) -> float:
    """
    Run one training stage:
      - unfreeze the right blocks
      - build a fresh optimizer + scheduler for this stage's lr
      - train for stage_cfg["epochs"] epochs
      - evaluate on test set at the end
      - return best val_acc achieved
    """
    stage_name   = stage_cfg["name"]
    n_epochs     = stage_cfg["epochs"]
    lr           = stage_cfg["lr"]
    wd           = stage_cfg["wd"]
    n_blocks     = stage_cfg["unfreeze_blocks"]
    save_prefix  = f"stage{stage_idx+1}"
    ckpt_path    = os.path.join(cfg.checkpoint_dir, f"{save_prefix}_best.pth")

    print(f"\n{'#'*60}")
    print(f"  {stage_name.upper()}")
    print(f"{'#'*60}")

    # [QUALITY-1] Set which layers are trainable for this stage
    unfreeze_blocks(model, n_blocks)

    # Fresh optimizer for new trainable params + new lr
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd,
    )

    # Cosine annealing over this stage's epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = make_scaler(device)

    best_val_acc   = 0.0
    best_wts       = None
    history        = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n  Epochs: {n_epochs}  |  LR: {lr:.1e}  |  "
          f"Unfreeze blocks: {n_blocks}  |  Eff. batch: {cfg.batch_size * cfg.accum_steps}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss            = train_one_epoch(model, train_loader, criterion,
                                                optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"  [{stage_name}] Ep {epoch:02d}/{n_epochs}  │  "
            f"train: {train_loss:.4f}  │  val: {val_loss:.4f}  │  "
            f"acc: {val_acc:.4f}  │  lr: {lr_now:.2e}  │  {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts     = copy.deepcopy(model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"    ↑ New best: {best_val_acc:.4f}  → {ckpt_path}")

    # Load best weights before test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # Save plots
    plot_history(history, stage_name, save_prefix)

    # [QUALITY-3] Full test evaluation after every stage
    full_test_evaluation(model, test_loader, device, stage_name, save_prefix)

    return best_val_acc


# ============================================================================
# 11. MAIN
# ============================================================================

def main():
    seed_everything(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}", end="")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  —  {torch.cuda.get_device_name(0)}  ({props.total_memory/1024**3:.1f} GB)")
    else:
        print("  ⚠  No GPU detected — training will be very slow.")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(device)

    # [SPEED-4] torch.compile() — compiles model to optimized CUDA kernels
    # Gives ~10-30% speedup after the first epoch (first epoch is slow due to compilation)
    # If this crashes (rare on some Windows builds), set cfg.use_compile = False
    if cfg.use_compile and device.type == "cuda":
        print("[COMPILE] Running torch.compile() — first epoch will be slow, rest faster ...")
        try:
            model = torch.compile(model)
            print("[COMPILE] ✓ Model compiled successfully")
        except Exception as e:
            print(f"[COMPILE] ✗ torch.compile() failed ({e}) — continuing without it")

    # ── Resume support ──────────────────────────────────────────────────────
    if cfg.start_from_stage > 0:
        ckpt = cfg.resume_checkpoint
        if os.path.exists(ckpt):
            print(f"\n[RESUME] Loading checkpoint: {ckpt}")
            state = torch.load(ckpt, map_location=device, weights_only=True)
            # torch.compile wraps the model — handle both cases
            try:
                model.load_state_dict(state)
            except Exception:
                model._orig_mod.load_state_dict(state)
            print(f"[RESUME] Starting from Stage {cfg.start_from_stage + 1}")
        else:
            print(f"[RESUME] ⚠ Checkpoint not found: {ckpt}  — starting from scratch")

    # ── Run all stages ──────────────────────────────────────────────────────
    all_results = []
    for i, stage in enumerate(cfg.stages):
        if i < cfg.start_from_stage:
            print(f"\n[SKIP] Stage {i+1} ({stage['name']}) — already done")
            continue
        best_acc = run_stage(stage, i, model, train_loader, val_loader, test_loader, device)
        all_results.append((stage["name"], best_acc))

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    for name, acc in all_results:
        print(f"  {name:<35} best val acc: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*60}\n")
    print("Checkpoints saved in:", cfg.checkpoint_dir)
    print("  stage1_best.pth          — head-only model")
    print("  stage2_best.pth          — last 4 blocks unfrozen")
    print("  stage3_best.pth          — last 8 blocks unfrozen  ← use this one")


if __name__ == "__main__":
    main()
