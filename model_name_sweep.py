"""
model_name sweep for ViT linear probing on EuroSAT.
lr, weight_decay, target_effective_batch, train/val ratio, img_size,
epochs are fixed at your current best-known values (update the
placeholders below as your other sweeps finish).

NOTE: this is the most VRAM-risky sweep after img_size. Larger
backbones (vit_large_*) mean bigger activations even though the
backbone is frozen -- on a 6GB 3050 this can crash outright or force
physical_bs down to 1-2 (unreliable results, flagged below). Runs
that OOM even at batch_size=1 are caught and skipped rather than
killing the whole sweep.
"""

import copy
import gc
import json
import time

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT

# ----------------------------
# Fixed config (unchanged from your table)
# ----------------------------
SEED = 42
NUM_CLASSES = 10
PRETRAINED = True
TRAIN_RATIO = 0.70  # placeholder -- update once your train/val ratio sweep finishes
VAL_RATIO = 0.15  # placeholder -- update once your train/val ratio sweep finishes
EPOCHS = 10  # placeholder -- update once your epochs sweep finishes
LR = 3e-2  # winner from the lr sweep
WEIGHT_DECAY = 1e-2  # placeholder -- update once your weight_decay sweep finishes
TARGET_EFFECTIVE_BATCH = 64  # placeholder -- update once your batch sweep finishes
IMG_SIZE = 224  # placeholder -- update once your img_size sweep finishes
NUM_WORKERS = 6
PREFETCH_FACTOR = 4
DATA_ROOT = "./data"

# ----------------------------
# model_name values to sweep -- must be valid timm IDs, patch size must divide IMG_SIZE
# ----------------------------
MODEL_NAME_VALUES = [
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe_batch_size(model, criterion, img_size, candidates=(8, 16, 24, 32, 48, 64, 96, 128, 192, 256), safety_margin=0.85):
    """Grows through candidate batch sizes until OOM, then backs off by
    safety_margin -- mirrors your original batch-probe log (8..96, 85% margin).
    Returns None if even batch_size=8 OOMs (e.g. vit_large on this GPU)."""
    dummy_head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(dummy_head_params, lr=1e-3)
    largest_working = None
    for bs in candidates:
        try:
            x = torch.randn(bs, 3, img_size, img_size, device=DEVICE)
            y = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            largest_working = bs
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Insufficient VRAM can surface many ways depending on where
            # the allocation failure happens: the clean OutOfMemoryError,
            # a plain "CUDA error: out of memory", or a cuBLAS/cuDNN
            # workspace failure (CUBLAS_STATUS_EXECUTION_FAILED /
            # CUBLAS_STATUS_ALLOC_FAILED / CUDNN_STATUS_ALLOC_FAILED) that
            # never mentions "memory" at all. Treat any of these as "this
            # batch size doesn't fit" rather than trying to whitelist exact
            # wording.
            msg = str(e).lower()
            is_capacity_failure = any(
                token in msg
                for token in (
                    "out of memory",
                    "cuda error",
                    "cublas_status",
                    "cudnn_status",
                )
            )
            if not is_capacity_failure:
                raise
            # The GPU/driver can be left in a shaky state right after this
            # kind of failure, so even cleanup can throw -- don't let that
            # secondary error mask the real result.
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            break
    if largest_working is None:
        return None
    return max(1, int(largest_working * safety_margin))


def build_model(model_name):
    model = timm.create_model(
        model_name, pretrained=PRETRAINED, num_classes=NUM_CLASSES
    )
    # Linear probing: freeze everything, then explicitly re-enable only the
    # classification head. Name-substring matching (e.g. "fc" in name) is
    # unsafe here -- ViT's MLP blocks are named blocks.N.mlp.fc1/fc2, so a
    # naive "fc" in name check also unfreezes every transformer block's
    # MLP, turning this into near-full fine-tuning instead of linear probing.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    return model.to(DEVICE)


def build_dataloaders(batch_size):
    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    full_dataset = EuroSAT(root=DATA_ROOT, download=True, transform=tfm)

    n = len(full_dataset)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, _ = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=False,  # disabled: unreliable across repeated runs on Windows multiprocessing
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,  # disabled: unreliable across repeated runs on Windows multiprocessing
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_one_run(model_name):
    set_seed(SEED)
    model = build_model(model_name)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    physical_bs = probe_batch_size(model, criterion, IMG_SIZE)
    if physical_bs is None:
        print(f"[model={model_name}] SKIPPED: OOM even at batch_size=1 on this GPU")
        return {
            "model_name": model_name,
            "total_params": n_params,
            "trainable_params": n_trainable,
            "skipped": True,
            "best_val_acc": -1.0,
        }, None

    accum_steps = max(1, round(TARGET_EFFECTIVE_BATCH / physical_bs))
    print(
        f"[model={model_name}] total_params={n_params:,} trainable_params={n_trainable:,} "
        f"physical_bs={physical_bs} accum_steps={accum_steps}"
    )
    if physical_bs <= 2:
        print(f"[model={model_name}] WARNING: physical_bs collapsed to {physical_bs} -- VRAM-starved, treat this result with caution")

    train_loader, val_loader = build_dataloaders(physical_bs)

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        running_loss, seen = 0.0, 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps * x.size(0)
            seen += x.size(0)

        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        train_loss = running_loss / seen
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"[model={model_name}] epoch {epoch+1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Explicitly release DataLoaders (and their worker processes) before
    # returning -- on Windows, leftover persistent_workers from this run
    # can race with the next model's freshly spawned workers over the same
    # image files, surfacing as a spurious OSError in the DataLoader.
    del train_loader, val_loader
    gc.collect()
    time.sleep(1)  # give Windows a moment to fully release worker processes

    return {
        "model_name": model_name,
        "total_params": n_params,
        "trainable_params": n_trainable,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "target_effective_batch": TARGET_EFFECTIVE_BATCH,
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "skipped": False,
        "best_val_acc": best_val_acc,
        "history": history,
        "physical_bs": physical_bs,
        "accum_steps": accum_steps,
    }, best_state


def main():
    results = []
    best_overall = {"model_name": None, "best_val_acc": -1.0}
    best_state_overall = None

    for model_name in MODEL_NAME_VALUES:
        t0 = time.time()
        try:
            result, state = train_one_run(model_name)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            is_capacity_failure = any(
                token in msg
                for token in ("out of memory", "cuda error", "cublas_status", "cudnn_status")
            )
            if not is_capacity_failure:
                raise
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            print(f"[model={model_name}] SKIPPED: OOM during training")
            result, state = {"model_name": model_name, "skipped": True, "best_val_acc": -1.0}, None
        result["elapsed_sec"] = time.time() - t0
        results.append(result)

        if not result.get("skipped") and result["best_val_acc"] > best_overall["best_val_acc"]:
            best_overall = result
            best_state_overall = state

        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    with open("model_name_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if best_state_overall is not None:]'and'
    ''
        torch.save(best_state_overall, "best_model_name_sweep_head.pth")

    print("\n=== Summary ===")
    for r in results:
        if r.get("skipped"):
            print(f"model={r['model_name']:<28} SKIPPED (OOM)")
        else:
            print(
                f"model={r['model_name']:<28} best_val_acc={r['best_val_acc']:.4f} "
                f"physical_bs={r['physical_bs']:<4} ({r['elapsed_sec']:.1f}s)"
            )
    print(f"\nBest model: {best_overall['model_name']} (val_acc={best_overall['best_val_acc']:.4f})")


if __name__ == "__main__":
    main()