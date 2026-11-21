"""
img_size sweep for ViT-B/16 linear probing on EuroSAT.
lr, weight_decay, target_effective_batch, train/val ratio are fixed
at your current best-known values (update the placeholders below as
your other sweeps finish).

NOTE: img_size must be a multiple of the patch size (16). Token count
scales as (img_size/16)^2 and attention cost scales O(N^2) with token
count, so this is the most VRAM-risky sweep on a 6GB 3050 -- watch the
batch-probe output, if physical_bs collapses to something tiny (e.g. 1-2)
the run is VRAM-starved and results at that size aren't reliable.
"""

import copy
import gc
import subprocess
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
MODEL_NAME = "vit_large_patch16_224"  # winner from the model_name sweep (val_acc=0.9696)
NUM_CLASSES = 10
PRETRAINED = True
TRAIN_RATIO = 0.80  # winner from the train/val ratio sweep (val_acc=0.9741, new best overall)
VAL_RATIO = 0.10  # winner from the train/val ratio sweep (val_acc=0.9741, new best overall)
EPOCHS = 10  # kept at 10, not the confirmed epochs=20 winner -- at ~15min/epoch (batch=32), 4 img_size candidates x 20 epochs could run ~20h+, worse at 336px which costs more per step. This sweep only needs to RANK candidates; rerun the winning img_size at epochs=20 afterward for the final precise number.
LR = 3e-2  # winner from the lr sweep
WEIGHT_DECAY = 5e-2  # winner from the weight_decay sweep (val_acc=0.9699, tiebreak on lower val_loss + still-improving trend)
TARGET_EFFECTIVE_BATCH = 32  # winner from the target_effective_batch sweep (val_acc=0.9716)
NUM_WORKERS = 2  # img_size changes the transform per candidate, so loaders must be rebuilt each time -- keep worker count low (few candidates x 2 loaders = manageable churn) rather than 0, to avoid the fully-serial CPU bottleneck seen on weight_decay
PREFETCH_FACTOR = 4
DATA_ROOT = "./data"

# ----------------------------
# img_size values to sweep -- must all be multiples of patch size 16
# ----------------------------
IMG_SIZE_VALUES = [112, 160, 224, 336]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_gpu_stats(tag=""):
    """Log GPU temp/clock/power via nvidia-smi to catch thermal throttling
    during long sweeps. A clock well below the max under sustained load is
    the signature of throttling."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            temp, clock_sm, clock_max, power, power_limit = [v.strip() for v in result.stdout.strip().split(",")]
            ratio = float(clock_sm) / float(clock_max) if float(clock_max) > 0 else 1.0
            flag = " <-- POSSIBLE THROTTLING" if ratio < 0.85 else ""
            label = f" {tag}" if tag else ""
            print(f"[GPU{label}] temp={temp}C clock={clock_sm}/{clock_max}MHz ({ratio:.0%}) power={power}/{power_limit}W{flag}")
    except Exception as e:
        print(f"[GPU] stats unavailable: {e}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe_batch_size(model, criterion, img_size, candidates=(8, 16, 24, 32, 48, 64, 96, 128, 192, 256), safety_margin=0.85):
    """Grows through candidate batch sizes until OOM, then backs off by
    safety_margin -- mirrors your original batch-probe log (8..96, 85% margin)."""
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


def build_model(img_size):
    # img_size must be passed to timm so positional embeddings are
    # interpolated correctly for non-native resolutions.
    model = timm.create_model(
        MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES, img_size=img_size
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


def build_dataloaders(batch_size, img_size):
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
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


def train_one_run(img_size):
    set_seed(SEED)
    model = build_model(img_size)
    criterion = nn.CrossEntropyLoss()

    vram_max_bs = probe_batch_size(model, criterion, img_size)
    physical_bs = min(vram_max_bs, TARGET_EFFECTIVE_BATCH)
    accum_steps = max(1, round(TARGET_EFFECTIVE_BATCH / physical_bs))
    print(f"[img_size={img_size}] vram_max_bs={vram_max_bs}, physical_bs={physical_bs}, accum_steps={accum_steps}")
    if physical_bs <= 2:
        print(f"[img_size={img_size}] WARNING: physical_bs collapsed to {physical_bs} -- VRAM-starved, treat this result with caution")

    train_loader, val_loader = build_dataloaders(physical_bs, img_size)

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
            f"[img_size={img_size}] epoch {epoch+1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        log_gpu_stats(f"epoch {epoch+1}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Explicitly release DataLoaders (and their worker processes) before
    # returning -- on Windows, leftover persistent_workers from this run
    # can race with the next run's freshly spawned workers over the same
    # image files, surfacing as a spurious OSError in the DataLoader.
    del train_loader, val_loader
    gc.collect()
    time.sleep(1)  # give Windows a moment to fully release worker processes

    return {
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "target_effective_batch": TARGET_EFFECTIVE_BATCH,
        "img_size": img_size,
        "best_val_acc": best_val_acc,
        "history": history,
        "physical_bs": physical_bs,
        "accum_steps": accum_steps,
    }, best_state


def main():
    results = []
    best_overall = {"img_size": None, "best_val_acc": -1.0}
    best_state_overall = None

    for img_size in IMG_SIZE_VALUES:
        t0 = time.time()
        result, state = train_one_run(img_size)
        result["elapsed_sec"] = time.time() - t0
        results.append(result)

        # Save after every candidate, not just at the end -- a crash on
        # candidate N shouldn't lose the results from candidates 1..N-1.
        with open("img_size_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

        if result["best_val_acc"] > best_overall["best_val_acc"]:
            best_overall = result
            best_state_overall = state

        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    with open("img_size_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if best_state_overall is not None:
        torch.save(best_state_overall, "best_img_size_sweep_head.pth")

    print("\n=== Summary ===")
    for r in results:
        print(
            f"img_size={r['img_size']:<5} physical_bs={r['physical_bs']:<4} "
            f"best_val_acc={r['best_val_acc']:.4f}  ({r['elapsed_sec']:.1f}s)"
        )
    print(f"\nBest img_size: {best_overall['img_size']} (val_acc={best_overall['best_val_acc']:.4f})")


if __name__ == "__main__":
    main()