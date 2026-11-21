"""
Final confirmation run for ViT linear probing on EuroSAT.

Combines all seven confirmed hyperparameter winners in a single run --
no sweep here has ever actually trained with all of them together at
once, since each sweep held the others at whatever was the interim
default at the time.

Confirmed configuration:
  model_name              vit_large_patch16_224   (val_acc 0.9696)
  lr                       0.03                    (val_acc 0.9649)
  target_effective_batch   32                      (val_acc 0.9716)
  weight_decay             0.05                    (val_acc 0.9699)
  epochs                   20                      (val_acc 0.9704)
  train/val ratio          0.80 / 0.10              (val_acc 0.9741)
  img_size                 224                      (val_acc 0.9741)

IMPORTANT: unlike every sweep script, this one also evaluates on the
TEST split -- the one part of the data that was never touched by any
of the seven tuning decisions above. val_acc from the sweeps is a bit
optimistic (7 choices were each picked because they scored well on
val), so test_acc here is the number that belongs in a final report.
"""

import copy
import gc
import json
import subprocess
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
# Final confirmed configuration -- every value below is a winner from
# its own sweep, now combined for the first time.
# ----------------------------
SEED = 42
MODEL_NAME = "vit_large_patch16_224"
NUM_CLASSES = 10
PRETRAINED = True
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
EPOCHS = 20
LR = 3e-2
WEIGHT_DECAY = 5e-2
TARGET_EFFECTIVE_BATCH = 32
IMG_SIZE = 224
NUM_WORKERS = 6  # single run, one DataLoader build -- full parallelism is safe
PREFETCH_FACTOR = 4
DATA_ROOT = "./data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_gpu_stats(tag=""):
    """Log GPU temp/clock/power via nvidia-smi to catch thermal throttling."""
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
    """Grows through candidate batch sizes until OOM, then backs off by safety_margin."""
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
            break
    if largest_working is None:
        return None
    return max(1, int(largest_working * safety_margin))


def build_model():
    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
    # Linear probing: freeze everything, then explicitly re-enable only the
    # classification head (name-substring matching like "fc" in name is
    # unsafe -- ViT's MLP blocks are named blocks.N.mlp.fc1/fc2).
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
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    print(f"[DATA] total={n} train={n_train} val={n_val} test={n_test}")

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        )

    return make_loader(train_set, True), make_loader(val_set, False), make_loader(test_set, False)


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


def main():
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"[GPU] {torch.cuda.get_device_name(0)} -- free: {free_bytes/1e9:.2f} GB / total: {total_bytes/1e9:.2f} GB")

    set_seed(SEED)
    model = build_model()
    criterion = nn.CrossEntropyLoss()

    vram_max_bs = probe_batch_size(model, criterion, IMG_SIZE)
    if vram_max_bs is None:
        raise RuntimeError("VRAM probe failed even at the smallest candidate batch size -- check GPU availability/memory.")
    physical_bs = min(vram_max_bs, TARGET_EFFECTIVE_BATCH)
    accum_steps = max(1, round(TARGET_EFFECTIVE_BATCH / physical_bs))
    print(f"[probe] vram_max_bs={vram_max_bs}, physical_bs={physical_bs}, accum_steps={accum_steps}")

    # Re-initialize the model fresh so the probe's warmup steps on random
    # noise don't leak into the real training run.
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass
    set_seed(SEED)
    model = build_model()

    train_loader, val_loader, test_loader = build_dataloaders(physical_bs)

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    history = []

    t_start = time.time()
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
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(
            f"[final] epoch {epoch+1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        log_gpu_stats(f"epoch {epoch+1}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t_start

    # Load the best-val checkpoint before the final, one-time test evaluation.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print("\n=== Final Result ===")
    print(f"Best val_acc:  {best_val_acc:.4f}  (epoch {best_epoch}/{EPOCHS})")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}   <-- unbiased final number, never used in any tuning decision")
    print(f"Elapsed:       {elapsed:.1f}s")

    result = {
        "config": {
            "model_name": MODEL_NAME, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "target_effective_batch": TARGET_EFFECTIVE_BATCH, "img_size": IMG_SIZE,
            "epochs": EPOCHS, "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO,
        },
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "physical_bs": physical_bs,
        "accum_steps": accum_steps,
        "elapsed_sec": elapsed,
    }
    with open("final_run_results.json", "w") as f:
        json.dump(result, f, indent=2)

    if best_state is not None:
        torch.save(best_state, "final_model_head.pth")

    del train_loader, val_loader, test_loader
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


if __name__ == "__main__":
    main()