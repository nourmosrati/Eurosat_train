"""
Test-Time Augmentation (TTA) evaluation for the ViT-B/16 EuroSAT Stage-3
checkpoint — 13-band multispectral version.

WHY THIS VERSION IS DIFFERENT FROM TTA.py:
Your original TTA.py targets the RGB pipeline (train_v2.py): torchvision's
27,000-image EuroSAT, ImageNet normalization, stage{N}_best.pth. This targets
the 13-band pipeline (train_v3_multispectral.py) instead:

  - Dataset: giswqs/EuroSAT_MS via Hugging Face `datasets`, 26,998 images —
    not 27,000. Same 10 classes/label order, different image count, so the
    split indices are NOT the same integers as the RGB run even though the
    seed/ratios/shuffle logic is identical.
  - Checkpoint: checkpoints/ms_stage3_best.pth (modality-prefixed), and the
    model needs its patch_embed conv reshaped to 13 input channels before
    load_state_dict will accept it — a straight timm.create_model() gives a
    3-channel conv and the load fails on a shape mismatch otherwise.
  - Normalization: not ImageNet mean/std. train_v3_multispectral.py computes
    real per-band stats from a sample of its training split and now saves
    them to checkpoints/ms_band_stats.json (added alongside this script) —
    this loads that file directly rather than trying to re-derive the same
    numbers by hand. Re-deriving them here via a second seeded sample would
    work in principle, but any tiny divergence between the two files (order
    of operations, sample size, whatever) would silently produce the wrong
    normalization with no error — reading the saved file removes that risk
    entirely.
  - TTA views: hflip/vflip/rot90/rot270 now operate on (13, 64, 64) tensors
    via transforms.v2 functional ops instead of PIL — PIL can't represent a
    13-band image, so the whole pipeline had to move to tensors.

Your original TTA.py is untouched and still correct for the RGB checkpoint.

Run once train_v3_multispectral.py has produced ms_stage3_best.pth and
checkpoints/ms_band_stats.json (both come from the same run — the stats file
gets written early, well before Stage 1 finishes, the checkpoint at the end):
    python TTA_ms.py
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms.v2 as T2
import torchvision.transforms.v2.functional as TF2
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset          # pip install datasets

# ============================== CONFIG ==============================
# These must match the Config class in train_v3_multispectral.py exactly.
SEED             = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
IMG_SIZE         = 224
NUM_BANDS        = 13
MS_DATASET_ID    = "giswqs/EuroSAT_MS"

# Must match cfg.use_geo_pretrained for whichever checkpoint you're pointing at —
# it controls both the checkpoint filename below and which normalization to use.
USE_GEO_PRETRAINED = True

CHECKPOINT_PATH  = f"checkpoints/ms_{'geo' if USE_GEO_PRETRAINED else 'own'}_stage3_best.pth"
BAND_STATS_PATH  = "checkpoints/ms_band_stats.json"
MODEL_NAME       = "vit_base_patch16_224"
NUM_CLASSES      = 10
BATCH_SIZE       = 32
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Fill this in from your training log's "TEST RESULTS — Stage-3" line (test_acc,
# not val_acc) once you have a completed run, e.g. 98.4. Leave as None to skip
# the sanity-check warning at the bottom.
EXPECTED_STAGE3_TEST_ACC = None
# ======================================================================


def build_model():
    # Same target architecture either way — Conv2d(13,768,16,16) patch_embed —
    # regardless of whether the checkpoint came from the geo-pretrained backbone
    # or this project's own mean-fill adaptation, so this doesn't need to branch.
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    # Reshape the input conv to 13 channels before loading — the checkpoint's
    # patch_embed.proj.weight is (768, 13, 16, 16); a fresh timm model defaults
    # to (768, 3, 16, 16) and load_state_dict would fail on the shape mismatch.
    # Only the SHAPE matters here — whatever values this gets initialized with
    # are immediately overwritten by load_state_dict below.
    old_proj = model.patch_embed.proj
    model.patch_embed.proj = nn.Conv2d(
        NUM_BANDS, old_proj.out_channels, kernel_size=old_proj.kernel_size,
        stride=old_proj.stride, padding=old_proj.padding, bias=(old_proj.bias is not None),
    )

    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_band_stats():
    if USE_GEO_PRETRAINED:
        # Matches SSL4EO-S12's own preprocessing — see train_v3_multispectral.py v3.3.
        return torch.zeros(NUM_BANDS), torch.full((NUM_BANDS,), 10000.0)
    with open(BAND_STATS_PATH) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    return mean, std


def load_ms_dataset(dataset_id: str):
    """Same load + flatten order as CachedEuroSATMS in train_v3_multispectral.py
    (train, then validation, then test, in each split's own row order) — has to
    match exactly, or "index i" here points at a different image than "index i"
    did during training, and the recovered test split would be silently wrong."""
    hf = load_dataset(dataset_id).with_format("numpy")
    images, labels = [], []
    for split_name in hf.keys():
        for row in hf[split_name]:
            images.append(np.asarray(row["image"], dtype=np.uint16))
            labels.append(int(row["label"]))
    return images, labels


def recover_test_indices(n: int):
    """
    Mirrors get_dataloaders_ms() in train_v3_multispectral.py exactly:
        indices = list(range(n)); random.Random(SEED).shuffle(indices)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train+n_val]
        test_idx  = indices[n_train+n_val :]
    """
    indices = list(range(n))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    test_idx = indices[n_train + n_val:]

    print(f"[SPLIT] total={n}  train={n_train}  val={n_val}  test={len(test_idx)}")
    print("        (compare these numbers to your training log's [DATA-MS] line — "
          "they should match exactly)")
    return test_idx


# Named top-level functions, not lambdas or closures — consistent with this
# codebase's rule (see train_v3_multispectral.py's MSTransform): a picklable
# transform costs nothing here since num_workers=0 never triggers spawn, but
# there's no reason to reintroduce the one thing that already broke training.
def view_original(x): return x
def view_hflip(x):    return TF2.hflip(x)
def view_vflip(x):    return TF2.vflip(x)
def view_rot90(x):    return TF2.rotate(x, 90)
def view_rot270(x):   return TF2.rotate(x, 270)

TTA_VIEWS = {
    "original": view_original,
    "hflip":    view_hflip,
    "vflip":    view_vflip,
    "rot90":    view_rot90,
    "rot270":   view_rot270,
}


class MSEvalTransform:
    """Resize -> CenterCrop -> per-band normalize. Matches MSTransform(train=False)
    in train_v3_multispectral.py exactly (same geometry, same normalization)."""
    def __init__(self, band_mean: torch.Tensor, band_std: torch.Tensor):
        self.band_mean = band_mean
        self.band_std = band_std
        self.geo = T2.Compose([
            T2.Resize(IMG_SIZE + 32, antialias=True),
            T2.CenterCrop(IMG_SIZE),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geo(x)
        return (x - self.band_mean[:, None, None]) / self.band_std[:, None, None]


class TTAView(Dataset):
    """Wraps the cached MS images/labels, restricted to test_idx, applying one
    TTA view + the standard eval transform — same order as the original
    TTA.py: view first (on the raw 64x64x13 array), then resize/crop/normalize."""
    def __init__(self, images, labels, indices, view_fn, eval_transform):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.view_fn = view_fn
        self.eval_transform = eval_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = torch.from_numpy(self.images[idx].astype(np.float32))   # (13, 64, 64)
        img = self.view_fn(img)
        img = self.eval_transform(img)
        return img, self.labels[idx]


@torch.no_grad()
def get_probs_and_labels(model, loader):
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def accuracy(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean().item()


def main():
    print(f"Device: {DEVICE}")

    images, labels = load_ms_dataset(MS_DATASET_ID)
    print(f"[DATA] Base dataset size: {len(images)} "
          f"(should be 26998 for EuroSAT_MS — NOT 27000, that's the RGB count)")

    test_idx = recover_test_indices(len(images))
    band_mean, band_std = load_band_stats()
    eval_transform = MSEvalTransform(band_mean, band_std)
    model = build_model()

    view_probs = {}
    labels_ref = None

    for name, view_fn in TTA_VIEWS.items():
        dataset = TTAView(images, labels, test_idx, view_fn, eval_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        probs, labels_t = get_probs_and_labels(model, loader)
        view_probs[name] = probs
        if labels_ref is None:
            labels_ref = labels_t
        else:
            assert torch.equal(labels_ref, labels_t), "Label order mismatch between views!"
        print(f"  {name:>10s} view alone: {accuracy(probs, labels_ref) * 100:.2f}%")

    baseline_acc = accuracy(view_probs["original"], labels_ref)
    stacked = torch.stack(list(view_probs.values()), dim=0)
    avg_probs = stacked.mean(dim=0)
    tta_acc = accuracy(avg_probs, labels_ref)

    print("\n===================== RESULTS =====================")
    print(f"Baseline (no TTA):   {baseline_acc * 100:.2f}%")
    print(f"With TTA (5 views):  {tta_acc * 100:.2f}%")
    print(f"Delta:               {(tta_acc - baseline_acc) * 100:+.2f}%")
    print("=====================================================")

    if EXPECTED_STAGE3_TEST_ACC is not None and abs(baseline_acc * 100 - EXPECTED_STAGE3_TEST_ACC) > 0.5:
        print(f"\n[WARNING] Baseline came out noticeably different from your reported "
              f"{EXPECTED_STAGE3_TEST_ACC:.2f}%. That likely means a config value here "
              "(seed/ratios/dataset) doesn't match what you actually used for training, "
              "or checkpoints/ms_band_stats.json is stale (from an earlier, different "
              "run). Don't trust the TTA delta until this lines up.")


if __name__ == "__main__":
    main()