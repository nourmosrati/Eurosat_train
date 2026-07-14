#!/usr/bin/env python3
"""
ViT-B/16 Fine-tuning on EuroSAT  —  Version 3.0 (13-Band Multispectral Support)
=================================================================================
Adds a full 13-band Sentinel-2 (RGB + NIR + red-edge + SWIR + ...) training path
alongside the original RGB-only one, switchable via cfg.modality ("rgb" / "ms").

Carried over from v2.2: unified multi-stage plotting, and a torch.compile
re-allocation fix across progressive-freezing stages. That fix is now also
corrected for checkpoint saving — the old single-level unwrap only stripped one
OptimizedModule layer, which left stage2_best.pth / stage3_best.pth with stray
"_orig_mod." prefixes in their state_dict keys. See the while-loop unwrap in
unfreeze_blocks() and run_stage().

NEW dependency for the multispectral path: pip install datasets
(Hugging Face `datasets`, used to pull giswqs/EuroSAT_MS — the 13-band release
of EuroSAT, same 10 classes / same class order as the RGB version.)

v3.1: the MS transform was a closure (a function returned by a function),
which crashes DataLoader worker startup on Windows — spawn has to pickle the
Dataset to hand it to each worker, and closures aren't picklable. Replaced
with MSTransform, a plain class. Also dropped MS's default worker count to 0:
the data is already fully RAM-cached, so there's no disk I/O left to hide, and
each spawned worker would otherwise pickle its own full copy of that cache.

v3.2: use_compile now auto-disables on Windows (os.name == "nt"). Inductor,
torch.compile's default backend, needs Triton, which pip-installed torch
doesn't bring on Windows the way it does on Linux — the fix is a separate,
version-matched wheel (triton-windows), not something to sort out mid-run.
With compile off, model never gets OptimizedModule-wrapped in the first
place, so the "_orig_mod." checkpoint-prefix issue from v2.2/v3.0 doesn't
come up either — nothing further to do there on this machine.

v3.3: added cfg.use_geo_pretrained (default True). Instead of adapting the
ImageNet-pretrained ViT's patch_embed by hand (3 real channels + 10 mean-
filled), this loads torchgeo's SSL4EO-S12 ViT-B/16 checkpoint — MAE
self-supervised pretraining on real 13-band Sentinel-2 imagery, so all 13
channels start from genuine pretrained weights instead of 10 of them being
an approximation. Same vit_base_patch16_224 architecture, so the rest of
the pipeline (unfreeze_blocks, run_stage, checkpointing) is unchanged.
Normalization switches to /10000 to match how that checkpoint was actually
pretrained, instead of our own computed per-band z-score. New dependency:
pip install torchgeo. Note this is SSL4EO-S12's MAE pretraining, not the
SatMAE paper's own checkpoint — that one needs 96x96 input, patch size 8,
and grouped (not flat 13-channel) input, which this architecture can't use
directly. adapt_patch_embed_to_multispectral() is kept as a fallback
(use_geo_pretrained=False) to compare against.
"""

import os
import copy
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T          # v1 — PIL-based, used by the RGB pipeline
import torchvision.transforms.v2 as T2      # v2 — tensor-based, used by the MS pipeline (arbitrary channel count)
import timm
from torchgeo.models import ViTBase16_Weights, vit_base_patch16_224 as geo_vit_base_patch16_224  # pip install torchgeo
from datasets import load_dataset           # pip install datasets  (Hugging Face)
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
    seed:           int   = 42
    data_root:      str   = "./data"
    checkpoint_dir: str   = "./checkpoints"

    model_name:   str  = "vit_base_patch16_224"
    num_classes:  int  = 10
    pretrained:   bool = True
    use_compile:  bool = (os.name != "nt")
    # torch.compile's default backend (inductor) needs Triton to generate GPU
    # kernels. Triton isn't bundled with pip-installed torch on Windows — it
    # needs a separate community wheel (triton-windows) matched to your exact
    # torch/CUDA build — so this auto-disables on Windows rather than crashing
    # on the first forward pass. On Linux, standard `pip install torch` with
    # CUDA already brings Triton, so this stays on there.

    # ---- Modality switch -----------------------------------------------
    # "rgb": original 3-channel pipeline (torchvision EuroSAT, ImageNet stats)
    # "ms" : full 13-band Sentinel-2 pipeline (Hugging Face giswqs/EuroSAT_MS)
    modality:          str   = "ms"
    num_bands:         int   = 13
    rgb_band_indices:  tuple = (3, 2, 1)            # Red=B04, Green=B03, Blue=B02 → positions in the 13-band stack
    nir_band_index:    int   = 7                    # B08, for reference / any NDVI-style work later
    ms_dataset_id:     str   = "giswqs/EuroSAT_MS"  # Hugging Face dataset id

    # Backbone pretraining source for the MS path:
    #   True  -> torchgeo's SSL4EO-S12 ViT-B/16, MAE-pretrained on real 13-band Sentinel-2
    #            imagery. All 13 patch_embed channels carry genuine pretrained weights —
    #            this is what "NIR + SatMAE" in the roadmap slide actually maps to. (Note:
    #            this is SSL4EO-S12's plain MAE pretraining, not the SatMAE paper's own
    #            temporal/channel-grouped checkpoint — that one needs a 96x96 input, patch
    #            size 8, and 3 channel-groups instead of a flat 13-channel conv, which isn't
    #            compatible with this pipeline's architecture. SSL4EO-S12 gets the same core
    #            benefit — real multispectral pretraining — as a drop-in vit_base_patch16_224.)
    #   False -> this project's own adapt_patch_embed_to_multispectral() surgery: 3 channels
    #            copied from ImageNet RGB weights, the other 10 mean-filled. Kept as the
    #            baseline to compare against.
    use_geo_pretrained: bool = True
    geo_weights:        str  = "SENTINEL2_ALL_MAE"   # or "SENTINEL2_ALL_FGMAE" — pip install torchgeo

    train_ratio: float = 0.70
    val_ratio:   float = 0.15

    batch_size:       int   = 16 
    accum_steps:      int   = 4    # Effective batch = 64
    num_workers:      int   = 4    
    num_workers_ms:   int   = 0    # MS images are already fully RAM-cached, so there's no disk I/O
                                    # for workers to hide — and on Windows (spawn, not fork) each
                                    # worker would pickle its own full copy of the ~2.7GB cache for
                                    # no benefit. Safe to raise now that MSTransform is picklable,
                                    # if profiling shows loading actually bottlenecking the GPU.
    prefetch_factor:  int   = 2    
    use_ram_cache:    bool = True  

    img_size:      int   = 224
    imagenet_mean        = (0.485, 0.456, 0.406)   # RGB path only
    imagenet_std         = (0.229, 0.224, 0.225)   # MS path computes its own per-band stats from the data

    stages = [
        {"name": "Stage-1  head only",      "epochs": 20, "lr": 3e-4,  "wd": 1e-2, "unfreeze_blocks": 0},
        {"name": "Stage-2  last 4 blocks",    "epochs": 10, "lr": 1e-5,  "wd": 1e-2, "unfreeze_blocks": 4},
        {"name": "Stage-3  last 8 blocks",    "epochs": 10, "lr": 5e-6,  "wd": 1e-2, "unfreeze_blocks": 8},
    ]

    # Set to 0 to train all stages from scratch.
    start_from_stage: int = 0
    resume_checkpoint: str = f"./checkpoints/{modality}_{'geo' if use_geo_pretrained else 'own'}_stage1_best.pth"


cfg = Config()

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
# giswqs/EuroSAT_MS uses this exact same label order (0..9), confirmed against its dataset card.


# ============================================================================
# 1. REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark   = True
    torch.backends.cudnn.deterministic = False 


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
# 3a. DATA AUGMENTATION — RGB (unchanged)
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
# 3b. DATA AUGMENTATION — Multispectral (NEW)
# ============================================================================
# PIL can't represent a 13-band image, so this pipeline works on tensors directly
# via transforms.v2 (RandomResizedCrop/flip/rotate are channel-count-agnostic there).
# ColorJitter has no multispectral equivalent — "hue"/"saturation" aren't defined
# outside 3-channel RGB — so it's replaced with a per-band gain jitter, which is
# the standard remote-sensing analogue of a brightness perturbation.
#
# This is a class, not a closure returning a nested function. DataLoader workers
# on Windows are spawned, not forked, which means the Dataset — and therefore
# this transform, stored as its .transform attribute — gets pickled and shipped
# to each worker process. A function defined inside another function has no
# importable name, so pickle can't reconstruct it in the worker (that's the
# `Can't get local object '...<locals>._transform'` error). A plain class with
# tensor attributes pickles fine.

def spectral_gain_jitter(x: torch.Tensor, jitter: float = 0.1) -> torch.Tensor:
    """Multiply each band by an independent random gain in [1-jitter, 1+jitter]."""
    gain = 1.0 + (torch.rand(x.shape[0], 1, 1) * 2 - 1) * jitter
    return x * gain


class MSTransform:
    def __init__(self, band_mean: torch.Tensor, band_std: torch.Tensor, train: bool):
        self.band_mean = band_mean
        self.band_std  = band_std
        self.train     = train
        if train:
            self.geo = T2.Compose([
                T2.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0), antialias=True),
                T2.RandomHorizontalFlip(p=0.5),
                T2.RandomVerticalFlip(p=0.5),
                T2.RandomRotation(degrees=90),
            ])
        else:
            self.geo = T2.Compose([
                T2.Resize(cfg.img_size + 32, antialias=True),
                T2.CenterCrop(cfg.img_size),
            ])

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img_np.astype(np.float32))   # (13, 64, 64)
        x = self.geo(x)
        if self.train:
            x = spectral_gain_jitter(x)
        x = (x - self.band_mean[:, None, None]) / self.band_std[:, None, None]
        return x


# ============================================================================
# 4a. RAM-CACHED DATASET — RGB (unchanged)
# ============================================================================

class CachedEuroSAT(Dataset):
    def __init__(self, root: str, transform=None, download: bool = True):
        base = torchvision.datasets.EuroSAT(root=root, download=download, transform=None)
        self.transform = transform
        self.classes   = base.classes
        self.images: list = []
        self.labels:  list = []

        print(f"[CACHE] Loading {len(base)} images into RAM ...", end="", flush=True)
        t0 = time.time()
        for img, label in base:
            self.images.append(img.copy())   
            self.labels.append(label)
        elapsed = time.time() - t0
        ram_mb  = sum(i.size[0] * i.size[1] * len(i.getbands()) for i in self.images[:100]) * len(self.images) / 100 / 1024 / 1024
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
# 4b. RAM-CACHED DATASET — Multispectral (NEW)
# ============================================================================

class CachedEuroSATMS(Dataset):
    """
    Loads the 13-band EuroSAT_MS dataset (Hugging Face) into RAM once, as raw
    uint16 arrays — no transform baked in here. Wrap with _TransformView to get
    a train/eval split with its own transform, without re-downloading or
    re-caching the ~2.25 GB of pixel data twice (unlike the RGB path above,
    which does cache train/eval separately — fine there since RGB is much
    smaller, but wasteful at this size).
    """
    def __init__(self, dataset_id: str):
        print(f"[CACHE-MS] Loading {dataset_id} (downloads on first run, ~2.25 GB) ...", end="", flush=True)
        t0 = time.time()
        hf = load_dataset(dataset_id).with_format("numpy")
        self.images: list = []
        self.labels: list = []
        for split_name in hf.keys():                      # combine train+validation+test
            for row in hf[split_name]:
                self.images.append(np.asarray(row["image"], dtype=np.uint16))  # (13, 64, 64)
                self.labels.append(int(row["label"]))
        elapsed = time.time() - t0
        ram_mb = sum(img.nbytes for img in self.images) / 1024 / 1024
        print(f" done in {elapsed:.1f}s  ({len(self.images)} images, ~{ram_mb:.0f} MB in RAM)")

    def __len__(self) -> int:
        return len(self.images)

    def raw(self, idx: int):
        return self.images[idx], self.labels[idx]


class _TransformView(Dataset):
    """Applies a given transform to a shared, already-cached CachedEuroSATMS instance."""
    def __init__(self, base: CachedEuroSATMS, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base.raw(idx)
        return self.transform(img), label


def compute_band_stats(full_ds: CachedEuroSATMS, train_idx: list):
    """
    Per-band mean/std computed over the TRAINING split only (no val/test leakage).
    ImageNet's 3-channel stats don't apply once every Sentinel-2 band is in play,
    so this replaces cfg.imagenet_mean/std for the MS pipeline. Sampled rather
    than run over all ~18,900 training images — 4000 is already ~16M pixels per
    band, plenty for a stable estimate, and keeps this from adding noticeable
    startup time.
    """
    sample_n = min(len(train_idx), 4000)
    sample_idx = random.Random(cfg.seed).sample(train_idx, sample_n)
    stacked = np.stack([full_ds.raw(i)[0] for i in sample_idx]).astype(np.float64)  # (N, 13, 64, 64)
    mean = stacked.mean(axis=(0, 2, 3))
    std  = stacked.std(axis=(0, 2, 3)) + 1e-6
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


# ============================================================================
# 5. DATALOADERS
# ============================================================================

def get_dataloaders():
    if cfg.modality == "rgb":
        return get_dataloaders_rgb()
    elif cfg.modality == "ms":
        return get_dataloaders_ms()
    raise ValueError(f"cfg.modality must be 'rgb' or 'ms', got {cfg.modality!r}")


def get_dataloaders_rgb():
    DatasetClass = CachedEuroSAT if cfg.use_ram_cache else torchvision.datasets.EuroSAT

    full_train_ds = DatasetClass(root=cfg.data_root, download=True, transform=get_train_transforms())
    full_eval_ds = DatasetClass(root=cfg.data_root, download=False, transform=get_eval_transforms())

    n = len(full_train_ds)
    indices  = list(range(n))
    rng      = random.Random(cfg.seed)
    rng.shuffle(indices)

    n_train  = int(cfg.train_ratio * n)
    n_val    = int(cfg.val_ratio   * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[DATA] Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    loader_kwargs = dict(
        num_workers      = cfg.num_workers,
        pin_memory       = True,          
        persistent_workers = cfg.num_workers > 0,   
        prefetch_factor  = cfg.prefetch_factor if cfg.num_workers > 0 else None,  
    )

    train_loader = DataLoader(Subset(full_train_ds, train_idx), batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(Subset(full_eval_ds, val_idx), batch_size=cfg.batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(full_eval_ds, test_idx), batch_size=cfg.batch_size * 2, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def get_dataloaders_ms():
    full_ds = CachedEuroSATMS(cfg.ms_dataset_id)

    n = len(full_ds)
    indices = list(range(n))
    rng = random.Random(cfg.seed)
    rng.shuffle(indices)

    n_train = int(cfg.train_ratio * n)
    n_val   = int(cfg.val_ratio   * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[DATA-MS] Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    if cfg.use_geo_pretrained:
        # Match SSL4EO-S12's own preprocessing exactly (raw digital number / 10000,
        # i.e. approximate reflectance) rather than our own per-band z-score — using
        # whatever normalization the backbone was actually pretrained with matters
        # more here than it would for a from-ImageNet backbone.
        band_mean = torch.zeros(cfg.num_bands)
        band_std  = torch.full((cfg.num_bands,), 10000.0)
        print("[DATA-MS] Normalization: /10000 (matches SSL4EO-S12 pretraining) — "
              "skipping per-band stats computation, not used in this mode")
    else:
        band_mean, band_std = compute_band_stats(full_ds, train_idx)
        print(f"[DATA-MS] Band mean: {[round(v, 1) for v in band_mean.tolist()]}")
        print(f"[DATA-MS] Band std : {[round(v, 1) for v in band_std.tolist()]}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    stats_path = os.path.join(cfg.checkpoint_dir, "ms_band_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"mean": band_mean.tolist(), "std": band_std.tolist(), "geo_pretrained": cfg.use_geo_pretrained}, f)
    print(f"[DATA-MS] Band stats saved → {stats_path}  (TTA_ms.py reads this — don't hand-copy the numbers)")

    train_view = _TransformView(full_ds, MSTransform(band_mean, band_std, train=True))
    eval_view  = _TransformView(full_ds, MSTransform(band_mean, band_std, train=False))

    loader_kwargs = dict(
        num_workers         = cfg.num_workers_ms,
        pin_memory          = True,
        persistent_workers  = cfg.num_workers_ms > 0,
        prefetch_factor     = cfg.prefetch_factor if cfg.num_workers_ms > 0 else None,
    )

    train_loader = DataLoader(Subset(train_view, train_idx), batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(Subset(eval_view, val_idx), batch_size=cfg.batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(Subset(eval_view, test_idx), batch_size=cfg.batch_size * 2, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# ============================================================================
# 6. MODEL + PROGRESSIVE UNFREEZING
# ============================================================================

def adapt_patch_embed_to_multispectral(model: nn.Module, num_bands: int, rgb_band_indices) -> None:
    """
    Replaces the pretrained 3-channel (RGB) patch-embedding conv with a
    num_bands-channel version, so the ImageNet-pretrained ViT can consume the
    full Sentinel-2 stack instead of just RGB. Kernel size/stride/padding are
    untouched, so spatial output (14x14 patches) and pos_embed stay valid —
    only the input-channel dimension of this one conv changes.

    Init strategy:
      - The 3 channels at rgb_band_indices (default: Red=B04 idx3, Green=B03
        idx2, Blue=B02 idx1) get the EXACT pretrained RGB weights, since those
        bands are literally what those filters were trained to look at.
      - The other 10 bands (red-edge, NIR, SWIR, ...) have no RGB analogue, so
        they're seeded with the per-output-channel MEAN of the pretrained RGB
        weights — a standard, well-behaved starting point that keeps this
        conv's output magnitude close to what the rest of the pretrained
        network expects, rather than starting from random noise.
      - This layer is kept trainable across every stage (see unfreeze_blocks)
        so the approximate weights on those 10 bands can specialize.
    """
    old_proj = model.patch_embed.proj
    out_ch, in_ch, kh, kw = old_proj.weight.shape
    assert in_ch == 3, f"Expected a 3-channel pretrained conv, got {in_ch} input channels"

    new_proj = nn.Conv2d(num_bands, out_ch, kernel_size=(kh, kw), stride=old_proj.stride,
                          padding=old_proj.padding, bias=(old_proj.bias is not None))

    with torch.no_grad():
        mean_filter = old_proj.weight.mean(dim=1, keepdim=True)         # (out_ch, 1, kh, kw)
        new_proj.weight.copy_(mean_filter.repeat(1, num_bands, 1, 1))
        for new_idx, old_idx in zip(rgb_band_indices, range(3)):        # (3,2,1) <- (R,G,B) = (0,1,2)
            new_proj.weight[:, new_idx] = old_proj.weight[:, old_idx]
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    model.patch_embed.proj = new_proj
    print(f"[MS] patch_embed.proj: Conv2d(3 → {num_bands}, {kh}x{kw}). "
          f"Bands {rgb_band_indices} seeded from pretrained R,G,B; "
          f"remaining {num_bands - 3} bands seeded from their mean.")


def build_model(device: torch.device) -> nn.Module:
    if cfg.modality == "ms" and cfg.use_geo_pretrained:
        weights = getattr(ViTBase16_Weights, cfg.geo_weights)
        assert weights.meta["in_chans"] == cfg.num_bands, \
            f"{cfg.geo_weights} expects {weights.meta['in_chans']} channels, cfg.num_bands={cfg.num_bands}"
        model = geo_vit_base_patch16_224(weights=weights, num_classes=cfg.num_classes)
        print(f"[MS] Backbone: torchgeo {cfg.geo_weights} — SSL4EO-S12, MAE-pretrained on real "
              f"13-band Sentinel-2 imagery. All 13 patch_embed channels are genuinely pretrained "
              f"(norm + head are fresh — MAE checkpoints don't include a classification head).")
    elif cfg.modality == "ms":
        model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
        adapt_patch_embed_to_multispectral(model, num_bands=cfg.num_bands, rgb_band_indices=cfg.rgb_band_indices)
    else:
        model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    if cfg.modality == "ms" and not cfg.use_geo_pretrained:
        for param in model.patch_embed.parameters():
            param.requires_grad = True

    model.set_grad_checkpointing(enable=True)
    model.to(device)
    return model


def unfreeze_blocks(model: nn.Module, n_blocks: int) -> None:
    # Fully unwrap regardless of how many times torch.compile has been called on
    # this model across stages — each call adds another OptimizedModule layer,
    # and a single-level unwrap only strips the outermost one (see run_stage).
    base_model = model
    while hasattr(base_model, '_orig_mod'):
        base_model = base_model._orig_mod

    for param in base_model.parameters():
        param.requires_grad = False

    for param in base_model.head.parameters():
        param.requires_grad = True

    if cfg.modality == "ms" and not cfg.use_geo_pretrained:
        # Freshly-adapted, only-partially-pretrained layer — keep it trainable
        # every stage, not just stage 1.
        for param in base_model.patch_embed.parameters():
            param.requires_grad = True

    if n_blocks > 0:
        total = len(base_model.blocks)       
        start = max(0, total - n_blocks)
        for i, block in enumerate(base_model.blocks):
            if i >= start:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(base_model, 'norm'):
            for param in base_model.norm.parameters():
                param.requires_grad = True

    total_p     = sum(p.numel() for p in base_model.parameters())
    trainable_p = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
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
        num_batches  = step + 1
        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

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
# 9. TEST EVALUATION + OVERALL PROGRESS PLOT
# ============================================================================

def full_test_evaluation(model, test_loader, device, stage_name: str, save_prefix: str):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    macro_f1 = f1_score(labels, preds, average="macro")
    cm       = confusion_matrix(labels, preds)

    print(f"\n{'='*60}\nTEST RESULTS  —  {stage_name}\n{'='*60}")
    print(f"  Loss:      {test_loss:.4f}\n  Accuracy:  {test_acc*100:.2f}%\n  Macro F1:  {macro_f1:.4f}\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title=f"{stage_name}  —  Acc: {test_acc*100:.2f}%")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = os.path.join(cfg.checkpoint_dir, f"{save_prefix}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    return test_acc, macro_f1


def plot_overall_history(global_history: dict, checkpoint_dir: str):
    """Generates a multi-stage evaluation plot mapping all 3 sequential runs together."""
    train_loss = global_history["train_loss"]
    val_loss   = global_history["val_loss"]
    val_acc    = global_history["val_acc"]
    stage_ends = global_history["stage_transition_epochs"]
    
    if len(train_loss) == 0:
        return
    
    epochs_range = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves plot
    ax1.plot(epochs_range, train_loss, label="Train Loss", color="royalblue", lw=2)
    ax1.plot(epochs_range, val_loss,   label="Val Loss",   color="orange",    lw=2)
    ax1.set_xlabel("Aggregated Total Epochs", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Multi-Stage Fine-Tuning: Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    for idx, edge in enumerate(stage_ends[:-1]):
        ax1.axvline(x=edge, color="red", linestyle="--", alpha=0.6)
        ax1.text(edge + 0.5, max(train_loss)*0.85, f"Stage {idx+2}", color="red", fontweight="bold")
    ax1.text(0.5, max(train_loss)*0.85, "Stage 1", color="red", fontweight="bold")
    ax1.legend()

    # Accuracy progression plot
    ax2.plot(epochs_range, val_acc, color="green", label="Val Accuracy", lw=2)
    ax2.set_xlabel("Aggregated Total Epochs", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Multi-Stage Fine-Tuning: Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    for idx, edge in enumerate(stage_ends[:-1]):
        ax2.axvline(x=edge, color="red", linestyle="--", alpha=0.6)
        ax2.text(edge + 0.5, min(val_acc) + (max(val_acc)-min(val_acc))*0.1, f"Stage {idx+2}", color="red", fontweight="bold")
    ax2.text(0.5, min(val_acc) + (max(val_acc)-min(val_acc))*0.1, "Stage 1", color="red", fontweight="bold")
    ax2.legend()

    fig.tight_layout()
    out_path = os.path.join(checkpoint_dir, "overall_multistage_progression.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[PLOT] Combined multi-stage progression curve saved → {out_path}")


# ============================================================================
# 10. ONE FULL STAGE
# ============================================================================

def run_stage(stage_cfg: dict, stage_idx: int, model: nn.Module, train_loader, val_loader, test_loader, device: torch.device, global_history: dict):
    stage_name   = stage_cfg["name"]
    n_epochs     = stage_cfg["epochs"]
    lr           = stage_cfg["lr"]
    wd           = stage_cfg["wd"]
    n_blocks     = stage_cfg["unfreeze_blocks"]
    geo_tag      = "geo" if (cfg.modality == "ms" and cfg.use_geo_pretrained) else "own"
    save_prefix  = f"{cfg.modality}_{geo_tag}_stage{stage_idx+1}" if cfg.modality == "ms" else f"{cfg.modality}_stage{stage_idx+1}"
    ckpt_path    = os.path.join(cfg.checkpoint_dir, f"{save_prefix}_best.pth")

    print(f"\n{'#'*60}\n   {stage_name.upper()}\n{'#'*60}")

    # Step freeze settings
    unfreeze_blocks(model, n_blocks)

    # Re-allocation setup for compiled models to acknowledge grad changes
    if cfg.use_compile and device.type == "cuda":
        print("[COMPILE] Resetting Dynamo graph and re-compiling for new layer states...")
        torch._dynamo.reset()
        model = torch.compile(model)

    # Fully unwrap, however many compile layers have stacked up across stages —
    # a single-level unwrap here was the source of the "_orig_mod." checkpoint bug.
    base_model = model
    while hasattr(base_model, '_orig_mod'):
        base_model = base_model._orig_mod
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, base_model.parameters()), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = make_scaler(device)

    best_val_acc   = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"  [{stage_name}] Ep {epoch:02d}/{n_epochs} │ train: {train_loss:.4f} │ val: {val_loss:.4f} │ acc: {val_acc:.4f} │ {elapsed:.1f}s")

        # Track explicitly into shared chronological tracker
        global_history["train_loss"].append(train_loss)
        global_history["val_loss"].append(val_loss)
        global_history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(base_model.state_dict(), ckpt_path)
            print(f"    ↑ New best: {best_val_acc:.4f} → {ckpt_path}")

    # Track phase termination points
    current_total_epochs = len(global_history["train_loss"])
    global_history["stage_transition_epochs"].append(current_total_epochs)

    base_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    full_test_evaluation(model, test_loader, device, stage_name, save_prefix)

    return model, best_val_acc


# ============================================================================
# 11. MAIN
# ============================================================================

def main():
    seed_everything(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    if device.type == "cuda":
        print(f"  —  {torch.cuda.get_device_name(0)}")
    pretrain_note = f", {'geo-pretrained (SSL4EO-S12)' if cfg.use_geo_pretrained else 'own patch_embed adaptation'}" if cfg.modality == "ms" else ""
    print(f"[MODALITY] {cfg.modality}  ({cfg.num_bands if cfg.modality == 'ms' else 3} input channels{pretrain_note})")

    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_model(device)

    if cfg.start_from_stage > 0:
        ckpt = cfg.resume_checkpoint
        if os.path.exists(ckpt):
            print(f"\n[RESUME] Loading checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        else:
            print(f"[RESUME] ⚠ Checkpoint not found: {ckpt} — starting from scratch")
            cfg.start_from_stage = 0

    all_results = []
    
    # Unified historical runtime structure initialized here
    global_history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "stage_transition_epochs": []
    }

    for i, stage in enumerate(cfg.stages):
        if i < cfg.start_from_stage:
            print(f"\n[SKIP] Stage {i+1} ({stage['name']}) — already done")
            continue
        model, best_acc = run_stage(stage, i, model, train_loader, val_loader, test_loader, device, global_history=global_history)
        all_results.append((stage["name"], best_acc))

    # Trigger comprehensive graphical timeline display output creation
    plot_overall_history(global_history, cfg.checkpoint_dir)

    print(f"\n{'='*60}\nTRAINING COMPLETE — SUMMARY\n{'='*60}")
    for name, acc in all_results:
        print(f"  {name:<35} best val acc: {acc*100:.2f}%")


if __name__ == "__main__":
    main()