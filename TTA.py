"""
Test-Time Augmentation (TTA) evaluation for the ViT-B/16 EuroSAT Stage 3 checkpoint.

WHY THIS VERSION IS DIFFERENT FROM THE FIRST ONE:
Your training script (train_v2.py) has no physical train/test folders — it
builds ONE dataset of all 27,000 images, then does:
    indices = list(range(27000))
    random.Random(42).shuffle(indices)
    train_idx = indices[:18900]   (70%)
    val_idx   = indices[18900:22950]  (15%)
    test_idx  = indices[22950:]   (15%, = 4050 images)

If we don't reproduce that exact shuffle, "TTA accuracy" would be measured
on the wrong images — possibly including ones the model was trained on,
which would silently inflate the number. This script rebuilds that exact
split so the comparison against your reported 98.30% is valid.

It also matches your exact eval preprocessing: Resize(256) -> CenterCrop(224),
not a direct resize to 224x224.

No retraining. Just run:
    python TTA.py
"""

import random
import torch
import torch.nn.functional as F
import timm
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# ============================== CONFIG ==============================
# These must match the Config class in your training script exactly.
SEED        = 42
DATA_ROOT   = "./data"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
IMG_SIZE    = 224

CHECKPOINT_PATH = "checkpoints/stage3_best.pth"   # the v2, progressive-unfreeze final model
MODEL_NAME      = "vit_base_patch16_224"
NUM_CLASSES     = 10
BATCH_SIZE      = 32
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
# ======================================================================


def build_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def recover_test_indices(n: int):
    """
    Mirrors get_dataloaders() in train_v2.py exactly:
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
    print("        (compare these numbers to your training log's [DATA] line — "
          "they should match exactly)")
    return test_idx


def get_eval_transform():
    # Matches get_eval_transforms() in train_v2.py: Resize(256) -> CenterCrop(224)
    return T.Compose([
        T.Resize(IMG_SIZE + 32),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def view_original(img):
    return img

def view_hflip(img):
    return T.functional.hflip(img)

def view_vflip(img):
    return T.functional.vflip(img)

def view_rot90(img):
    return T.functional.rotate(img, 90)

def view_rot270(img):
    return T.functional.rotate(img, 270)


# Named top-level functions instead of lambdas — lambdas can't be pickled by
# Windows' multiprocessing "spawn" start method, which DataLoader workers need.
TTA_VIEWS = {
    "original": view_original,
    "hflip":    view_hflip,
    "vflip":    view_vflip,
    "rot90":    view_rot90,
    "rot270":   view_rot270,
}


class TTAView(Dataset):
    """Wraps the base (transform=None) EuroSAT dataset, restricted to test_idx,
    applying one TTA view + the standard eval transform."""
    def __init__(self, base_dataset, indices, view_fn, eval_transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = T.Compose([view_fn, eval_transform])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]   # PIL image, since base transform=None
        return self.transform(img), label


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

    # Same underlying dataset your training script used. download=False since
    # it's already on disk under DATA_ROOT/eurosat/2750/<ClassName>/*.jpg
    base_dataset = torchvision.datasets.EuroSAT(root=DATA_ROOT, download=False, transform=None)
    print(f"[DATA] Base dataset size: {len(base_dataset)} "
          f"(should be 27000 for full EuroSAT RGB)")

    test_idx = recover_test_indices(len(base_dataset))
    eval_transform = get_eval_transform()
    model = build_model()

    view_probs = {}
    labels_ref = None

    for name, view_fn in TTA_VIEWS.items():
        dataset = TTAView(base_dataset, test_idx, view_fn, eval_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        probs, labels = get_probs_and_labels(model, loader)
        view_probs[name] = probs
        if labels_ref is None:
            labels_ref = labels
        else:
            assert torch.equal(labels_ref, labels), "Label order mismatch between views!"
        print(f"  {name:>10s} view alone: {accuracy(probs, labels) * 100:.2f}%")

    baseline_acc = accuracy(view_probs["original"], labels_ref)
    stacked = torch.stack(list(view_probs.values()), dim=0)
    avg_probs = stacked.mean(dim=0)
    tta_acc = accuracy(avg_probs, labels_ref)

    print("\n===================== RESULTS =====================")
    print(f"Baseline (no TTA):   {baseline_acc * 100:.2f}%")
    print(f"With TTA (5 views):  {tta_acc * 100:.2f}%")
    print(f"Delta:               {(tta_acc - baseline_acc) * 100:+.2f}%")
    print("=====================================================")

    if abs(baseline_acc * 100 - 98.30) > 0.5:
        print("\n[WARNING] Baseline came out noticeably different from your reported "
              "98.30%. That likely means a config value here (seed/ratios/data_root) "
              "doesn't match what you actually used for training. Don't trust the TTA "
              "delta until this lines up — let me know the number you got.")


if __name__ == "__main__":
    main()