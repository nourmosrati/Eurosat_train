import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, classification_report
import random

# ----------------------------
# 1. Config
# ----------------------------
DATA_ROOT = Path("data/eurosat/2750")
CHECKPOINT_PATH = "checkpoints/stage3_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.10  # adjust to match whatever split you used in training
SEED = 42                # must match training seed for this to be YOUR actual test set

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ----------------------------
# 2. Build the same test split used in training
# ----------------------------
random.seed(SEED)

all_samples = []  # list of (path, label_idx)
for class_name in CLASS_NAMES:
    class_dir = DATA_ROOT / class_name
    images = sorted(class_dir.glob("*.jpg"))
    for img_path in images:
        all_samples.append((img_path, CLASS_TO_IDX[class_name]))

random.shuffle(all_samples)
test_size = int(len(all_samples) * TEST_SPLIT_RATIO)
test_samples = all_samples[:test_size]

print(f"Total images: {len(all_samples)}, Test set size: {len(test_samples)}")

# ----------------------------
# 3. Dataset / DataLoader
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

class EuroSATTestDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

test_dataset = EuroSATTestDataset(test_samples, preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------
# 4. Build BASE model (ImageNet-pretrained, untrained head)
# ----------------------------
base_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
base_model.to(DEVICE)
base_model.eval()

# ----------------------------
# 5. Build FINE-TUNED model (optional, for direct comparison)
# ----------------------------
finetuned_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
finetuned_model.to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
finetuned_model.load_state_dict(cleaned_state_dict)
finetuned_model.eval()

# ----------------------------
# 6. Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate(model, loader):
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return all_labels, all_preds

# ----------------------------
# 7. Run evaluation and compute F1
# ----------------------------
if __name__ == "__main__":
    print("\nEvaluating BASE model (ImageNet-pretrained, untrained head)...")
    base_labels, base_preds = evaluate(base_model, test_loader)
    base_f1_macro = f1_score(base_labels, base_preds, average="macro")
    base_f1_weighted = f1_score(base_labels, base_preds, average="weighted")

    print(f"\nBase model F1 (macro):    {base_f1_macro:.4f}")
    print(f"Base model F1 (weighted): {base_f1_weighted:.4f}")
    print("\nBase model classification report:")
    print(classification_report(base_labels, base_preds, target_names=CLASS_NAMES, zero_division=0))

    print("\nEvaluating FINE-TUNED model...")
    ft_labels, ft_preds = evaluate(finetuned_model, test_loader)
    ft_f1_macro = f1_score(ft_labels, ft_preds, average="macro")
    ft_f1_weighted = f1_score(ft_labels, ft_preds, average="weighted")

    print(f"\nFine-tuned model F1 (macro):    {ft_f1_macro:.4f}")
    print(f"Fine-tuned model F1 (weighted): {ft_f1_weighted:.4f}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<25}{'Base':<15}{'Fine-tuned':<15}")
    print(f"{'F1 (macro)':<25}{base_f1_macro:<15.4f}{ft_f1_macro:<15.4f}")
    print(f"{'F1 (weighted)':<25}{base_f1_weighted:<15.4f}{ft_f1_weighted:<15.4f}")