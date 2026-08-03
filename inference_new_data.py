import torch
from torchvision import transforms as T
import timm
from pathlib import Path
from PIL import Image

# ----------------------------
# 1. Config — matches train.py exactly
# ----------------------------
CHECKPOINT_PATH = "checkpoints/best_vit_eurosat.pth"
NEW_DATA_DIR = Path("new_images")  # put your new images here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------
# 2. Rebuild model architecture
# ----------------------------
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
model.to(DEVICE)

# ----------------------------
# 3. Load checkpoint (raw state dict, as saved by train.py)
# ----------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(cleaned_state_dict)
model.eval()

# ----------------------------
# 4. Eval transform — MUST match train.py's get_eval_transforms exactly
# ----------------------------
eval_transform = T.Compose([
    T.Resize(IMG_SIZE + 32),   # Resize to 256
    T.CenterCrop(IMG_SIZE),    # Crop to 224
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ----------------------------
# 5. Inference on a single image
# ----------------------------
@torch.no_grad()
def predict(image_path, top_k=3):
    img = Image.open(image_path).convert("RGB")
    x = eval_transform(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idxs = torch.topk(probs, top_k)
    return [(CLASS_NAMES[i], p.item()) for i, p in zip(top_idxs, top_probs)]

# ----------------------------
# 6. Run over every image in NEW_DATA_DIR
# ----------------------------
if __name__ == "__main__":
    image_paths = sorted(
        list(NEW_DATA_DIR.glob("*.jpg")) +
        list(NEW_DATA_DIR.glob("*.png")) +
        list(NEW_DATA_DIR.glob("*.jpeg")) +
        list(NEW_DATA_DIR.glob("*.tif"))
    )

    if not image_paths:
        print(f"No images found in {NEW_DATA_DIR}. Add some images there first.")
    else:
        for img_path in image_paths:
            results = predict(img_path)
            print(f"\n{img_path.name}")
            for cls, prob in results:
                print(f"  {cls:<25} {prob:.4f}")