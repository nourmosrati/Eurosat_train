# ViT Fine-Tuning on EuroSAT — Land Cover Classification

Fine-tune a Vision Transformer (`vit_base_patch16_224`) on **EuroSAT** (27,000 Sentinel-2 satellite patches, 10 land-cover classes) using head-only training, mixed precision, and gradient checkpointing — all within **6 GB VRAM**.

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn matplotlib tqdm

# 2. Run training (downloads EuroSAT automatically on first run)
python train.py
```

## What the Script Does

| Stage | Details |
|-------|---------|
| **Data** | Downloads EuroSAT via `torchvision.datasets.EuroSAT`, splits 70/15/15 |
| **Model** | `timm` `vit_base_patch16_224` with ImageNet weights |
| **Training** | Head-only fine-tuning, backbone frozen |
| **Memory** | Mixed precision (FP16), gradient checkpointing, batch 16 × 4 accum |
| **Optimizer** | AdamW + CosineAnnealingLR, label smoothing 0.1 |
| **Epochs** | 20, best checkpoint saved by val accuracy |
| **Evaluation** | Accuracy, macro F1, per-class report, confusion matrix PNG |

## Outputs

After training completes you'll find:

```
checkpoints/
├── best_vit_eurosat.pth      # Best model weights
├── training_curves.png        # Loss & accuracy plots
└── confusion_matrix.png       # Test-set confusion matrix
```

## Estimated Training Time

| Hardware | Per Epoch | Full Run (20 epochs) |
|----------|-----------|----------------------|
| **RTX 3050 Laptop (6 GB)** | ~45–60 s | ~15–20 min |
| **Kaggle / Colab T4 (16 GB)** | ~30–40 s | ~10–13 min |

> Times are for head-only fine-tuning. Full fine-tuning would be significantly slower.

## Expected Accuracy

Head-only fine-tuning typically reaches **92–95%** test accuracy on EuroSAT after 20 epochs. This is strong for a frozen backbone — the ImageNet features transfer well to satellite RGB patches.

## Ways to Improve Accuracy (≤ 6 GB VRAM)

1. **Gradual Unfreezing** — After the head converges (~5 epochs), unfreeze the last 2–3 transformer blocks and continue training with a 10× lower learning rate. This lets the backbone adapt its high-level features to satellite imagery while keeping VRAM usage manageable.

2. **Stronger Augmentation (RandAugment / TrivialAugment)** — Replace the manual pipeline with `torchvision.transforms.v2.RandAugment(num_ops=2, magnitude=9)`. This regularizes more aggressively and typically adds 1–2% accuracy. Satellite-specific augmentations like random 90° rotations also help since satellite images have no canonical "up."

3. **Use a Smaller, More Efficient Backbone** — Switch to `vit_small_patch16_224` (22M params vs 86M) and do *full* fine-tuning. The smaller model fits in 6 GB with full gradients, and end-to-end training on domain-specific data often beats head-only on a larger model. Alternatively, try `convnext_tiny` or `swin_tiny_patch4_window7_224` which have better inductive biases for spatial data.

## License

EuroSAT is released under MIT License. Model weights from timm are Apache-2.0.
