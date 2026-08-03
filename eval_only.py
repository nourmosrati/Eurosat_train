#!/usr/bin/env python3
"""
Run test-set evaluation on an already-trained checkpoint, without retraining.
Use this after train.py successfully saved a checkpoint but crashed during
the final test evaluation step (e.g. the Windows shared-memory issue).
"""

import gc
import torch

from train import (
    cfg,
    seed_everything,
    build_model,
    get_dataloaders,
    full_test_evaluation,
)


def main():
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a conservative worker count here too, and only build the test
    # loader's data — but get_dataloaders() builds all three at once, so
    # we free train/val loaders immediately after construction.
    model = build_model(device)

    print("\nLoading best checkpoint ...")
    model.load_state_dict(
        torch.load(cfg.best_model_path, map_location=device, weights_only=True)
    )

    train_loader, val_loader, test_loader = get_dataloaders()
    del train_loader, val_loader
    gc.collect()

    full_test_evaluation(model, test_loader, device)


if __name__ == "__main__":
    main()
