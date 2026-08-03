#!/usr/bin/env python3
"""
Hyperparameter search over learning rate and weight decay.
Imports model/data/training functions from train.py instead of duplicating them.

Run from the same folder as train.py:
    python lr.py
"""

import itertools
import json
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------
# Import everything we need from train.py.
# Because train.py guards its training run with `if __name__ == "__main__":`,
# importing it here does NOT trigger a full training run — only the
# top-level code (Config class, function defs, cfg = Config()) executes.
# ----------------------------------------------------------------------
from train import (
    cfg,
    seed_everything,
    build_model,
    find_max_batch_size,
    get_dataloaders,
    make_scaler,
    train_one_epoch,
    evaluate,
)

# ----------------------------
# Search space
# ----------------------------
LR_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3]
WD_CANDIDATES = [1e-3, 1e-2, 5e-2]
SEARCH_EPOCHS = 3   # short runs per combo — not full 10 epochs, to save time


def run_search():
    results = []

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force single-process data loading for this search script.
    # Windows' spawn-based multiprocessing re-imports this file (and train.py)
    # in every worker process, which has repeatedly caused stale-cache /
    # re-entrancy crashes across many back-to-back short runs. num_workers=0
    # avoids that entire class of bugs; the small loading-speed cost is
    # negligible next to the ~100s/epoch GPU compute time.
    cfg.num_workers = 0

    for lr, wd in itertools.product(LR_CANDIDATES, WD_CANDIDATES):
        print(f"\n{'='*60}")
        print(f"Testing lr={lr}  weight_decay={wd}")
        print(f"{'='*60}")

        cfg.lr = lr
        cfg.weight_decay = wd

        model = build_model(device)
        cfg.batch_size = find_max_batch_size(model, device)
        train_loader, val_loader, _ = get_dataloaders()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=SEARCH_EPOCHS, eta_min=1e-6
        )
        scaler = make_scaler(device)

        best_val_acc = 0.0
        diverged = False

        for epoch in range(1, SEARCH_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)

            if train_loss != train_loss:  # NaN check (NaN != NaN is True in Python)
                print(f"  epoch {epoch}: train_loss=NaN — diverged, skipping rest of this combo")
                diverged = True
                break

            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            print(f"  epoch {epoch}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
            best_val_acc = max(best_val_acc, val_acc)

        results.append({
            "lr": lr,
            "weight_decay": wd,
            "best_val_acc": best_val_acc,
            "diverged": diverged,
        })

    # ----------------------------
    # Report results, sorted best first
    # ----------------------------
    results.sort(key=lambda r: r["best_val_acc"], reverse=True)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH RESULTS (sorted by val_acc)")
    print("=" * 60)
    for r in results:
        flag = "  [DIVERGED]" if r["diverged"] else ""
        print(f"  lr={r['lr']:<8}  weight_decay={r['weight_decay']:<8}  val_acc={r['best_val_acc']:.4f}{flag}")

    with open("hparam_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest combo: lr={results[0]['lr']}, weight_decay={results[0]['weight_decay']}, "
          f"val_acc={results[0]['best_val_acc']:.4f}")


# ----------------------------------------------------------------------
# CRITICAL on Windows: DataLoader(num_workers>0) spawns new Python
# processes that re-import this file from scratch. Without this guard,
# every worker process re-runs the entire search loop, which is what
# caused the duplicate output and the RuntimeError crash.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_search()