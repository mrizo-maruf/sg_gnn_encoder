#!/usr/bin/env python3
"""
Training script for Scene Graph GNN Encoder with contrastive learning.

Usage:
    python train.py --data_dir Data/ --out_dir runs --epochs 200 --seed 0

Implements:
  - SimCLR-style graph contrastive learning with InfoNCE loss
  - Optional relation prediction auxiliary loss
  - Early stopping on validation loss
  - Overfitting detection
  - Checkpoint saving, metrics CSV, config JSON
  - Training visualizations (loss curves, PCA embeddings)
"""

import argparse
import csv
import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from augment import augment_graph
from dataset import (
    SceneGraphDataset,
    SimpleSceneGraphDataset,
    load_all_json_paths,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    SIMPLE_NODE_FEAT_DIM,
    SIMPLE_EDGE_FEAT_DIM,
)
from losses import CombinedLoss
from model import SceneGraphEncoder, SceneGraphEncoderLight, SceneGraphEncoderSimple, SceneGraphEncoderSimple3Layer
from visualize import plot_loss_curves, plot_overfitting_gap, plot_embedding_pca


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dataset(
    json_paths: List[Path], train_ratio: float = 0.8, seed: int = 0
) -> Tuple[List[Path], List[Path]]:
    """
    Split JSON file paths into train/val sets by scene files.

    Args:
        json_paths: Sorted list of JSON file paths.
        train_ratio: Fraction of files for training.
        seed: Random seed for shuffling.

    Returns:
        (train_paths, val_paths)
    """
    paths = list(json_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_train = max(1, int(len(paths) * train_ratio))
    return paths[:n_train], paths[n_train:]


def collate_augmented_pairs(batch: List[Data], device: torch.device) -> Tuple[Batch, Batch]:
    """
    Given a batch of Data objects, create two augmented views and collate.

    Args:
        batch: List of PyG Data objects (original graphs).
        device: Target device.

    Returns:
        (batch1, batch2): Two batched augmented views.
    """
    views1 = []
    views2 = []
    for data in batch:
        v1 = augment_graph(data)
        v2 = augment_graph(data)
        views1.append(v1)
        views2.append(v2)

    batch1 = Batch.from_data_list(views1).to(device)
    batch2 = Batch.from_data_list(views2).to(device)
    return batch1, batch2


# ──────────────────────────────────────────────────────────────────────
# Training / Evaluation loops
# ──────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: SceneGraphEncoder,
    dataset: SceneGraphDataset,
    loss_fn: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The GATv2 encoder model.
        dataset: Training dataset.
        loss_fn: Combined loss function.
        optimizer: Optimizer.
        device: Compute device.
        batch_size: Batch size.

    Returns:
        Dict with average 'total', 'contrast', 'relation' losses.
    """
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    total_losses = {"total": 0.0, "contrast": 0.0, "relation": 0.0}
    n_batches = 0

    for batch_list in loader:
        # batch_list is a Batch — convert back to list of Data for augmentation
        data_list = batch_list.to_data_list()

        # Skip degenerate batches
        if len(data_list) < 2:
            continue

        batch1, batch2 = collate_augmented_pairs(data_list, device)

        # Forward pass — view 1
        out1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        # Forward pass — view 2
        out2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        z1 = out1["embedding"]  # [B, 32]
        z2 = out2["embedding"]  # [B, 32]

        # Relation loss uses edge representations from view 1
        losses = loss_fn(
            z1, z2,
            edge_repr=out1["edge_repr"],
            edge_attr=batch1.edge_attr,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    if n_batches > 0:
        for k in total_losses:
            total_losses[k] /= n_batches

    return total_losses


@torch.no_grad()
def evaluate(
    model: SceneGraphEncoder,
    dataset: SceneGraphDataset,
    loss_fn: CombinedLoss,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Evaluate on validation set.

    Returns:
        Dict with average 'total', 'contrast', 'relation' losses.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_losses = {"total": 0.0, "contrast": 0.0, "relation": 0.0}
    n_batches = 0

    for batch_list in loader:
        data_list = batch_list.to_data_list()
        if len(data_list) < 2:
            continue

        batch1, batch2 = collate_augmented_pairs(data_list, device)

        out1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        out2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        z1 = out1["embedding"]
        z2 = out2["embedding"]

        losses = loss_fn(
            z1, z2,
            edge_repr=out1["edge_repr"],
            edge_attr=batch1.edge_attr,
        )

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    if n_batches > 0:
        for k in total_losses:
            total_losses[k] /= n_batches

    return total_losses


def detect_overfitting(
    train_losses: List[float],
    val_losses: List[float],
    window: int = 10,
    gap_threshold: float = 0.3,
) -> Tuple[bool, str]:
    """
    Detect overfitting by analyzing train-val loss divergence.

    Checks:
      1. Growing gap: val_loss - train_loss increasing over recent window.
      2. Absolute gap: recent average gap exceeds threshold.
      3. Val plateau: val_loss not improving while train_loss still dropping.

    Args:
        train_losses: Train loss history.
        val_losses: Validation loss history.
        window: Window size for analysis.
        gap_threshold: Threshold for absolute gap warning.

    Returns:
        (is_overfitting, message)
    """
    if len(train_losses) < window:
        return False, ""

    recent_gaps = [
        v - t for t, v in zip(train_losses[-window:], val_losses[-window:])
    ]
    avg_gap = sum(recent_gaps) / len(recent_gaps)

    # Check if gap is increasing
    first_half = recent_gaps[: window // 2]
    second_half = recent_gaps[window // 2 :]
    gap_increasing = (sum(second_half) / len(second_half)) > (
        sum(first_half) / len(first_half) + 0.01
    )

    # Check absolute gap
    large_gap = avg_gap > gap_threshold

    # Check val plateau (val not improving in last window epochs)
    val_recent = val_losses[-window:]
    val_not_improving = min(val_recent) >= min(val_losses) and len(val_losses) > window

    if gap_increasing and large_gap:
        return True, (
            f"⚠ OVERFITTING DETECTED: Train-val gap increasing "
            f"(avg gap={avg_gap:.4f}, threshold={gap_threshold})"
        )
    elif large_gap and val_not_improving:
        return True, (
            f"⚠ OVERFITTING WARNING: Large train-val gap ({avg_gap:.4f}) "
            f"and val loss plateaued"
        )
    return False, ""


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train Scene Graph GNN Encoder")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to JSON scene graph directory")
    parser.add_argument("--out_dir", type=str, default="runs", help="Output directory (default: runs)")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature")
    parser.add_argument("--lambda_rel", type=float, default=0.5, help="Relation loss weight")
    parser.add_argument("--no_relation_loss", action="store_true", help="Disable relation prediction loss")
    parser.add_argument("--model", type=str, default="3layer", choices=["3layer", "2layer", "simple", "simple3layer"],
                        help="Model architecture: '3layer' | '2layer' | 'simple' (2-layer 6-dim) | 'simple3layer' (3-layer 6-dim)")
    parser.add_argument("--hidden_dim", type=int, default=40, help="GATv2 hidden dimension")
    parser.add_argument("--output_dim", type=int, default=32, help="Scene embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    args = parser.parse_args()

    # ---- Setup ----
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[init] Seed set to {args.seed}")

    device = get_device()
    print(f"[init] Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"[init] Run directory: {run_dir}")

    # ---- Save config ----
    config = vars(args)
    config["timestamp"] = timestamp
    config["device"] = str(device)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Load data ----
    json_paths = load_all_json_paths(args.data_dir)
    print(f"[data] Found {len(json_paths)} JSON files in {args.data_dir}")

    if len(json_paths) < 2:
        warnings.warn(
            f"Only {len(json_paths)} scene(s) found. "
            "Contrastive learning requires ≥ 2 scenes in each batch. "
            "Training with duplicated data for demonstration."
        )

    # Split
    seed_for_split = args.seed if args.seed is not None else 0
    train_paths, val_paths = split_dataset(json_paths, args.train_ratio, seed_for_split)

    # If too few files, ensure both sets have data
    if len(val_paths) == 0:
        val_paths = train_paths.copy()
        warnings.warn("Val set empty — using train set for validation")
    if len(train_paths) == 0:
        train_paths = val_paths.copy()
        warnings.warn("Train set empty — using val set for training")

    DatasetClass = SimpleSceneGraphDataset if args.model in ("simple", "simple3layer") else SceneGraphDataset
    train_dataset = DatasetClass(args.data_dir, json_paths=train_paths)
    val_dataset = DatasetClass(args.data_dir, json_paths=val_paths)
    # Full dataset for embedding visualization
    full_dataset = DatasetClass(args.data_dir)

    print(f"[data] Train: {len(train_dataset)} graphs, Val: {len(val_dataset)} graphs")

    # ---- Build model ----
    if args.model == "simple":
        ModelClass = SceneGraphEncoderSimple
        nfd, efd = SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM
    elif args.model == "simple3layer":
        ModelClass = SceneGraphEncoderSimple3Layer
        nfd, efd = SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM
    elif args.model == "2layer":
        ModelClass = SceneGraphEncoderLight
        nfd, efd = NODE_FEAT_DIM, EDGE_FEAT_DIM
    else:
        ModelClass = SceneGraphEncoder
        nfd, efd = NODE_FEAT_DIM, EDGE_FEAT_DIM
    model = ModelClass(
        node_feat_dim=nfd,
        edge_feat_dim=efd,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"[model] Architecture: {args.model} ({ModelClass.__name__})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"[model] Output embedding dim: {args.output_dim}")

    # ---- Loss ----
    use_rel_loss = not args.no_relation_loss
    loss_fn = CombinedLoss(
        temperature=args.temperature,
        lambda_rel=args.lambda_rel,
        use_relation_loss=use_rel_loss,
        edge_hidden_dim=args.hidden_dim,
        num_relations=6,
    ).to(device)

    # ---- Optimizer + Scheduler ----
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- Metrics tracking ----
    metrics = {
        "train_total": [],
        "train_contrast": [],
        "train_relation": [],
        "val_total": [],
        "val_contrast": [],
        "val_relation": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # ---- CSV writer ----
    csv_path = run_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_total", "train_contrast", "train_relation",
        "val_total", "val_contrast", "val_relation", "lr", "overfitting",
    ])

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"  Starting training for {args.epochs} epochs")
    print(f"  InfoNCE τ={args.temperature}, λ_rel={args.lambda_rel}")
    print(f"  Relation loss: {'ON' if use_rel_loss else 'OFF'}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_dataset, loss_fn, optimizer, device, args.batch_size
        )

        # Validate
        val_loss = evaluate(model, val_dataset, loss_fn, device, args.batch_size)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record
        for k in ["total", "contrast", "relation"]:
            metrics[f"train_{k}"].append(train_loss[k])
            metrics[f"val_{k}"].append(val_loss[k])

        # Overfitting detection
        overfit, overfit_msg = detect_overfitting(
            metrics["train_total"], metrics["val_total"]
        )

        # CSV
        csv_writer.writerow([
            epoch,
            f"{train_loss['total']:.6f}",
            f"{train_loss['contrast']:.6f}",
            f"{train_loss['relation']:.6f}",
            f"{val_loss['total']:.6f}",
            f"{val_loss['contrast']:.6f}",
            f"{val_loss['relation']:.6f}",
            f"{current_lr:.6f}",
            "YES" if overfit else "NO",
        ])
        csv_file.flush()

        # Logging
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:>4d}/{args.epochs} | "
            f"Train: {train_loss['total']:.4f} (C:{train_loss['contrast']:.4f} R:{train_loss['relation']:.4f}) | "
            f"Val: {val_loss['total']:.4f} (C:{val_loss['contrast']:.4f} R:{val_loss['relation']:.4f}) | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )
        if overfit:
            print(f"  {overfit_msg}")

        # ---- Early stopping / checkpoint ----
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss_fn_state_dict": loss_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                run_dir / "best_checkpoint.pt",
            )
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\n[early stop] No improvement for {args.patience} epochs. "
                    f"Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}"
                )
                break

    csv_file.close()
    total_time = time.time() - t_start
    print(f"\n[done] Training completed in {total_time:.1f}s")
    print(f"[done] Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.4f}")

    # ── Visualizations ──
    print("\n[viz] Generating plots...")
    plot_loss_curves(metrics, str(run_dir))
    plot_overfitting_gap(metrics, str(run_dir))

    # Load best model for embedding visualization
    ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model = ModelClass(
        node_feat_dim=nfd,
        edge_feat_dim=efd,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    plot_embedding_pca(model, full_dataset, str(run_dir), device)

    print(f"\n[done] All outputs saved to {run_dir}/")
    print(f"  ├── best_checkpoint.pt")
    print(f"  ├── config.json")
    print(f"  ├── metrics.csv")
    print(f"  └── plots/")
    print(f"       ├── loss_curves.png")
    print(f"       ├── overfitting_gap.png")
    print(f"       └── embedding_pca.png")


if __name__ == "__main__":
    main()
