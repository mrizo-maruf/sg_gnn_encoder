#!/usr/bin/env python3
"""
Hyperparameter sweep for the Simple Scene Graph Encoder.

Searches over: learning rate, hidden dimension, regularization (dropout + weight_decay).
Uses only SceneGraphEncoderSimple with 6-dim node/edge features.

Selects the best configuration using Alignment & Uniformity metrics
(Wang & Isola, 2020), which directly measure contrastive embedding quality:

  Alignment:   E[ ||z_i - z_i+||^2 ]           (lower = positive pairs closer)
  Uniformity:  log E[ e^{-2||z_i - z_j||^2} ]   (lower = more uniform spread)

Usage:
    python sweep_simple.py --data_dir Data/ --out_dir sweeps_simple --epochs 100 --seed 42
    python sweep_simple.py --data_dir Data/ --out_dir sweeps_simple --epochs 100 --seed 42 --quick
"""

import argparse
import csv
import itertools
import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from augment import augment_graph
from dataset import SimpleSceneGraphDataset, load_all_json_paths, SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM
from losses import CombinedLoss
from model import SceneGraphEncoderSimple


# ──────────────────────────────────────────────────────────────────────
# Metrics: Alignment & Uniformity
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_alignment(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Alignment: average squared L2 distance between positive pairs.
    Lower is better — positive pairs should map close together.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return (z1 - z2).norm(dim=1).pow(alpha).mean().item()


@torch.no_grad()
def compute_uniformity(z: torch.Tensor, t: float = 2.0) -> float:
    """
    Uniformity: log of average pairwise Gaussian potential.
    Lower is better — embeddings spread uniformly on the hypersphere.
    """
    z = F.normalize(z, dim=1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    kernel = torch.exp(-t * sq_pdist[mask])
    return torch.log(kernel.mean()).item()


@torch.no_grad()
def evaluate_alignment_uniformity(
    model,
    dataset,
    device: torch.device,
    n_aug_passes: int = 3,
) -> Dict[str, float]:
    """
    Compute alignment & uniformity on a dataset.
    """
    model.eval()

    # ---- Collect original embeddings ----
    all_z = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        all_z.append(out["embedding"])
    all_z = torch.cat(all_z, dim=0)

    # ---- Uniformity ----
    uniformity = compute_uniformity(all_z)

    # ---- Alignment ----
    alignments = []
    for _ in range(n_aug_passes):
        z1_list, z2_list = [], []
        for data in loader:
            data = data.to(device)
            v1 = augment_graph(data.to("cpu")).to(device)
            v2 = augment_graph(data.to("cpu")).to(device)

            batch_vec = torch.zeros(v1.x.size(0), dtype=torch.long, device=device)
            out1 = model(v1.x, v1.edge_index, v1.edge_attr, batch_vec)
            batch_vec2 = torch.zeros(v2.x.size(0), dtype=torch.long, device=device)
            out2 = model(v2.x, v2.edge_index, v2.edge_attr, batch_vec2)

            z1_list.append(out1["embedding"])
            z2_list.append(out2["embedding"])

        z1_all = torch.cat(z1_list, dim=0)
        z2_all = torch.cat(z2_list, dim=0)
        alignments.append(compute_alignment(z1_all, z2_all))

    alignment = float(np.mean(alignments))

    return {
        "alignment": alignment,
        "uniformity": uniformity,
        "combined": alignment + uniformity,
    }


# ──────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_augmented_pairs(data_list, device):
    views1, views2 = [], []
    for data in data_list:
        views1.append(augment_graph(data))
        views2.append(augment_graph(data))
    batch1 = Batch.from_data_list(views1).to(device)
    batch2 = Batch.from_data_list(views2).to(device)
    return batch1, batch2


def train_one_epoch(model, dataset, loss_fn, optimizer, device, batch_size=8):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    total_loss = 0.0
    n = 0
    for batch_list in loader:
        data_list = batch_list.to_data_list()
        if len(data_list) < 2:
            continue
        batch1, batch2 = collate_augmented_pairs(data_list, device)
        out1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        out2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        losses = loss_fn(out1["embedding"], out2["embedding"],
                         edge_repr=out1["edge_repr"], edge_attr=batch1.edge_attr)
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += losses["total"].item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(model, dataset, loss_fn, device, batch_size=8):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    n = 0
    for batch_list in loader:
        data_list = batch_list.to_data_list()
        if len(data_list) < 2:
            continue
        batch1, batch2 = collate_augmented_pairs(data_list, device)
        out1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        out2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        losses = loss_fn(out1["embedding"], out2["embedding"],
                         edge_repr=out1["edge_repr"], edge_attr=batch1.edge_attr)
        total_loss += losses["total"].item()
        n += 1
    return total_loss / max(n, 1)


def train_config(
    config: dict,
    train_dataset,
    val_dataset,
    full_dataset,
    device: torch.device,
    seed: int,
) -> dict:
    """Train a single configuration and return metrics."""
    set_seed(seed)

    model = SceneGraphEncoderSimple(
        node_feat_dim=SIMPLE_NODE_FEAT_DIM,
        edge_feat_dim=SIMPLE_EDGE_FEAT_DIM,
        hidden_dim=config["hidden_dims"],
        output_dim=config["output_dims"],
        dropout=config["dropouts"],
    ).to(device)

    loss_fn = CombinedLoss(
        temperature=config["temperatures"],
        lambda_rel=config["lambda_rels"],
        use_relation_loss=True,
        edge_hidden_dim=config["hidden_dims"],
        num_relations=6,
    ).to(device)

    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = AdamW(all_params, lr=config["lrs"], weight_decay=config["weight_decays"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lrs"] * 0.01)

    n_params = sum(p.numel() for p in model.parameters())

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    patience = 30
    train_losses = []
    val_losses = []

    for epoch in range(1, config["epochs"] + 1):
        t_loss = train_one_epoch(model, train_dataset, loss_fn, optimizer, device)
        v_loss = eval_loss(model, val_dataset, loss_fn, device)
        scheduler.step()
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model and compute alignment/uniformity
    model.load_state_dict(best_state)
    au_metrics = evaluate_alignment_uniformity(model, full_dataset, device)

    return {
        **config,
        "n_params": n_params,
        "best_val_loss": best_val,
        "final_epoch": len(train_losses),
        "alignment": au_metrics["alignment"],
        "uniformity": au_metrics["uniformity"],
        "au_combined": au_metrics["combined"],
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────


def plot_sweep_results(results: List[dict], out_dir: str) -> None:
    """Generate all sweep visualizations."""
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Alignment vs Uniformity scatter ----
    fig, ax = plt.subplots(figsize=(9, 7))
    aligns = [r["alignment"] for r in results]
    uniforms = [r["uniformity"] for r in results]
    colors = [r["hidden_dims"] for r in results]
    sc = ax.scatter(uniforms, aligns, c=colors, marker="^", s=100,
                    edgecolors="black", linewidth=0.5, alpha=0.85,
                    cmap="viridis", label="simple")

    # Mark the best config
    best = min(results, key=lambda r: r["au_combined"])
    ax.scatter(best["uniformity"], best["alignment"],
               s=250, facecolors="none", edgecolors="red", linewidths=2.5,
               zorder=5, label=f"Best (#{results.index(best)+1})")

    ax.set_xlabel("Uniformity  (lower = more spread, better)", fontsize=11)
    ax.set_ylabel("Alignment  (lower = tighter positives, better)", fontsize=11)
    ax.set_title("Alignment vs Uniformity — Simple Model Sweep", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("hidden_dim")
    plt.tight_layout()
    fig.savefig(str(plots_dir / "alignment_vs_uniformity.png"), dpi=150)
    plt.close(fig)

    # ---- 2. Bar chart: combined score per config ----
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.6), 6))
    labels = []
    au_scores = []
    bar_colors = []
    for i, r in enumerate(sorted(results, key=lambda x: x["au_combined"])):
        lbl = (f"H={r['hidden_dims']}\nO={r['output_dims']}\n"
               f"lr={r['lrs']}\ndo={r['dropouts']}\nwd={r['weight_decays']}")
        labels.append(lbl)
        au_scores.append(r["au_combined"])
        bar_colors.append("tab:green" if r is best else "tab:purple")

    bars = ax.bar(range(len(au_scores)), au_scores, color=bar_colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("Alignment + Uniformity (lower = better)", fontsize=11)
    ax.set_title("Combined A+U Score — Simple Model Configs", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, au_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(str(plots_dir / "combined_score_bars.png"), dpi=150)
    plt.close(fig)

    # ---- 3. Heatmap: hidden_dim × lr colored by A+U score ----
    hidden_dims = sorted(set(r["hidden_dims"] for r in results))
    lrs = sorted(set(r["lrs"] for r in results))

    grid = np.full((len(hidden_dims), len(lrs)), np.nan)
    for r in results:
        hi = hidden_dims.index(r["hidden_dims"])
        li = lrs.index(r["lrs"])
        val = r["au_combined"]
        if np.isnan(grid[hi, li]) or val < grid[hi, li]:
            grid[hi, li] = val

    fig, ax = plt.subplots(figsize=(max(6, len(lrs) * 1.2), max(4, len(hidden_dims) * 0.8)))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], fontsize=9)
    ax.set_yticks(range(len(hidden_dims)))
    ax.set_yticklabels(hidden_dims, fontsize=9)
    ax.set_xlabel("Learning Rate", fontsize=11)
    ax.set_ylabel("Hidden Dim", fontsize=11)
    ax.set_title("A+U Score Heatmap — Simple Model", fontsize=13)

    for i in range(len(hidden_dims)):
        for j in range(len(lrs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if grid[i, j] > np.nanmedian(grid) else "black")

    fig.colorbar(im, ax=ax, pad=0.02, label="Align + Uniform (lower = better)")
    plt.tight_layout()
    fig.savefig(str(plots_dir / "heatmap_simple.png"), dpi=150)
    plt.close(fig)

    # ---- 4. Regularization impact ----
    dropouts = sorted(set(r["dropouts"] for r in results))
    wds = sorted(set(r["weight_decays"] for r in results))

    if len(dropouts) > 1 or len(wds) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for do in dropouts:
            runs = [r for r in results if r["dropouts"] == do]
            if runs:
                au_vals = [r["au_combined"] for r in runs]
                ax.scatter([do] * len(au_vals), au_vals, marker="^",
                           s=60, alpha=0.7)
        for do in dropouts:
            runs = [r for r in results if r["dropouts"] == do]
            if runs:
                mean_au = np.mean([r["au_combined"] for r in runs])
                ax.plot(do, mean_au, "D", color="red", markersize=10, zorder=5)
        ax.set_xlabel("Dropout", fontsize=11)
        ax.set_ylabel("A+U Score (lower = better)", fontsize=11)
        ax.set_title("Dropout Impact — Simple Model", fontsize=12)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for wd in wds:
            runs = [r for r in results if r["weight_decays"] == wd]
            if runs:
                au_vals = [r["au_combined"] for r in runs]
                ax.scatter([wd] * len(au_vals), au_vals, marker="^",
                           s=60, alpha=0.7)
        for wd in wds:
            runs = [r for r in results if r["weight_decays"] == wd]
            if runs:
                mean_au = np.mean([r["au_combined"] for r in runs])
                ax.plot(wd, mean_au, "D", color="red", markersize=10, zorder=5)
        ax.set_xlabel("Weight Decay", fontsize=11)
        ax.set_ylabel("A+U Score (lower = better)", fontsize=11)
        ax.set_title("Weight Decay Impact — Simple Model", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(plots_dir / "regularization_impact.png"), dpi=150)
        plt.close(fig)

    # ---- 5. Training curves for top-5 configs ----
    top_k = sorted(results, key=lambda r: r["au_combined"])[:5]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, r in enumerate(top_k):
        lbl = f"#{i+1} H={r['hidden_dims']} O={r['output_dims']} lr={r['lrs']} do={r['dropouts']} wd={r['weight_decays']}"
        ax.plot(r["train_losses"], label=lbl, linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Curves — Top 5 Simple Configs")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, r in enumerate(top_k):
        lbl = f"#{i+1} H={r['hidden_dims']} O={r['output_dims']} lr={r['lrs']} do={r['dropouts']} wd={r['weight_decays']}"
        ax.plot(r["val_losses"], label=lbl, linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Curves — Top 5 Simple Configs")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(plots_dir / "top5_training_curves.png"), dpi=150)
    plt.close(fig)

    # ---- 6. Output dim comparison ----
    output_dims = sorted(set(r["output_dims"] for r in results))
    if len(output_dims) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for od in output_dims:
            runs = [r for r in results if r["output_dims"] == od]
            au_vals = [r["au_combined"] for r in runs]
            ax.boxplot(au_vals, positions=[od], widths=4, patch_artist=True,
                       boxprops=dict(facecolor="lightblue", alpha=0.7))
        ax.set_xlabel("Output Dim", fontsize=11)
        ax.set_ylabel("A+U Score (lower = better)", fontsize=11)
        ax.set_title("Output Dim Impact — Simple Model", fontsize=13)
        ax.set_xticks(output_dims)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(str(plots_dir / "output_dim_comparison.png"), dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def build_search_grid(quick: bool = False) -> List[dict]:
    """
    Build the hyperparameter search grid for the simple model.

    Full grid: 3 lr × 3 hidden_dim × 2 output_dim × 2 dropout × 2 weight_decay × 2 temp = 144 configs
    Quick grid: 2 lr × 2 hidden_dim × 2 output_dim × 2 dropout × 1 weight_decay × 1 temp = 16 configs
    """
    if quick:
        return list_configs(
            lrs=[5e-4, 1e-3],
            hidden_dims=[48, 64],
            output_dims=[24, 32],
            dropouts=[0.1, 0.2],
            weight_decays=[1e-4],
            temperatures=[0.1],
            lambda_rels=[0.5],
        )
    else:
        return list_configs(
            lrs=[3e-4, 5e-4, 1e-3],
            hidden_dims=[32, 48, 64],
            output_dims=[24, 32],
            dropouts=[0.1, 0.2, 0.3],
            weight_decays=[1e-4, 5e-4],
            temperatures=[0.07, 0.1],
            lambda_rels=[0.5],
        )


def list_configs(models=None, **kwargs) -> List[dict]:
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg["model"] = "simple"
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Simple SG-GNN Encoder")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to JSON scene graph directory")
    parser.add_argument("--out_dir", type=str, default="sweeps_simple", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per config (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Use smaller search grid (16 configs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_dir) / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ---- Load data ----
    json_paths = load_all_json_paths(args.data_dir)
    paths = list(json_paths)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    n_train = max(1, int(len(paths) * 0.8))
    train_paths, val_paths = paths[:n_train], paths[n_train:]
    if not val_paths:
        val_paths = train_paths.copy()

    train_dataset = SimpleSceneGraphDataset(args.data_dir, json_paths=train_paths)
    val_dataset = SimpleSceneGraphDataset(args.data_dir, json_paths=val_paths)
    full_dataset = SimpleSceneGraphDataset(args.data_dir)

    print(f"[sweep-simple] Device: {device}")
    print(f"[sweep-simple] Node features: {SIMPLE_NODE_FEAT_DIM}-dim (center + extent)")
    print(f"[sweep-simple] Edge features: {SIMPLE_EDGE_FEAT_DIM}-dim (relation one-hot)")
    print(f"[sweep-simple] Data: {len(train_dataset)} train, {len(val_dataset)} val, {len(full_dataset)} total")
    print(f"[sweep-simple] Output: {sweep_dir}")

    # ---- Build grid ----
    search_grid = build_search_grid(quick=args.quick)
    n_configs = len(search_grid)
    print(f"[sweep-simple] Grid size: {n_configs} configurations")
    print(f"[sweep-simple] Epochs per config: {args.epochs}")
    print(f"[sweep-simple] Estimated total epochs: {n_configs * args.epochs}")

    # ---- CSV ----
    csv_path = sweep_dir / "sweep_results.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "rank", "hidden_dim", "output_dim", "lr", "dropout", "weight_decay",
        "temperature", "lambda_rel",
        "n_params", "best_val_loss", "alignment", "uniformity", "au_combined",
        "final_epoch",
    ])

    # ---- Run sweep ----
    results = []
    t_start = time.time()

    for i, config in enumerate(search_grid):
        config["epochs"] = args.epochs

        print(f"\n{'─'*60}")
        print(f"  Config {i+1}/{n_configs}: "
              f"H={config['hidden_dims']} | O={config['output_dims']} | "
              f"lr={config['lrs']} | do={config['dropouts']} | "
              f"wd={config['weight_decays']} | T={config['temperatures']}")
        print(f"{'─'*60}")

        t_cfg = time.time()

        result = train_config(
            config, train_dataset, val_dataset, full_dataset, device, args.seed
        )
        results.append(result)

        elapsed = time.time() - t_cfg
        print(f"  -> val_loss={result['best_val_loss']:.4f} | "
              f"align={result['alignment']:.4f} | "
              f"uniform={result['uniformity']:.4f} | "
              f"A+U={result['au_combined']:.4f} | "
              f"epochs={result['final_epoch']} | {elapsed:.1f}s")

    csv_file.close()

    # ---- Rank by A+U combined score ----
    results.sort(key=lambda r: r["au_combined"])

    # Re-write CSV sorted
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "hidden_dim", "output_dim", "lr", "dropout", "weight_decay",
            "temperature", "lambda_rel",
            "n_params", "best_val_loss", "alignment", "uniformity", "au_combined",
            "final_epoch",
        ])
        for rank, r in enumerate(results, 1):
            writer.writerow([
                rank, r["hidden_dims"], r["output_dims"], r["lrs"], r["dropouts"],
                r["weight_decays"], r["temperatures"], r["lambda_rels"],
                r["n_params"], f"{r['best_val_loss']:.6f}",
                f"{r['alignment']:.6f}", f"{r['uniformity']:.6f}",
                f"{r['au_combined']:.6f}", r["final_epoch"],
            ])

    # ---- Save full results JSON ----
    json_results = [{k: v for k, v in r.items() if k not in ("train_losses", "val_losses")}
                    for r in results]
    with open(sweep_dir / "sweep_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # ---- Print ranking ----
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {n_configs} configs in {total_time:.1f}s")
    print(f"{'='*70}\n")

    print(f"{'Rank':<5} {'H':<5} {'O':<4} {'LR':<10} {'DO':<6} {'WD':<10} "
          f"{'Temp':<6} {'Align':>8} {'Uniform':>9} {'A+U':>8} {'ValLoss':>9}")
    print("─" * 90)
    for rank, r in enumerate(results, 1):
        marker = " *" if rank == 1 else ""
        print(f"{rank:<5} {r['hidden_dims']:<5} {r['output_dims']:<4} {r['lrs']:<10.0e} "
              f"{r['dropouts']:<6.2f} {r['weight_decays']:<10.0e} "
              f"{r['temperatures']:<6.2f} "
              f"{r['alignment']:>8.4f} {r['uniformity']:>9.4f} "
              f"{r['au_combined']:>8.4f} {r['best_val_loss']:>9.4f}{marker}")

    best = results[0]
    print(f"\n* Best config:")
    print(f"  Hidden dim:   {best['hidden_dims']}")
    print(f"  Output dim:   {best['output_dims']}")
    print(f"  LR:           {best['lrs']}")
    print(f"  Dropout:      {best['dropouts']}")
    print(f"  Weight decay: {best['weight_decays']}")
    print(f"  Temperature:  {best['temperatures']}")
    print(f"  Lambda rel:   {best['lambda_rels']}")
    print(f"  Params:       {best['n_params']:,}")
    print(f"  Alignment:    {best['alignment']:.6f}  (lower = better)")
    print(f"  Uniformity:   {best['uniformity']:.6f}  (lower = better)")
    print(f"  A+U:          {best['au_combined']:.6f}")

    # ---- Visualizations ----
    print("\n[viz] Generating plots...")
    plot_sweep_results(results, str(sweep_dir))

    print(f"\n[done] All outputs saved to {sweep_dir}/")
    print(f"  ├── sweep_results.csv")
    print(f"  ├── sweep_results.json")
    print(f"  └── plots/")
    print(f"       ├── alignment_vs_uniformity.png")
    print(f"       ├── combined_score_bars.png")
    print(f"       ├── heatmap_simple.png")
    print(f"       ├── regularization_impact.png")
    print(f"       ├── top5_training_curves.png")
    print(f"       └── output_dim_comparison.png")

    # Print recommended train command
    print(f"\n[recommend] Train the best config with full epochs:")
    print(f"  python train.py --data_dir {args.data_dir} --out_dir runs "
          f"--model simple --hidden_dim {best['hidden_dims']} "
          f"--output_dim {best['output_dims']} "
          f"--lr {best['lrs']} --dropout {best['dropouts']} "
          f"--weight_decay {best['weight_decays']} "
          f"--temperature {best['temperatures']} "
          f"--epochs 200 --seed {args.seed}")


if __name__ == "__main__":
    main()
