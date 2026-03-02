"""
Visualization utilities for training metrics and embedding analysis.

Produces:
  - Loss curves (train/val contrastive + relation)
  - PCA scatter plot of scene embeddings
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader

from model import SceneGraphEncoder


def plot_loss_curves(
    metrics: Dict[str, List[float]],
    out_dir: str,
    filename: str = "loss_curves.png",
) -> str:
    """
    Plot train/val loss curves and save to disk.

    Args:
        metrics: Dict with keys like 'train_contrast', 'val_contrast',
                 'train_relation', 'val_relation', 'train_total', 'val_total'.
                 Values are lists over epochs.
        out_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to the saved plot.
    """
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(metrics.get("train_total", [])) + 1)
    if len(epochs) == 0:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Contrastive loss ---
    ax = axes[0]
    if "train_contrast" in metrics:
        ax.plot(epochs, metrics["train_contrast"], label="Train Contrastive", linewidth=1.5)
    if "val_contrast" in metrics:
        ax.plot(epochs, metrics["val_contrast"], label="Val Contrastive", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Contrastive Loss (InfoNCE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Relation loss / Total loss ---
    ax = axes[1]
    if "train_total" in metrics:
        ax.plot(epochs, metrics["train_total"], label="Train Total", linewidth=1.5)
    if "val_total" in metrics:
        ax.plot(epochs, metrics["val_total"], label="Val Total", linewidth=1.5, linestyle="--")
    if "train_relation" in metrics and any(v > 0 for v in metrics["train_relation"]):
        ax.plot(epochs, metrics["train_relation"], label="Train Relation", linewidth=1.0, alpha=0.7)
    if "val_relation" in metrics and any(v > 0 for v in metrics["val_relation"]):
        ax.plot(epochs, metrics["val_relation"], label="Val Relation", linewidth=1.0, alpha=0.7, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total / Relation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(plots_dir / filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Loss curves saved to {save_path}")
    return save_path


def plot_overfitting_gap(
    metrics: Dict[str, List[float]],
    out_dir: str,
    filename: str = "overfitting_gap.png",
) -> str:
    """
    Plot the train-val gap to visualize overfitting.

    Args:
        metrics: Dict with 'train_total' and 'val_total' lists.
        out_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to the saved plot.
    """
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_loss = metrics.get("train_total", [])
    val_loss = metrics.get("val_total", [])
    if not train_loss or not val_loss:
        return ""

    epochs = range(1, len(train_loss) + 1)
    gap = [v - t for t, v in zip(train_loss, val_loss)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, gap, label="Val - Train (gap)", color="red", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.fill_between(epochs, 0, gap, alpha=0.15, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Gap (Val − Train)")
    ax.set_title("Overfitting Detection: Train-Val Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(plots_dir / filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Overfitting gap plot saved to {save_path}")
    return save_path


@torch.no_grad()
def plot_embedding_pca(
    model: SceneGraphEncoder,
    dataset,
    out_dir: str,
    device: torch.device,
    filename: str = "embedding_pca.png",
) -> str:
    """
    Compute scene embeddings for all graphs and plot a 2D PCA scatter.

    Args:
        model: Trained SceneGraphEncoder.
        dataset: SceneGraphDataset (all scenes).
        out_dir: Directory to save the plot.
        device: torch.device for inference.
        filename: Output filename.

    Returns:
        Path to the saved plot.
    """
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    embeddings = []
    scene_ids = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
        data = data.to(device)
        result = model(data.x, data.edge_index, data.edge_attr, data.batch)
        emb = result["embedding"].cpu().numpy()
        embeddings.append(emb[0])
        scene_ids.append(data.scene_id[0] if isinstance(data.scene_id, list) else data.scene_id)

    if len(embeddings) < 2:
        print("[viz] Not enough embeddings for PCA (need >= 2)")
        return ""

    embeddings = np.array(embeddings)

    # PCA to 2D
    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    emb_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1] if n_components >= 2 else np.zeros(len(emb_2d)),
        c=range(len(scene_ids)),
        cmap="tab20",
        s=80,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Annotate with scene IDs
    for i, sid in enumerate(scene_ids):
        label = sid if len(str(sid)) < 25 else str(sid)[:22] + "..."
        ax.annotate(
            label,
            (emb_2d[i, 0], emb_2d[i, 1] if n_components >= 2 else 0),
            fontsize=7,
            alpha=0.75,
            xytext=(5, 5),
            textcoords="offset points",
        )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
    if n_components >= 2:
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    ax.set_title("Scene Embeddings — PCA Projection (32D → 2D)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(plots_dir / filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] PCA embedding plot saved to {save_path}")
    return save_path
