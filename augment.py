"""
Graph augmentation strategies for contrastive learning.

Provides stochastic augmentations on PyG Data objects to create
two different views of the same scene graph for SimCLR-style training.
"""

import copy
import random

import torch
from torch_geometric.data import Data


def augment_graph(
    data: Data,
    node_drop_prob: float = 0.1,
    edge_drop_prob: float = 0.15,
    feat_noise_std: float = 0.05,
    feat_mask_prob: float = 0.1,
) -> Data:
    """
    Apply stochastic augmentations to a scene graph.

    Augmentations applied (in order):
      1. Node feature masking  — randomly zero out some node feature dims.
      2. Node feature noise    — add Gaussian noise to node features.
      3. Edge dropping         — randomly remove edges.
      4. Node dropping         — randomly remove nodes (and their edges).
         Skipped if graph would become < 2 nodes.

    Args:
        data: Input PyG Data object (will NOT be modified in-place).
        node_drop_prob: Probability of dropping each node.
        edge_drop_prob: Probability of dropping each edge.
        feat_noise_std: Std-dev of Gaussian noise added to node features.
        feat_mask_prob: Probability of masking each feature dimension per node.

    Returns:
        Augmented PyG Data (new object, original unchanged).
    """
    # Deep copy to avoid mutating original
    aug = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
    )
    if hasattr(data, "scene_id"):
        aug.scene_id = data.scene_id

    num_nodes = aug.x.size(0)
    num_edges = aug.edge_index.size(1)

    # --- 1. Node feature masking ---
    if feat_mask_prob > 0:
        mask = torch.bernoulli(
            torch.full(aug.x.shape, 1.0 - feat_mask_prob)
        ).to(aug.x.device)
        aug.x = aug.x * mask

    # --- 2. Node feature noise ---
    if feat_noise_std > 0:
        noise = torch.randn_like(aug.x) * feat_noise_std
        aug.x = aug.x + noise

    # --- 3. Edge dropping ---
    if edge_drop_prob > 0 and num_edges > 0:
        edge_keep_mask = torch.rand(num_edges) > edge_drop_prob
        # Ensure at least 1 edge survives
        if edge_keep_mask.sum() == 0:
            edge_keep_mask[random.randint(0, num_edges - 1)] = True
        aug.edge_index = aug.edge_index[:, edge_keep_mask]
        if aug.edge_attr is not None:
            aug.edge_attr = aug.edge_attr[edge_keep_mask]

    # --- 4. Node dropping (only if graph stays >= 2 nodes) ---
    if node_drop_prob > 0 and num_nodes > 2:
        node_keep_mask = torch.rand(num_nodes) > node_drop_prob
        # Ensure at least 2 nodes survive
        if node_keep_mask.sum() < 2:
            keep_ids = random.sample(range(num_nodes), 2)
            node_keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_keep_mask[keep_ids] = True

        kept_indices = torch.where(node_keep_mask)[0]
        old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
        old_to_new[kept_indices] = torch.arange(kept_indices.size(0))

        # Filter node features
        aug.x = aug.x[kept_indices]

        # Remap and filter edges
        if aug.edge_index.size(1) > 0:
            src, tgt = aug.edge_index
            # Keep only edges where both endpoints survived
            valid_edges = node_keep_mask[src] & node_keep_mask[tgt]
            if valid_edges.sum() == 0:
                # Fallback: keep a self-loop on first kept node to avoid empty graph
                aug.edge_index = torch.zeros((2, 1), dtype=torch.long)
                if aug.edge_attr is not None:
                    aug.edge_attr = aug.edge_attr[:1] * 0  # zero edge attr
            else:
                aug.edge_index = torch.stack(
                    [old_to_new[src[valid_edges]], old_to_new[tgt[valid_edges]]]
                )
                if aug.edge_attr is not None:
                    aug.edge_attr = aug.edge_attr[valid_edges]

    aug.num_nodes = aug.x.size(0)
    return aug
