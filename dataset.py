"""
Scene Graph Dataset Loader.

Loads scene graph JSON files from a directory and converts them into
PyTorch Geometric Data objects for GNN training.
"""

import json
import math
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


# Node feature dimension: center(3) + extent(3) + log_extent(3) + volume(1) = 10
NODE_FEAT_DIM = 10
# Edge feature dimension: relation_vector(6) + delta_center(3) + dist(1) + extent_ratio(3) = 13
EDGE_FEAT_DIM = 13

# Small epsilon for safe log / division
EPS = 1e-8


def load_all_json_paths(data_dir: str) -> List[Path]:
    """
    Discover and sort all .json files in a directory.

    Args:
        data_dir: Path to the directory containing scene graph JSON files.

    Returns:
        Sorted list of Path objects for reproducibility.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    json_paths = sorted(data_path.glob("*.json"))
    if len(json_paths) == 0:
        warnings.warn(f"No JSON files found in {data_dir}")
    return json_paths


def build_node_features(center: List[float], extent: List[float]) -> torch.Tensor:
    """
    Construct a 10-dim node feature vector.

    Features: [center(3), extent(3), log(extent+eps)(3), volume(1)]

    Args:
        center: [x, y, z] center coordinates.
        extent: [ex, ey, ez] extent (bounding box half-sizes).

    Returns:
        Tensor of shape [10].
    """
    center_t = torch.tensor(center, dtype=torch.float32)
    extent_t = torch.tensor(extent, dtype=torch.float32)
    log_extent = torch.log(extent_t.abs() + EPS)
    volume = torch.prod(extent_t.abs() + EPS).unsqueeze(0)
    return torch.cat([center_t, extent_t, log_extent, volume])


def build_edge_features(
    source_center: List[float],
    source_extent: List[float],
    target_center: List[float],
    target_extent: List[float],
    relation_vector: List[float],
) -> torch.Tensor:
    """
    Construct a 13-dim edge feature vector.

    Features: [relation_onehot(6), delta_center(3), dist(1), extent_ratio(3)]

    Args:
        source_center: Source node center [x, y, z].
        source_extent: Source node extent [ex, ey, ez].
        target_center: Target node center [x, y, z].
        target_extent: Target node extent [ex, ey, ez].
        relation_vector: One-hot relation vector of length 6.

    Returns:
        Tensor of shape [13].
    """
    relation_t = torch.tensor(relation_vector, dtype=torch.float32)

    src_c = torch.tensor(source_center, dtype=torch.float32)
    tgt_c = torch.tensor(target_center, dtype=torch.float32)
    delta_center = tgt_c - src_c  # (3,)
    dist = torch.norm(delta_center).unsqueeze(0)  # (1,)

    src_e = torch.tensor(source_extent, dtype=torch.float32).abs() + EPS
    tgt_e = torch.tensor(target_extent, dtype=torch.float32).abs() + EPS
    extent_ratio = tgt_e / src_e  # (3,)

    return torch.cat([relation_t, delta_center, dist, extent_ratio])


def parse_scene_graph(edges_list: list, scene_id: str) -> Data:
    """
    Parse a list of edge dicts into a PyG Data object.

    Args:
        edges_list: List of edge dictionaries from JSON.
        scene_id: Identifier string for the scene (filename stem).

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr, scene_id.

    Raises:
        ValueError: If the edge list is empty or malformed.
    """
    if not edges_list:
        raise ValueError(f"Empty edge list for scene {scene_id}")

    # --- Collect unique nodes ---
    node_info = {}  # node_id -> (center, extent)
    for edge in edges_list:
        src = edge["source"]
        tgt = edge["target"]
        sid = src["source_id"]
        tid = tgt["target_id"]
        if sid not in node_info:
            node_info[sid] = (src["source_center"], src["source_extent"])
        if tid not in node_info:
            node_info[tid] = (tgt["target_center"], tgt["target_extent"])

    # Deterministic ordering
    sorted_node_ids = sorted(node_info.keys(), key=lambda x: (isinstance(x, str), x))
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    num_nodes = len(sorted_node_ids)

    # --- Build node feature matrix ---
    node_features = []
    for nid in sorted_node_ids:
        center, extent = node_info[nid]
        node_features.append(build_node_features(center, extent))
    x = torch.stack(node_features)  # [num_nodes, 10]

    # --- Build edge index and edge attributes ---
    src_indices = []
    tgt_indices = []
    edge_attrs = []
    for edge in edges_list:
        src = edge["source"]
        tgt = edge["target"]
        sid = src["source_id"]
        tid = tgt["target_id"]

        src_indices.append(node_id_to_idx[sid])
        tgt_indices.append(node_id_to_idx[tid])

        edge_feat = build_edge_features(
            source_center=src["source_center"],
            source_extent=src["source_extent"],
            target_center=tgt["target_center"],
            target_extent=tgt["target_extent"],
            relation_vector=edge["relation_vector"],
        )
        edge_attrs.append(edge_feat)

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)  # [2, E]
    edge_attr = torch.stack(edge_attrs)  # [E, 13]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.scene_id = scene_id
    data.num_nodes = num_nodes
    return data


class SceneGraphDataset(Dataset):
    """
    PyTorch Dataset that loads scene graph JSON files from a directory.

    Each JSON file is one scene graph. __getitem__ returns a PyG Data object.
    """

    def __init__(self, data_dir: str, json_paths: Optional[List[Path]] = None):
        """
        Args:
            data_dir: Directory containing .json scene graph files.
            json_paths: Optional pre-filtered list of paths (for train/val splits).
        """
        super().__init__()
        if json_paths is not None:
            self.json_paths = sorted(json_paths)
        else:
            self.json_paths = load_all_json_paths(data_dir)

        # Pre-load all graphs into memory for speed
        self.graphs: List[Data] = []
        for jp in self.json_paths:
            try:
                with open(jp, "r") as f:
                    edges = json.load(f)
                graph = parse_scene_graph(edges, scene_id=jp.stem)
                self.graphs.append(graph)
            except Exception as e:
                warnings.warn(f"Skipping broken JSON {jp.name}: {e}")

        if len(self.graphs) == 0:
            warnings.warn(f"No valid graphs loaded from {data_dir}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
