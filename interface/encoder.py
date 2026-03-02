"""
Unified inference interface for all Scene Graph Encoder variants.

Loads a trained checkpoint + config and exposes:
  - encode_scene(json_path)         → embedding tensor
  - encode_scene_from_edges(edges)  → embedding tensor
  - encode_batch(json_paths)        → stacked embeddings
  - compare(path_a, path_b)         → cosine similarity
  - pairwise_similarity(paths)      → similarity matrix

Supported model types: 3layer, 2layer, simple, simple3layer
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

import sys

# Allow imports from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset import (
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    SIMPLE_NODE_FEAT_DIM,
    SIMPLE_EDGE_FEAT_DIM,
    parse_scene_graph,
    parse_scene_graph_simple,
)
from model import (
    SceneGraphEncoder,
    SceneGraphEncoderLight,
    SceneGraphEncoderSimple,
    SceneGraphEncoderSimple3Layer,
)

# Mapping from model type string → (ModelClass, node_feat_dim, edge_feat_dim, parser_fn)
MODEL_REGISTRY: Dict[str, tuple] = {
    "3layer": (SceneGraphEncoder, NODE_FEAT_DIM, EDGE_FEAT_DIM, parse_scene_graph),
    "2layer": (SceneGraphEncoderLight, NODE_FEAT_DIM, EDGE_FEAT_DIM, parse_scene_graph),
    "simple": (SceneGraphEncoderSimple, SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM, parse_scene_graph_simple),
    "simple3layer": (SceneGraphEncoderSimple3Layer, SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM, parse_scene_graph_simple),
}


class SceneGraphEncoderInterface:
    """
    High-level inference interface for any trained Scene Graph Encoder.

    Example:
        >>> iface = SceneGraphEncoderInterface.from_run_dir("runs/20260302_145714")
        >>> emb = iface.encode_scene("Data/franka_cabinet.json")
        >>> print(emb.shape)  # [1, 32]

        >>> sim = iface.compare("Data/franka_cabinet.json", "Data/franka_cabinet_2.json")
        >>> print(f"Cosine similarity: {sim:.4f}")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        device: torch.device,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device
        _, _, _, self._parse_fn = MODEL_REGISTRY[model_type]
        self.model.eval()

    # ──────────────────────────────────────────────────────────────────
    #  Constructors
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str,
        device: Optional[torch.device] = None,
    ) -> "SceneGraphEncoderInterface":
        """
        Load a trained model from a run directory containing
        config.json and best_checkpoint.pt.

        Args:
            run_dir: Path to the run directory.
            device: Compute device. Auto-detected if None.

        Returns:
            Initialized SceneGraphEncoderInterface.
        """
        run_path = Path(run_dir)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        config_path = run_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {run_dir}")
        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model", "3layer")
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Supported: {list(MODEL_REGISTRY.keys())}"
            )

        # Build model
        ModelClass, nfd, efd, _ = MODEL_REGISTRY[model_type]
        model = ModelClass(
            node_feat_dim=nfd,
            edge_feat_dim=efd,
            hidden_dim=config.get("hidden_dim", 128),
            output_dim=config.get("output_dim", 32),
            dropout=config.get("dropout", 0.15),
        ).to(device)

        # Load checkpoint
        ckpt_path = run_path / "best_checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"best_checkpoint.pt not found in {run_dir}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model=model, model_type=model_type, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str = "3layer",
        hidden_dim: int = 128,
        output_dim: int = 32,
        dropout: float = 0.15,
        device: Optional[torch.device] = None,
    ) -> "SceneGraphEncoderInterface":
        """
        Load a model directly from a checkpoint file (without config.json).

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            model_type: One of '3layer', '2layer', 'simple', 'simple3layer'.
            hidden_dim: Hidden dimension used during training.
            output_dim: Output embedding dimension used during training.
            dropout: Dropout used during training.
            device: Compute device. Auto-detected if None.

        Returns:
            Initialized SceneGraphEncoderInterface.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ModelClass, nfd, efd, _ = MODEL_REGISTRY[model_type]
        model = ModelClass(
            node_feat_dim=nfd,
            edge_feat_dim=efd,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model=model, model_type=model_type, device=device)

    # ──────────────────────────────────────────────────────────────────
    #  Core encoding methods
    # ──────────────────────────────────────────────────────────────────

    def _parse_json(self, json_path: str) -> Data:
        """Load a JSON file and parse it into a PyG Data object."""
        with open(json_path) as f:
            edges = json.load(f)
        return self._parse_fn(edges, scene_id=Path(json_path).stem)

    @torch.no_grad()
    def encode_scene(self, json_path: str) -> torch.Tensor:
        """
        Encode a single scene graph from a JSON file.

        Args:
            json_path: Path to a scene graph JSON file.

        Returns:
            Embedding tensor of shape [1, output_dim].
        """
        data = self._parse_json(json_path).to(self.device)
        return self.model.get_scene_embedding(data)

    @torch.no_grad()
    def encode_scene_from_edges(self, edges_list: list, scene_id: str = "scene") -> torch.Tensor:
        """
        Encode a scene graph from an in-memory edge list (same format as JSON content).

        Args:
            edges_list: List of edge dictionaries.
            scene_id: Optional scene identifier.

        Returns:
            Embedding tensor of shape [1, output_dim].
        """
        data = self._parse_fn(edges_list, scene_id=scene_id).to(self.device)
        return self.model.get_scene_embedding(data)

    @torch.no_grad()
    def encode_batch(self, json_paths: List[str]) -> torch.Tensor:
        """
        Encode multiple scene graphs in a single batched forward pass.

        Args:
            json_paths: List of paths to JSON scene graph files.

        Returns:
            Embedding tensor of shape [N, output_dim].
        """
        data_list = [self._parse_json(p) for p in json_paths]
        batch = Batch.from_data_list(data_list).to(self.device)
        out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return out["embedding"]

    # ──────────────────────────────────────────────────────────────────
    #  Similarity methods
    # ──────────────────────────────────────────────────────────────────

    def compare(self, path_a: str, path_b: str) -> float:
        """
        Compute cosine similarity between two scene graphs.

        Args:
            path_a: Path to first scene graph JSON.
            path_b: Path to second scene graph JSON.

        Returns:
            Cosine similarity in [-1, 1]. Higher = more similar.
        """
        embs = self.encode_batch([path_a, path_b])
        sim = F.cosine_similarity(embs[0:1], embs[1:2]).item()
        return sim

    def pairwise_similarity(self, json_paths: List[str]) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix for a list of scenes.

        Args:
            json_paths: List of paths to JSON scene graph files.

        Returns:
            Similarity matrix of shape [N, N] with values in [-1, 1].
        """
        embs = self.encode_batch(json_paths)  # [N, D]
        embs_norm = F.normalize(embs, dim=1)
        return embs_norm @ embs_norm.T

    # ──────────────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_full_output(self, json_path: str) -> Dict[str, torch.Tensor]:
        """
        Get all model outputs (embedding, node_embeddings, edge_repr).

        Args:
            json_path: Path to a scene graph JSON file.

        Returns:
            Dict with 'embedding', 'node_embeddings', 'edge_repr'.
        """
        data = self._parse_json(json_path).to(self.device)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        return self.model(data.x, data.edge_index, data.edge_attr, batch)

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension."""
        return self.model.output_dim

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> str:
        """Return a human-readable summary string."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        _, nfd, efd, _ = MODEL_REGISTRY[self.model_type]
        return (
            f"SceneGraphEncoderInterface\n"
            f"  Model type:    {self.model_type}\n"
            f"  Model class:   {self.model.__class__.__name__}\n"
            f"  Node feat dim: {nfd}\n"
            f"  Edge feat dim: {efd}\n"
            f"  Hidden dim:    {self.model.hidden_dim}\n"
            f"  Output dim:    {self.model.output_dim}\n"
            f"  Parameters:    {self.num_parameters:,} ({trainable:,} trainable)\n"
            f"  Device:        {self.device}"
        )

    def __repr__(self) -> str:
        return (
            f"SceneGraphEncoderInterface("
            f"model_type='{self.model_type}', "
            f"hidden_dim={self.model.hidden_dim}, "
            f"output_dim={self.model.output_dim}, "
            f"device={self.device})"
        )
