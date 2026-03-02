"""
GATv2-based Scene Graph Encoder.

Architecture:
  1. Node encoder MLP:  node_feat_dim → 256 → hidden_dim (128)
  2. 3x GATv2Conv layers with edge conditioning, residual connections
  3. Attention-based graph pooling → graph-level vector
  4. Scene encoder MLP:  pool_dim → 64 → 32

Output: 32-dim scene embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Data, Batch

from dataset import NODE_FEAT_DIM, EDGE_FEAT_DIM, SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM


class SceneGraphEncoder(nn.Module):
    """
    GATv2-based encoder that maps a variable-size scene graph to a
    fixed 32-dimensional scene embedding vector.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 128,
        output_dim: int = 32,
        dropout: float = 0.15,
    ):
        """
        Args:
            node_feat_dim: Input node feature dimension (default 10).
            edge_feat_dim: Input edge feature dimension (default 13).
            hidden_dim: Hidden dimension for GATv2 layers (default 128).
            output_dim: Final scene embedding dimension (default 32).
            dropout: Dropout probability between GATv2 layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # ---- 1. Node encoder MLP ----
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),  # -> 128
        )

        # ---- 2. Edge encoder (project edge attrs for relation loss) ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),  # -> 128  (for relation loss head)
        )

        # ---- 3. GATv2 layers ----
        head_dim = hidden_dim // 2  # per-head output dimension
        intermediate_dim = head_dim * 4  # after concat of 4 heads

        # Layer 1: in=hidden_dim, out=head_dim, heads=4, concat -> head_dim*4
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(intermediate_dim)

        # Layer 2: in=intermediate_dim, out=head_dim, heads=4, concat -> intermediate_dim
        self.gat2 = GATv2Conv(
            in_channels=intermediate_dim,
            out_channels=head_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(intermediate_dim)

        # Residual projection for skip connection (hidden_dim -> intermediate_dim)
        self.res_proj = nn.Linear(hidden_dim, intermediate_dim)

        # Layer 3: in=intermediate_dim, out=head_dim, heads=2, concat -> hidden_dim
        self.gat3 = GATv2Conv(
            in_channels=intermediate_dim,
            out_channels=head_dim,
            heads=2,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Residual projection for skip connection (intermediate_dim -> hidden_dim)
        self.res_proj2 = nn.Linear(intermediate_dim, hidden_dim)

        # ---- 4. Attention pooling ----
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        # ---- 5. Scene encoder MLP → 32-dim output ----
        self.scene_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw node features through the node MLP."""
        return self.node_encoder(x)

    def encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode raw edge features through the edge MLP (for relation loss)."""
        return self.edge_encoder(edge_attr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Node features [N_total, node_feat_dim].
            edge_index: Edge index [2, E_total].
            edge_attr: Edge attributes [E_total, edge_feat_dim].
            batch: Batch vector mapping nodes to graphs [N_total].

        Returns:
            Dict with:
              - 'embedding': Scene embeddings [B, 32].
              - 'node_embeddings': Per-node embeddings after GATv2 [N_total, 128].
              - 'edge_repr': Edge representations [E_total, 128].
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ---- Node encoding ----
        h = self.encode_nodes(x)  # [N, 128]
        h_skip = h

        # ---- GATv2 Layer 1 ----
        h = self.gat1(h, edge_index, edge_attr=edge_attr)  # [N, 256]
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---- GATv2 Layer 2 (with residual from layer 1 output) ----
        h_res = h  # save for residual
        h = self.gat2(h, edge_index, edge_attr=edge_attr)  # [N, 256]
        h = self.norm2(h)
        h = h + h_res  # residual connection (256 == 256 ✓)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Also add residual from input (128 -> 256 via projection)
        h = h + self.res_proj(h_skip)

        # ---- GATv2 Layer 3 ----
        h_res2 = h  # [N, 256]
        h = self.gat3(h, edge_index, edge_attr=edge_attr)  # [N, 128]
        h = self.norm3(h)
        h = h + self.res_proj2(h_res2)  # residual (256 -> 128)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h  # [N, 128]

        # ---- Edge representations (for relation loss) ----
        edge_repr = self.encode_edges(edge_attr)  # [E, 128]

        # ---- Attention pooling ----
        g = self.pool(h, batch)  # [B, 128]

        # ---- Scene encoder -> 32-dim ----
        z = self.scene_encoder(g)  # [B, 32]

        return {
            "embedding": z,
            "node_embeddings": node_embeddings,
            "edge_repr": edge_repr,
        }

    def get_scene_embedding(self, data: Data) -> torch.Tensor:
        """
        Convenience method: get a 32-dim embedding from a single Data object.

        Args:
            data: Single PyG Data object.

        Returns:
            Scene embedding tensor [1, 32].
        """
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )
        result = self.forward(data.x, data.edge_index, data.edge_attr, batch)
        return result["embedding"]


class SceneGraphEncoderLight(nn.Module):
    """
    Lighter GATv2-based encoder with only 2 GATv2 layers.

    Architecture:
      1. Node encoder MLP:  node_feat_dim → 128 → hidden_dim
      2. 2x GATv2Conv layers with edge conditioning + 1 residual connection
      3. Attention-based graph pooling → graph-level vector
      4. Scene encoder MLP:  hidden_dim → 64 → output_dim

    Fewer parameters and faster training — better suited for small datasets.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # ---- 1. Node encoder MLP ----
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # ---- 2. Edge encoder (for relation loss) ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # ---- 3. GATv2 layers (2 layers) ----
        head_dim = hidden_dim // 2
        intermediate_dim = head_dim * 4  # after concat of 4 heads

        # Layer 1: hidden_dim → head_dim × 4 heads → intermediate_dim
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(intermediate_dim)

        # Layer 2: intermediate_dim → head_dim × 2 heads → hidden_dim
        self.gat2 = GATv2Conv(
            in_channels=intermediate_dim,
            out_channels=head_dim,
            heads=2,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Residual projection: hidden_dim → hidden_dim (skip from input)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)

        # ---- 4. Attention pooling ----
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        # ---- 5. Scene encoder MLP ----
        self.scene_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_encoder(x)

    def encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(edge_attr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ---- Node encoding ----
        h = self.encode_nodes(x)  # [N, H]
        h_skip = h

        # ---- GATv2 Layer 1 ----
        h = self.gat1(h, edge_index, edge_attr=edge_attr)  # [N, 2H]
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---- GATv2 Layer 2 ----
        h = self.gat2(h, edge_index, edge_attr=edge_attr)  # [N, H]
        h = self.norm2(h)
        h = h + self.res_proj(h_skip)  # residual from input
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h  # [N, H]

        # ---- Edge representations (for relation loss) ----
        edge_repr = self.encode_edges(edge_attr)  # [E, H]

        # ---- Attention pooling ----
        g = self.pool(h, batch)  # [B, H]

        # ---- Scene encoder ----
        z = self.scene_encoder(g)  # [B, D]

        return {
            "embedding": z,
            "node_embeddings": node_embeddings,
            "edge_repr": edge_repr,
        }

    def get_scene_embedding(self, data: Data) -> torch.Tensor:
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )
        result = self.forward(data.x, data.edge_index, data.edge_attr, batch)
        return result["embedding"]


class SceneGraphEncoderSimple3Layer(nn.Module):
    """
    3-layer GATv2 encoder with simplified 6-dim features.

    Input features:
      - Node: center(3) + extent(3) = 6
      - Edge: relation_onehot(6)

    Architecture:
      1. Node encoder MLP:  6 → 64 → hidden_dim
      2. Edge encoder MLP:  6 → 32 → hidden_dim  (for relation loss)
      3. 3x GATv2Conv with edge conditioning + residual connections
      4. Attention pooling → graph vector
      5. Scene encoder MLP: hidden_dim → 64 → output_dim
    """

    def __init__(
        self,
        node_feat_dim: int = SIMPLE_NODE_FEAT_DIM,
        edge_feat_dim: int = SIMPLE_EDGE_FEAT_DIM,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # ---- 1. Node encoder MLP ----
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # ---- 2. Edge encoder (for relation loss) ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )

        # ---- 3. GATv2 layers (3 layers) ----
        head_dim = hidden_dim // 2
        intermediate_dim = head_dim * 4  # after concat of 4 heads

        # Layer 1: hidden_dim → head_dim × 4 heads → intermediate_dim
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(intermediate_dim)

        # Layer 2: intermediate_dim → head_dim × 4 heads → intermediate_dim
        self.gat2 = GATv2Conv(
            in_channels=intermediate_dim,
            out_channels=head_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(intermediate_dim)

        # Residual projection: hidden_dim → intermediate_dim (skip from input)
        self.res_proj = nn.Linear(hidden_dim, intermediate_dim)

        # Layer 3: intermediate_dim → head_dim × 2 heads → hidden_dim
        self.gat3 = GATv2Conv(
            in_channels=intermediate_dim,
            out_channels=head_dim,
            heads=2,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Residual projection: intermediate_dim → hidden_dim
        self.res_proj2 = nn.Linear(intermediate_dim, hidden_dim)

        # ---- 4. Attention pooling ----
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        # ---- 5. Scene encoder MLP ----
        self.scene_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_encoder(x)

    def encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(edge_attr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ---- Node encoding ----
        h = self.encode_nodes(x)       # [N, H]
        h_skip = h

        # ---- GATv2 Layer 1 ----
        h = self.gat1(h, edge_index, edge_attr=edge_attr)  # [N, 2H]
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---- GATv2 Layer 2 (with residual from layer 1) ----
        h_res = h
        h = self.gat2(h, edge_index, edge_attr=edge_attr)  # [N, 2H]
        h = self.norm2(h)
        h = h + h_res  # residual (2H == 2H)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Add residual from input (H → 2H via projection)
        h = h + self.res_proj(h_skip)

        # ---- GATv2 Layer 3 ----
        h_res2 = h  # [N, 2H]
        h = self.gat3(h, edge_index, edge_attr=edge_attr)  # [N, H]
        h = self.norm3(h)
        h = h + self.res_proj2(h_res2)  # residual (2H → H)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h

        # ---- Edge representations ----
        edge_repr = self.encode_edges(edge_attr)

        # ---- Attention pooling ----
        g = self.pool(h, batch)

        # ---- Scene encoder ----
        z = self.scene_encoder(g)

        return {
            "embedding": z,
            "node_embeddings": node_embeddings,
            "edge_repr": edge_repr,
        }

    def get_scene_embedding(self, data: Data) -> torch.Tensor:
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )
        result = self.forward(data.x, data.edge_index, data.edge_attr, batch)
        return result["embedding"]


class SceneGraphEncoderSimple(nn.Module):
    """
    Minimal 2-layer GATv2 encoder with simplified features.

    Input features:
      - Node: center(3) + extent(3) = 6
      - Edge: relation_onehot(6)

    Architecture:
      1. Node encoder MLP:  6 → 64 → hidden_dim
      2. Edge encoder MLP:  6 → 32 → hidden_dim  (for relation loss)
      3. 2x GATv2Conv with edge conditioning + residual
      4. Attention pooling → graph vector
      5. Scene encoder MLP: hidden_dim → 64 → output_dim
    """

    def __init__(
        self,
        node_feat_dim: int = SIMPLE_NODE_FEAT_DIM,
        edge_feat_dim: int = SIMPLE_EDGE_FEAT_DIM,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # ---- 1. Node encoder MLP ----
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # ---- 2. Edge encoder (for relation loss) ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )

        # ---- 3. GATv2 layers (2 layers) ----
        head_dim = hidden_dim // 2
        intermediate_dim = head_dim * 4  # after concat of 4 heads

        # Layer 1: hidden_dim → head_dim × 4 heads → intermediate_dim
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(intermediate_dim)

        # Layer 2: intermediate_dim → head_dim × 2 heads → hidden_dim
        self.gat2 = GATv2Conv(
            in_channels=intermediate_dim,
            out_channels=head_dim,
            heads=2,
            concat=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Residual projection: hidden_dim → hidden_dim
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)

        # ---- 4. Attention pooling ----
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        # ---- 5. Scene encoder MLP ----
        self.scene_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_encoder(x)

    def encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(edge_attr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ---- Node encoding ----
        h = self.encode_nodes(x)       # [N, H]
        h_skip = h

        # ---- GATv2 Layer 1 ----
        h = self.gat1(h, edge_index, edge_attr=edge_attr)  # [N, 2H]
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---- GATv2 Layer 2 ----
        h = self.gat2(h, edge_index, edge_attr=edge_attr)  # [N, H]
        h = self.norm2(h)
        h = h + self.res_proj(h_skip)  # residual from input
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h

        # ---- Edge representations ----
        edge_repr = self.encode_edges(edge_attr)

        # ---- Attention pooling ----
        g = self.pool(h, batch)

        # ---- Scene encoder ----
        z = self.scene_encoder(g)

        return {
            "embedding": z,
            "node_embeddings": node_embeddings,
            "edge_repr": edge_repr,
        }

    def get_scene_embedding(self, data: Data) -> torch.Tensor:
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )
        result = self.forward(data.x, data.edge_index, data.edge_attr, batch)
        return result["embedding"]