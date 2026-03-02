"""
Loss functions for contrastive scene graph representation learning.

Implements:
  - InfoNCE (NT-Xent / SimCLR) contrastive loss
  - Relation prediction auxiliary loss
  - Combined loss with weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss (SimCLR style).

    For a batch of N graphs, we produce 2N embeddings (two augmented views).
    Positive pairs: (z1_i, z2_i) for the same scene.
    Negative pairs: all other embeddings in the batch.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Scaling temperature τ for cosine similarities.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z1: Embeddings from view 1, shape [N, D].
            z2: Embeddings from view 2, shape [N, D].

        Returns:
            Scalar loss.
        """
        N = z1.size(0)
        device = z1.device

        # L2-normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [z1_0, ..., z1_{N-1}, z2_0, ..., z2_{N-1}]
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        # Full similarity matrix [2N, 2N]
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]

        # Mask out self-similarity (diagonal)
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        # Positive pair indices:
        #   z1_i (idx i)   -> z2_i (idx N+i)
        #   z2_i (idx N+i) -> z1_i (idx i)
        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),  # z1 -> z2
            torch.arange(0, N, device=device),       # z2 -> z1
        ])  # [2N]

        # InfoNCE = -log(exp(sim_pos) / sum(exp(sim_all_except_self)))
        # Using cross-entropy with positive indices as labels
        loss = F.cross_entropy(sim, pos_indices)
        return loss


class RelationPredictionLoss(nn.Module):
    """
    Auxiliary edge-level relation prediction loss.

    Predicts the relation type from learned edge representations.
    Uses cross-entropy on the relation one-hot vectors.
    """

    def __init__(self, edge_hidden_dim: int = 128, num_relations: int = 6):
        """
        Args:
            edge_hidden_dim: Dimension of the edge representations from GNN.
            num_relations: Number of relation classes.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(edge_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_relations),
        )
        self.num_relations = num_relations

    def forward(
        self, edge_repr: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relation prediction loss.

        Args:
            edge_repr: Edge representations from the GNN, shape [E, edge_hidden_dim].
            edge_attr: Original edge attributes containing relation one-hot, shape [E, 13].

        Returns:
            Scalar cross-entropy loss.
        """
        # Ground-truth labels from the first 6 dims (one-hot -> class index)
        relation_onehot = edge_attr[:, :self.num_relations]
        labels = relation_onehot.argmax(dim=1)

        logits = self.classifier(edge_repr)
        return F.cross_entropy(logits, labels)


class CombinedLoss(nn.Module):
    """
    Combined contrastive + relation prediction loss.

    total = loss_contrast + λ * loss_relation
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_rel: float = 0.5,
        use_relation_loss: bool = True,
        edge_hidden_dim: int = 128,
        num_relations: int = 6,
    ):
        """
        Args:
            temperature: InfoNCE temperature.
            lambda_rel: Weight for relation prediction loss.
            use_relation_loss: Whether to include relation prediction.
            edge_hidden_dim: Hidden dim for relation classifier input.
            num_relations: Number of relation categories.
        """
        super().__init__()
        self.contrast_loss = InfoNCELoss(temperature)
        self.lambda_rel = lambda_rel
        self.use_relation_loss = use_relation_loss
        if use_relation_loss:
            self.relation_loss = RelationPredictionLoss(edge_hidden_dim, num_relations)
        else:
            self.relation_loss = None

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        edge_repr: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            z1, z2: Scene embeddings for two augmented views [N, 32].
            edge_repr: Optional edge representations for relation loss.
            edge_attr: Optional original edge attributes.

        Returns:
            Dict with 'total', 'contrast', and optionally 'relation' losses.
        """
        loss_contrast = self.contrast_loss(z1, z2)
        result = {"contrast": loss_contrast}

        if (
            self.use_relation_loss
            and self.relation_loss is not None
            and edge_repr is not None
            and edge_attr is not None
        ):
            loss_rel = self.relation_loss(edge_repr, edge_attr)
            result["relation"] = loss_rel
            result["total"] = loss_contrast + self.lambda_rel * loss_rel
        else:
            result["relation"] = torch.tensor(0.0, device=z1.device)
            result["total"] = loss_contrast

        return result
