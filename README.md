# Scene Graph GNN Encoder

A GATv2-based graph neural network that learns fixed-size scene embeddings from spatial scene graphs via **self-supervised contrastive learning**.

---

## Architecture

```
                         Scene Graph
                    (nodes + edges + attrs)
                             │
                ┌────────────┴────────────┐
                │                         │
         Node Features               Edge Features
         [N, 10]                      [E, 13]
         center(3)                    relation_onehot(6)
         extent(3)                    delta_center(3)
         log_extent(3)                distance(1)
         volume(1)                    extent_ratio(3)
                │                         │
        ┌───────┴───────┐         ┌───────┴───────┐
        │  Node Encoder │         │  Edge Encoder  │
        │   MLP         │         │   MLP          │
        │  10→256→H     │         │  13→64→H       │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                │  h₀ [N, H]              │  edge_repr [E, H]
                │                         │        │
     ┌──────────┼─────────────────────────┘        │
     │          │         (edge conditioning)      │
     │  ┌───────┴────────┐                         │
     │  │  GATv2 Layer 1 │                         │
     │  │  H → H/2 × 4h │──── concat ──→ [N, 2H]  │
     │  │  + LayerNorm   │                         │
     │  │  + ELU         │                         │
     │  │  + Dropout     │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │  ┌───────┴────────┐                         │
     │  │  GATv2 Layer 2 │                         │
     │  │  2H → H/2 × 4h │── concat ──→ [N, 2H]   │
     │  │  + LayerNorm   │                         │
     │  │  + Residual    │◄── from Layer 1 (2H)    │
     │  │  + ELU         │                         │
     │  │  + Dropout     │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │          │◄──── + Residual from h₀ (H→2H via linear proj)
     │          │                                  │
     │  ┌───────┴────────┐                         │
     │  │  GATv2 Layer 3 │                         │
     │  │  2H → H/2 × 2h │── concat ──→ [N, H]    │
     │  │  + LayerNorm   │                         │
     │  │  + Residual    │◄── from Layer 2 (2H→H via linear proj)
     │  │  + ELU         │                         │
     │  │  + Dropout     │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │   node_embeddings [N, H]                    │
     │          │                                  │
     │  ┌───────┴────────────┐                     │
     │  │ Attention Pooling  │                     │
     │  │ gate: H → 64 → 1  │                     │
     │  │ (learned weights)  │                     │
     │  └───────┬────────────┘                     │
     │          │                                  │
     │    graph vector [B, H]                      │
     │          │                                  │
     │  ┌───────┴────────┐                         │
     │  │ Scene Encoder  │                         │
     │  │  H → 64 → D   │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │   scene embedding                    edge_repr [E, H]
     │      z [B, D]                               │
     │          │                                  │
     └──────────┴──────────────────────────────────┘
                         │
                      Output:
                  {embedding: [B, D],
                   node_embeddings: [N, H],
                   edge_repr: [E, H]}

    H = hidden_dim (default 40)
    D = output_dim (default 32)
```

### Layer Details

| Component | Architecture | Dimensions (default H=40) |
|---|---|---|
| **Node Encoder** | Linear → ReLU → Linear | 10 → 256 → 40 |
| **Edge Encoder** | Linear → ReLU → Linear | 13 → 64 → 40 |
| **GATv2 Layer 1** | 4 heads, concat, edge-conditioned + LayerNorm + ELU | 40 → 80 |
| **GATv2 Layer 2** | 4 heads, concat, edge-conditioned + LayerNorm + ELU + residual | 80 → 80 |
| **GATv2 Layer 3** | 2 heads, concat, edge-conditioned + LayerNorm + ELU + residual | 80 → 40 |
| **Attention Pool** | Learned gate (H → 64 → 1), weighted node sum | 40 → 40 |
| **Scene Encoder** | Linear → ReLU → Linear | 40 → 64 → 32 |

### Key Design Choices

- **GATv2Conv** (not GAT): uses dynamic attention — attends based on both source and target node features, strictly more expressive than static GAT.
- **Edge conditioning**: raw 13-dim edge attributes are passed into every GATv2 layer so the attention mechanism is relation-aware.
- **Dual residual connections**: Layer 2 has a same-dimension residual from Layer 1 output, plus a projected residual from the initial node encoding (h₀). Layer 3 has a projected residual from Layer 2.
- **Attention pooling** (not mean/max): a learned gating network decides which nodes matter most for the scene-level representation.

---

## Architecture B — 2-Layer Light (`--model 2layer`)

A lighter variant with only 2 GATv2 layers, fewer parameters, and a single residual connection. Better suited for small datasets where the 3-layer model may overfit.

```
                         Scene Graph
                    (nodes + edges + attrs)
                             │
                ┌────────────┴────────────┐
                │                         │
         Node Features               Edge Features
         [N, 10]                      [E, 13]
                │                         │
        ┌───────┴───────┐         ┌───────┴───────┐
        │  Node Encoder │         │  Edge Encoder  │
        │  10→128→H     │         │  13→64→H       │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                │  h₀ [N, H]              │  edge_repr [E, H]
                │                         │        │
     ┌──────────┼─────────────────────────┘        │
     │          │         (edge conditioning)      │
     │  ┌───────┴────────┐                         │
     │  │  GATv2 Layer 1 │                         │
     │  │  H → H/2 × 4h │──── concat ──→ [N, 2H]  │
     │  │  + LayerNorm   │                         │
     │  │  + ELU         │                         │
     │  │  + Dropout     │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │  ┌───────┴────────┐                         │
     │  │  GATv2 Layer 2 │                         │
     │  │  2H → H/2 × 2h │── concat ──→ [N, H]    │
     │  │  + LayerNorm   │                         │
     │  │  + Residual    │◄── from h₀ (H→H via linear proj)
     │  │  + ELU         │                         │
     │  │  + Dropout     │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │   node_embeddings [N, H]                    │
     │          │                                  │
     │  ┌───────┴────────────┐                     │
     │  │ Attention Pooling  │                     │
     │  │ gate: H → 64 → 1  │                     │
     │  └───────┬────────────┘                     │
     │          │                                  │
     │  ┌───────┴────────┐                         │
     │  │ Scene Encoder  │                         │
     │  │  H → 64 → D   │                         │
     │  └───────┬────────┘                         │
     │          │                                  │
     │   scene embedding                    edge_repr [E, H]
     │      z [B, D]                               │
     │          │                                  │
     └──────────┴──────────────────────────────────┘
```

### Layer Details (2-Layer)

| Component | Architecture | Dimensions (default H=64) |
|---|---|---|
| **Node Encoder** | Linear → ReLU → Linear | 10 → 128 → 64 |
| **Edge Encoder** | Linear → ReLU → Linear | 13 → 64 → 64 |
| **GATv2 Layer 1** | 4 heads, concat, edge-conditioned + LayerNorm + ELU | 64 → 128 |
| **GATv2 Layer 2** | 2 heads, concat, edge-conditioned + LayerNorm + ELU + residual | 128 → 64 |
| **Attention Pool** | Learned gate (H → 64 → 1), weighted node sum | 64 → 64 |
| **Scene Encoder** | Linear → ReLU → Linear | 64 → 64 → 32 |

### Comparison

| | 3-Layer (`SceneGraphEncoder`) | 2-Layer (`SceneGraphEncoderLight`) |
|---|---|---|
| GATv2 layers | 3 | 2 |
| Residual connections | 3 (L1→L2, h₀→L2, L2→L3) | 1 (h₀→L2) |
| Node encoder width | 256 | 128 |
| Message-passing hops | 3 | 2 |
| Best for | Larger datasets, complex scenes | Small datasets (< 50 scenes) |

---

## Training Task

The model is trained with **SimCLR-style self-supervised contrastive learning** on scene graphs — no labels required.

### Contrastive Learning Pipeline

```
   Scene Graph Gᵢ
        │
   ┌────┴────┐
   │  Aug 1  │  Aug 2  │      Stochastic augmentations
   └────┬────┘────┬────┘
        │         │
    Encoder    Encoder        Shared weights (same model)
        │         │
      z1ᵢ       z2ᵢ           Scene embeddings
        │         │
        └────┬────┘
             │
       Should be similar       ← positive pair (same scene)
       (maximize similarity)

   z1ᵢ vs z1ⱼ (i ≠ j)         ← negative pairs (different scenes)
       Should be dissimilar
       (minimize similarity)
```

**Goal**: Learn an encoder $f_\theta$ such that two augmented views of the *same* scene graph map to nearby points in embedding space, while views from *different* scenes are pushed apart.

### Graph Augmentations

Each scene graph is randomly augmented to produce two different views. Four augmentations are applied stochastically:

| Augmentation | Default Probability | Effect |
|---|---|---|
| **Node feature masking** | 10% per feature dim | Zeros out random feature dimensions |
| **Node feature noise** | $\sigma = 0.05$ | Adds Gaussian noise to all features |
| **Edge dropping** | 15% per edge | Randomly removes edges (keeps ≥ 1) |
| **Node dropping** | 10% per node | Removes nodes and their edges (keeps ≥ 2 nodes) |

These ensure the model learns representations invariant to small perturbations in graph structure and features.

---

## Loss Functions

The total loss is a weighted combination:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \lambda \cdot \mathcal{L}_{\text{relation}}$$

### 1. InfoNCE Loss (Primary)

The main contrastive objective, following the NT-Xent / SimCLR formulation.

For a batch of $N$ graphs producing $2N$ embeddings (two views each):

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2N} \sum_{i=1}^{2N} \log \frac{\exp(\text{sim}(z_i, z_{i^+}) / \tau)}{\sum_{j \neq i} \exp(\text{sim}(z_i, z_j) / \tau)}$$

where:
- $z_i, z_{i^+}$ are embeddings of the two augmented views of the same scene (positive pair)
- $z_j$ for $j \neq i$ are all other embeddings in the batch (negatives)
- $\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$ is cosine similarity
- $\tau$ is the temperature parameter (default 0.1)

**Intuition**: For each embedding, the loss is a softmax cross-entropy that treats the matching view as the correct class and all other $2N-2$ embeddings as incorrect classes. Lower temperature $\tau$ makes the distribution sharper, demanding stronger separation.

### 2. Relation Prediction Loss (Auxiliary)

An edge-level cross-entropy loss that acts as a regularizer, ensuring the encoder preserves spatial relation information.

$$\mathcal{L}_{\text{relation}} = -\frac{1}{E} \sum_{e=1}^{E} \sum_{c=1}^{6} y_{e,c} \log \hat{y}_{e,c}$$

where:
- $E$ is the number of edges in the batch
- $y_{e,c}$ is the ground-truth one-hot relation label (6 classes: left, right, front, behind, above, below)
- $\hat{y}_{e,c}$ is the predicted probability from a small classifier head (H → 64 → 6) applied to the learned edge representations

**Why this helps**: Without it, the contrastive loss alone only optimizes for graph-level discrimination. The relation loss forces the edge encoder to preserve meaningful spatial semantics, which improves the quality of the learned representations, especially with small datasets.

### Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| $\tau$ (`temperature`) | 0.1 | InfoNCE sharpness — lower = stricter separation |
| $\lambda$ (`lambda_rel`) | 0.5 | Weight of relation loss — balances contrastive vs. auxiliary |

---

## Usage

```bash
conda activate gen

# Default training (3-layer model)
python train.py --data_dir Data/ --out_dir runs --epochs 200 --seed 42

# 2-layer light model
python train.py --data_dir Data/ --out_dir runs --model 2layer --epochs 200 --seed 42

# 2-layer with custom config
python train.py --data_dir Data/ --out_dir runs --model 2layer \
    --hidden_dim 64 --output_dim 32 --dropout 0.3 \
    --lr 5e-4 --temperature 0.1 --lambda_rel 0.5 \
    --epochs 200 --patience 30 --seed 42

# 3-layer with custom config
python train.py --data_dir Data/ --out_dir runs --model 3layer \
    --hidden_dim 40 --output_dim 32 --dropout 0.3 \
    --lr 5e-4 --temperature 0.1 --lambda_rel 0.5 \
    --epochs 200 --patience 30 --seed 42
```

### Outputs

Each run saves to `runs/<timestamp>/`:

```
runs/<timestamp>/
├── best_checkpoint.pt     # Model weights (best val loss)
├── config.json            # All hyperparameters
├── metrics.csv            # Per-epoch train/val losses
└── plots/
    ├── loss_curves.png    # Train & val loss over epochs
    ├── overfitting_gap.png # Train-val divergence
    └── embedding_pca.png  # PCA of learned scene embeddings
```

### Hyperparameter Sweep

Find the optimal configuration using **Alignment & Uniformity** metrics:

- **Alignment** $\mathcal{E}[\|z_i - z_i^+\|^2]$ — how close positive pairs are (lower = better)
- **Uniformity** $\log \mathcal{E}[e^{-2\|z_i - z_j\|^2}]$ — how uniformly spread embeddings are (lower = better, near 0 = collapse)

```bash
# Full sweep (2 models × 3 lr × 3 hidden_dim × 3 dropout × 2 weight_decay = 108 configs)
python sweep.py --data_dir Data/ --out_dir sweeps --epochs 100 --seed 42

# Quick sweep (16 configs, for fast iteration)
python sweep.py --data_dir Data/ --out_dir sweeps --epochs 100 --seed 42 --quick
```

Sweep outputs saved to `sweeps/<timestamp>/`:

```
sweeps/<timestamp>/
├── sweep_results.csv          # Ranked configs with all metrics
├── sweep_results.json         # Full results as JSON
└── plots/
    ├── alignment_vs_uniformity.png   # Scatter: align vs uniform per config
    ├── combined_score_bars.png       # Bar chart: A+U score ranked
    ├── heatmap_2layer.png            # hidden_dim × lr heatmap (2-layer)
    ├── heatmap_3layer.png            # hidden_dim × lr heatmap (3-layer)
    ├── regularization_impact.png     # Dropout & weight_decay effects
    └── top5_training_curves.png      # Loss curves for top 5 configs
```

The script prints a recommended `train.py` command for the best config at the end.
