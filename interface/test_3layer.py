#!/usr/bin/env python3
"""
Test the SceneGraphEncoder (3-layer, 10-dim nodes, 13-dim edges)
through the unified interface — no checkpoint required.

Tests:
  1. Build model from scratch and verify forward pass shapes
  2. Single-scene encoding
  3. Batch encoding
  4. Cosine similarity between scenes
  5. Pairwise similarity matrix
  6. Full output (node embeddings + edge repr)
  7. Summary / repr
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch_geometric.data import Data

from dataset import NODE_FEAT_DIM, EDGE_FEAT_DIM, SceneGraphDataset, load_all_json_paths
from model import SceneGraphEncoder
from interface.encoder import SceneGraphEncoderInterface

DATA_DIR = "Data"
MODEL_TYPE = "3layer"
HIDDEN_DIM = 124
OUTPUT_DIM = 32


def make_interface() -> SceneGraphEncoderInterface:
    """Create an interface with a randomly initialized 3-layer model."""
    model = SceneGraphEncoder(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return SceneGraphEncoderInterface(model=model, model_type=MODEL_TYPE, device=device)


def test_summary(iface: SceneGraphEncoderInterface):
    print("=" * 60)
    print("TEST 1: Summary")
    print("=" * 60)
    print(iface.summary())
    print(repr(iface))
    print(f"  embedding_dim = {iface.embedding_dim}")
    print(f"  num_parameters = {iface.num_parameters:,}")
    assert iface.embedding_dim == OUTPUT_DIM
    print("  PASSED\n")


def test_single_encode(iface: SceneGraphEncoderInterface, json_path: str):
    print("=" * 60)
    print(f"TEST 2: Single scene encoding — {Path(json_path).name}")
    print("=" * 60)
    emb = iface.encode_scene(json_path)
    print(f"  Input:  {json_path}")
    print(f"  Output: shape={emb.shape}, dtype={emb.dtype}")
    assert emb.shape == (1, OUTPUT_DIM), f"Expected (1, {OUTPUT_DIM}), got {emb.shape}"
    print("  PASSED\n")
    return emb


def test_batch_encode(iface: SceneGraphEncoderInterface, json_paths: list):
    print("=" * 60)
    print(f"TEST 3: Batch encoding — {len(json_paths)} scenes")
    print("=" * 60)
    embs = iface.encode_batch(json_paths)
    print(f"  Input:  {len(json_paths)} files")
    print(f"  Output: shape={embs.shape}")
    assert embs.shape == (len(json_paths), OUTPUT_DIM)
    print("  PASSED\n")
    return embs


def test_compare(iface: SceneGraphEncoderInterface, path_a: str, path_b: str):
    print("=" * 60)
    print("TEST 4: Cosine similarity between two scenes")
    print("=" * 60)
    sim = iface.compare(path_a, path_b)
    print(f"  {Path(path_a).name} vs {Path(path_b).name}")
    print(f"  Cosine similarity: {sim:.6f}")
    assert -1.0 <= sim <= 1.0, f"Similarity out of range: {sim}"
    print("  PASSED\n")


def test_pairwise(iface: SceneGraphEncoderInterface, json_paths: list):
    print("=" * 60)
    print(f"TEST 5: Pairwise similarity matrix — {len(json_paths)} scenes")
    print("=" * 60)
    sim_mat = iface.pairwise_similarity(json_paths)
    n = len(json_paths)
    print(f"  Matrix shape: {sim_mat.shape}")
    assert sim_mat.shape == (n, n)
    # Diagonal should be ~1.0 (self-similarity)
    diag = sim_mat.diag()
    print(f"  Diagonal (self-sim): min={diag.min():.4f}, max={diag.max():.4f}")
    assert diag.min() > 0.99, f"Self-similarity too low: {diag.min():.4f}"
    print("  PASSED\n")


def test_full_output(iface: SceneGraphEncoderInterface, json_path: str):
    print("=" * 60)
    print("TEST 6: Full model output")
    print("=" * 60)
    out = iface.get_full_output(json_path)
    print(f"  Keys: {list(out.keys())}")
    print(f"  embedding:       {out['embedding'].shape}")
    print(f"  node_embeddings: {out['node_embeddings'].shape}")
    print(f"  edge_repr:       {out['edge_repr'].shape}")
    assert out["embedding"].shape[1] == OUTPUT_DIM
    assert out["node_embeddings"].shape[1] == HIDDEN_DIM
    assert out["edge_repr"].shape[1] == HIDDEN_DIM
    print("  PASSED\n")


def test_encode_from_edges(iface: SceneGraphEncoderInterface, json_path: str):
    print("=" * 60)
    print("TEST 7: Encode from in-memory edges")
    print("=" * 60)
    import json as json_mod
    with open(json_path) as f:
        edges = json_mod.load(f)
    emb = iface.encode_scene_from_edges(edges, scene_id="test_scene")
    print(f"  Output: shape={emb.shape}")
    assert emb.shape == (1, OUTPUT_DIM)
    print("  PASSED\n")


def main():
    print(f"\n{'#' * 60}")
    print(f"  Testing {MODEL_TYPE} model (SceneGraphEncoder)")
    print(f"  Node features: {NODE_FEAT_DIM}-dim | Edge features: {EDGE_FEAT_DIM}-dim")
    print(f"  Hidden: {HIDDEN_DIM} | Output: {OUTPUT_DIM}")
    print(f"{'#' * 60}\n")

    iface = make_interface()
    json_paths = sorted(Path(DATA_DIR).glob("*.json"))
    assert len(json_paths) >= 3, f"Need at least 3 JSON files in {DATA_DIR}"
    paths_str = [str(p) for p in json_paths]

    test_summary(iface)
    test_single_encode(iface, paths_str[0])
    test_batch_encode(iface, paths_str[:5])
    test_compare(iface, paths_str[0], paths_str[1])
    test_pairwise(iface, paths_str[:5])
    test_full_output(iface, paths_str[0])
    test_encode_from_edges(iface, paths_str[0])

    print("=" * 60)
    print("  ALL TESTS PASSED — 3layer model")
    print("=" * 60)


if __name__ == "__main__":
    main()
