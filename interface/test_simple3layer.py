#!/usr/bin/env python3
"""
Test the SceneGraphEncoderSimple3Layer (3-layer, 6-dim nodes, 6-dim edges)
through the unified interface — no checkpoint required.

Tests:
  1. Summary / properties
  2. Single-scene encoding
  3. Batch encoding
  4. Cosine similarity
  5. Pairwise similarity matrix
  6. Full output shapes
  7. Encode from in-memory edges
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from dataset import SIMPLE_NODE_FEAT_DIM, SIMPLE_EDGE_FEAT_DIM
from model import SceneGraphEncoderSimple3Layer
from interface.encoder import SceneGraphEncoderInterface

DATA_DIR = "Data"
MODEL_TYPE = "simple3layer"
HIDDEN_DIM = 64
OUTPUT_DIM = 32


def make_interface() -> SceneGraphEncoderInterface:
    model = SceneGraphEncoderSimple3Layer(
        node_feat_dim=SIMPLE_NODE_FEAT_DIM,
        edge_feat_dim=SIMPLE_EDGE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return SceneGraphEncoderInterface(model=model, model_type=MODEL_TYPE, device=device)


def test_summary(iface):
    print("=" * 60)
    print("TEST 1: Summary")
    print("=" * 60)
    print(iface.summary())
    print(repr(iface))
    assert iface.embedding_dim == OUTPUT_DIM
    print("  PASSED\n")


def test_single_encode(iface, json_path):
    print("=" * 60)
    print(f"TEST 2: Single scene encoding — {Path(json_path).name}")
    print("=" * 60)
    emb = iface.encode_scene(json_path)
    print(f"  Output: shape={emb.shape}")
    assert emb.shape == (1, OUTPUT_DIM)
    print("  PASSED\n")


def test_batch_encode(iface, json_paths):
    print("=" * 60)
    print(f"TEST 3: Batch encoding — {len(json_paths)} scenes")
    print("=" * 60)
    embs = iface.encode_batch(json_paths)
    print(f"  Output: shape={embs.shape}")
    assert embs.shape == (len(json_paths), OUTPUT_DIM)
    print("  PASSED\n")


def test_compare(iface, path_a, path_b):
    print("=" * 60)
    print("TEST 4: Cosine similarity")
    print("=" * 60)
    sim = iface.compare(path_a, path_b)
    print(f"  {Path(path_a).name} vs {Path(path_b).name}: {sim:.6f}")
    assert -1.0 <= sim <= 1.0
    print("  PASSED\n")


def test_pairwise(iface, json_paths):
    print("=" * 60)
    print(f"TEST 5: Pairwise similarity — {len(json_paths)} scenes")
    print("=" * 60)
    sim_mat = iface.pairwise_similarity(json_paths)
    n = len(json_paths)
    print(f"  Matrix shape: {sim_mat.shape}")
    assert sim_mat.shape == (n, n)
    assert sim_mat.diag().min() > 0.99
    print("  PASSED\n")


def test_full_output(iface, json_path):
    print("=" * 60)
    print("TEST 6: Full model output")
    print("=" * 60)
    out = iface.get_full_output(json_path)
    print(f"  embedding:       {out['embedding'].shape}")
    print(f"  node_embeddings: {out['node_embeddings'].shape}")
    print(f"  edge_repr:       {out['edge_repr'].shape}")
    assert out["embedding"].shape[1] == OUTPUT_DIM
    assert out["node_embeddings"].shape[1] == HIDDEN_DIM
    assert out["edge_repr"].shape[1] == HIDDEN_DIM
    print("  PASSED\n")


def test_encode_from_edges(iface, json_path):
    print("=" * 60)
    print("TEST 7: Encode from in-memory edges")
    print("=" * 60)
    import json
    with open(json_path) as f:
        edges = json.load(f)
    emb = iface.encode_scene_from_edges(edges)
    print(f"  Output: shape={emb.shape}")
    assert emb.shape == (1, OUTPUT_DIM)
    print("  PASSED\n")


def main():
    print(f"\n{'#' * 60}")
    print(f"  Testing {MODEL_TYPE} model (SceneGraphEncoderSimple3Layer)")
    print(f"  Node features: {SIMPLE_NODE_FEAT_DIM}-dim | Edge features: {SIMPLE_EDGE_FEAT_DIM}-dim")
    print(f"  Hidden: {HIDDEN_DIM} | Output: {OUTPUT_DIM}")
    print(f"{'#' * 60}\n")

    iface = make_interface()
    paths = [str(p) for p in sorted(Path(DATA_DIR).glob("*.json"))]
    assert len(paths) >= 3

    test_summary(iface)
    test_single_encode(iface, paths[0])
    test_batch_encode(iface, paths[:5])
    test_compare(iface, paths[0], paths[1])
    test_pairwise(iface, paths[:5])
    test_full_output(iface, paths[0])
    test_encode_from_edges(iface, paths[0])

    print("=" * 60)
    print("  ALL TESTS PASSED — simple3layer model (3-layer, 6-dim)")
    print("=" * 60)


if __name__ == "__main__":
    main()
