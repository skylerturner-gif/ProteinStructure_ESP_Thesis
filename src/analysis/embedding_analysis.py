"""
src/analysis/embedding_analysis.py

Post-training embedding analysis utilities:
  - load_model_frozen       : rebuild and load model from checkpoint, freeze weights
  - embedding_cosine_sim    : 7×7 pairwise cosine similarity of atom-type embeddings
  - collect_attention_stats : per-element attention weight mean/std across a dataset
  - compare_embedding_tables: per-element cosine sim between two models' embeddings

All functions are read-only — no model weights are modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.graph_builder import ELEMENT_VOCAB, N_ELEMENT_TYPES, RESIDUE_VOCAB, N_RESIDUE_TYPES
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.utils.paths import ProteinPaths

__all__ = [
    "ELEMENT_NAMES",
    "RESIDUE_NAMES",
    "load_model_frozen",
    "embedding_cosine_sim",
    "collect_attention_stats",
    "collect_attention_stats_residue",
    "compare_embedding_tables",
]

# H=0 C=1 N=2 O=3 S=4 P=5 unknown=6
ELEMENT_NAMES: list[str] = [
    e for e, _ in sorted(ELEMENT_VOCAB.items(), key=lambda x: x[1])
] + ["unknown"]

# 20 standard amino acids (0-19) + unknown (20)
RESIDUE_NAMES: list[str] = [
    r for r, _ in sorted(RESIDUE_VOCAB.items(), key=lambda x: x[1])
] + ["unknown"]


def load_model_frozen(
    ckpt_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Rebuild the model from checkpoint metadata, load weights, and freeze.

    The checkpoint must contain model_config and feature_spec (written by
    07_train.py via Trainer.extra_state). Loads best_model.pt, falling back
    to latest_model.pt.

    Returns:
        (model, ckpt) — frozen model on device; raw checkpoint dict with
        esp_mean, esp_std, model_config, feature_spec, model_name.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "latest_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc   = ckpt["model_config"]
    fc   = ckpt.get("feature_spec", {})

    common = dict(
        hidden_dim           = mc["hidden_dim"],
        n_rbf                = mc["n_rbf"],
        n_bond_radial_rounds = mc["n_bond_radial_rounds"],
        n_aq_rounds          = mc["n_aq_rounds"],
        n_qq_rounds          = mc["n_qq_rounds"],
        multi_agg            = mc.get("multi_agg", False),
        has_curvature        = fc.get("query_curvature", False),
        has_normal           = fc.get("query_normal", False),
    )

    if ckpt["model_name"] == "attention":
        model = AttentionESPN(**common, n_heads=mc.get("n_heads", 4))
    else:
        model = DistanceESPN(**common)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.requires_grad_(False)
    return model.to(device), ckpt


def _load_graph(protein_id: str, data_root: Path):
    """Load a pre-built HeteroData graph directly (bypasses feature_spec check)."""
    p = ProteinPaths(protein_id, data_root)
    return torch.load(p.graph_path(), weights_only=False)


def embedding_cosine_sim(model: nn.Module) -> tuple[np.ndarray, list[str]]:
    """
    Compute the pairwise cosine similarity matrix of the atom-type embedding table.

    Returns:
        sim:    (N_ELEMENT_TYPES, N_ELEMENT_TYPES) float32 array
        labels: element name strings, length N_ELEMENT_TYPES
    """
    w = model.atom_encoder.atom_emb.weight.detach().cpu().float().numpy()
    norms  = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    w_norm = w / norms
    sim    = (w_norm @ w_norm.T).astype(np.float32)
    return sim, ELEMENT_NAMES


def collect_attention_stats(
    model: AttentionESPN,
    protein_ids: list[str],
    data_root: Path,
    device: torch.device,
) -> dict[str, dict]:
    """
    Accumulate per-element attention weight statistics over a set of proteins.

    Uses the last AQ round's alpha weights (model.aq_layer._last_attn).
    Restores model.aq_layer.return_attn = False when done.

    Returns:
        {element_name: {"mean": [float]*n_heads, "std": [float]*n_heads, "count": int}}
    """
    if not isinstance(model, AttentionESPN):
        raise TypeError("collect_attention_stats requires an AttentionESPN model")

    n_heads      = model.aq_layer.n_heads
    alpha_sum    = np.zeros((N_ELEMENT_TYPES, n_heads), dtype=np.float64)
    alpha_sq_sum = np.zeros((N_ELEMENT_TYPES, n_heads), dtype=np.float64)
    elem_count   = np.zeros(N_ELEMENT_TYPES, dtype=np.float64)

    model.aq_layer.return_attn = True
    try:
        with torch.no_grad():
            for pid in protein_ids:
                data = _load_graph(pid, data_root)
                data = data.to(device)
                model(data)

                alpha    = model.aq_layer._last_attn.cpu().numpy()  # (E, H)
                src_idx  = data["atom", "aq", "query"].edge_index[0].cpu().numpy()
                elem_idx = data["atom"].atom_type[src_idx].cpu().numpy()  # (E,)

                for e in range(N_ELEMENT_TYPES):
                    mask = elem_idx == e
                    if mask.any():
                        alpha_sum[e]    += alpha[mask].sum(0)
                        alpha_sq_sum[e] += (alpha[mask] ** 2).sum(0)
                        elem_count[e]   += mask.sum()
    finally:
        model.aq_layer.return_attn = False
        model.aq_layer._last_attn  = None

    results: dict[str, dict] = {}
    for i, name in enumerate(ELEMENT_NAMES):
        n = elem_count[i]
        if n > 0:
            mean = alpha_sum[i] / n
            var  = np.maximum(alpha_sq_sum[i] / n - mean ** 2, 0)
            results[name] = {
                "mean":  mean.tolist(),
                "std":   np.sqrt(var).tolist(),
                "count": int(n),
            }
        else:
            results[name] = {"mean": [0.0] * n_heads, "std": [0.0] * n_heads, "count": 0}

    return results


def collect_attention_stats_residue(
    model: AttentionESPN,
    protein_ids: list[str],
    data_root: Path,
    device: torch.device,
) -> dict[str, dict]:
    """
    Accumulate per-residue attention weight statistics over a set of proteins.

    Groups AQ edge alpha weights by the residue type of the source atom node.
    Uses the last AQ round's alpha weights (model.aq_layer._last_attn).

    Returns:
        {residue_name: {"mean": [float]*n_heads, "std": [float]*n_heads, "count": int}}
    """
    if not isinstance(model, AttentionESPN):
        raise TypeError("collect_attention_stats_residue requires an AttentionESPN model")

    n_heads      = model.aq_layer.n_heads
    alpha_sum    = np.zeros((N_RESIDUE_TYPES, n_heads), dtype=np.float64)
    alpha_sq_sum = np.zeros((N_RESIDUE_TYPES, n_heads), dtype=np.float64)
    res_count    = np.zeros(N_RESIDUE_TYPES, dtype=np.float64)

    model.aq_layer.return_attn = True
    try:
        with torch.no_grad():
            for pid in protein_ids:
                data = _load_graph(pid, data_root)
                data = data.to(device)
                model(data)

                alpha   = model.aq_layer._last_attn.cpu().numpy()   # (E, H)
                src_idx = data["atom", "aq", "query"].edge_index[0].cpu().numpy()
                res_idx = data["atom"].residue_type[src_idx].cpu().numpy()  # (E,)

                for r in range(N_RESIDUE_TYPES):
                    mask = res_idx == r
                    if mask.any():
                        alpha_sum[r]    += alpha[mask].sum(0)
                        alpha_sq_sum[r] += (alpha[mask] ** 2).sum(0)
                        res_count[r]    += mask.sum()
    finally:
        model.aq_layer.return_attn = False
        model.aq_layer._last_attn  = None

    results: dict[str, dict] = {}
    for i, name in enumerate(RESIDUE_NAMES):
        n = res_count[i]
        if n > 0:
            mean = alpha_sum[i] / n
            var  = np.maximum(alpha_sq_sum[i] / n - mean ** 2, 0)
            results[name] = {
                "mean":  mean.tolist(),
                "std":   np.sqrt(var).tolist(),
                "count": int(n),
            }
        else:
            results[name] = {"mean": [0.0] * n_heads, "std": [0.0] * n_heads, "count": 0}

    return results


def compare_embedding_tables(model_a: nn.Module, model_b: nn.Module) -> np.ndarray:
    """
    Per-element cosine similarity between two models' atom-type embedding tables.

    Returns:
        (N_ELEMENT_TYPES,) float32 — one cosine similarity per element.
    """
    w_a = model_a.atom_encoder.atom_emb.weight.detach().cpu().float().numpy()
    w_b = model_b.atom_encoder.atom_emb.weight.detach().cpu().float().numpy()

    sims = np.zeros(w_a.shape[0], dtype=np.float32)
    for i in range(w_a.shape[0]):
        a, b    = w_a[i], w_b[i]
        denom   = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        sims[i] = float(np.dot(a, b) / denom)
    return sims
