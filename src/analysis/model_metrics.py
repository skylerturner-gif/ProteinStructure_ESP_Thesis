"""
src/analysis/model_metrics.py

Post-training full-mesh metrics for ESP model predictions.

Functions
---------
compute_full_mesh_metrics
    For each test protein, reconstruct the model's sparse query-point
    predictions to the full mesh via RBF, then compare against the APBS
    ground-truth at all vertices.  Reports complete_rmse and complete_pearson_r.

_build_metrics_df
    Helper: join per-protein metric dicts with protein metadata (sequence
    length, pLDDT, surface area) into a pandas DataFrame for plotting.

Naming convention
-----------------
  interp_*    — RBF reconstruction from ground-truth samples (baseline,
                see src/analysis/esp_stats.py)
  complete_*  — RBF reconstruction from model predictions vs ground truth
                (computed here)
  sparse_*    — model predictions at query vertices only vs ground truth
                at those same vertices (computed in trainer.evaluate_test)

Usage:
    from src.analysis.model_metrics import compute_full_mesh_metrics
    metrics = compute_full_mesh_metrics(ckpt_dir, data_root)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.surface.esp_mapping import reconstruct_full_mesh
from src.utils.io import load_metadata
from src.utils.paths import ProteinPaths


def compute_full_mesh_metrics(
    ckpt_dir: Path,
    data_root: Path,
    reconstruction: str = "multiquadric",
    force: bool = False,
) -> dict:
    """
    Reconstruct every test-protein prediction to the full mesh and compute
    metrics against the APBS ground truth at all vertices.

    Results are cached as test_metrics_fullmesh.json in ckpt_dir.
    Pass force=True to recompute even if the cache exists.

    Args:
        ckpt_dir:       checkpoint directory (contains test_predictions/)
        data_root:      protein data root
        reconstruction: RBF kernel — "multiquadric" (default), "gaussian",
                        or "nearest"
        force:          ignore cache and recompute from scratch

    Returns:
        Per-protein dict keyed by protein_id, each entry containing:
            complete_rmse, complete_pearson_r, mae, n_verts
    """
    cache_path = ckpt_dir / "test_metrics_fullmesh.json"
    if cache_path.exists() and not force:
        print(f"  [full-mesh] Loading cached metrics ({cache_path.name})")
        with open(cache_path) as f:
            return json.load(f)

    pred_dir = ckpt_dir / "test_predictions"
    if not pred_dir.exists():
        print(f"  [full-mesh] test_predictions/ not found at {pred_dir}")
        return {}

    npz_files = sorted(pred_dir.glob("*_pred.npz"))
    if not npz_files:
        print(f"  [full-mesh] No *_pred.npz files in {pred_dir}")
        return {}

    per_protein: dict = {}
    for npz_path in npz_files:
        pid = npz_path.name.replace("_pred.npz", "")
        p   = ProteinPaths(pid, data_root)

        if not p.mesh_path.exists():
            print(f"  [full-mesh] {pid}: mesh not found, skipping")
            continue
        if not p.esp_path.exists():
            print(f"  [full-mesh] {pid}: APBS ESP not found, skipping")
            continue

        pred_data     = np.load(npz_path)
        query_pos     = pred_data["query_pos"]
        pred_esp      = pred_data["pred_esp"]

        mesh_data     = np.load(p.mesh_path)
        mesh_verts    = mesh_data["verts"]

        esp_data      = np.load(p.esp_path)
        true_esp_full = esp_data["esp_verts"]

        print(
            f"  [full-mesh] {pid}  ({len(mesh_verts):,} verts, "
            f"{len(query_pos):,} query pts) ...",
            end=" ", flush=True,
        )

        pred_esp_full = reconstruct_full_mesh(
            query_pos, pred_esp, mesh_verts, method=reconstruction,
        )

        diff              = pred_esp_full - true_esp_full
        complete_rmse     = float(np.sqrt(np.mean(diff ** 2)))
        mae               = float(np.mean(np.abs(diff)))
        complete_pearson  = float(np.corrcoef(
            pred_esp_full.astype(float), true_esp_full.astype(float)
        )[0, 1])

        per_protein[pid] = {
            "complete_rmse":      round(complete_rmse,    5),
            "complete_pearson_r": round(complete_pearson, 5),
            "mae":                round(mae,              5),
            "n_verts":            int(len(mesh_verts)),
        }
        print(f"r={complete_pearson:.4f}  rmse={complete_rmse:.4f}")

    with open(cache_path, "w") as f:
        json.dump(per_protein, f, indent=2)
    print(f"  [full-mesh] Cached → {cache_path}")

    return per_protein


def _build_metrics_df(per_protein: dict, data_root: Path) -> pd.DataFrame:
    """
    Join a per-protein metrics dict with protein metadata.

    Args:
        per_protein: dict from compute_full_mesh_metrics or test_metrics.json
                     per_protein section.  Values must have rmse, pearson_r
                     (either key prefix is accepted).
        data_root:   protein data root for loading metadata

    Returns:
        DataFrame with columns: protein_id, rmse, mae, pearson_r,
        sequence_length, plddt, surface_area
    """
    rows = []
    for pid, m in per_protein.items():
        try:
            meta = load_metadata(pid, data_root)
        except FileNotFoundError:
            meta = {}

        # Accept either naming convention (complete_* or bare)
        rmse      = m.get("complete_rmse",      m.get("rmse",      float("nan")))
        pearson_r = m.get("complete_pearson_r", m.get("pearson_r", float("nan")))
        mae       = m.get("mae", float("nan"))

        rows.append({
            "protein_id":      pid,
            "rmse":            rmse,
            "mae":             mae,
            "pearson_r":       pearson_r,
            "sequence_length": meta.get("sequence_length", float("nan")),
            "plddt":           meta.get("plddt_mean",      float("nan")),
            "surface_area":    meta.get("surface_area",    float("nan")),
        })
    return pd.DataFrame(rows)
