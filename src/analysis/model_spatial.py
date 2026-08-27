"""
src/analysis/model_spatial.py

Spatial error metrics for ESP model predictions.

Two metrics computed per protein from test_predictions/*.npz:

  morans_i            — Moran's I spatial autocorrelation of per-vertex
                        absolute errors, using a KNN graph built from
                        query_pos.  Range ≈ [−1, 1].
                          → High (→1): errors cluster in coherent surface
                            patches — model has regional blind spots.
                          → Low (→0): errors are spatially diffuse —
                            model makes independent errors per vertex.

  esp_error_spearman  — Spearman r between |error| and |true_esp| across
                        all query vertices for one protein.
                          → High: model error correlates with ESP magnitude
                            — struggles more near strongly charged regions.
                          → Low: error is independent of ESP magnitude.

Results are cached as test_spatial_metrics.json in the checkpoint dir.

Usage:
    from src.analysis.model_spatial import compute_spatial_metrics
    metrics = compute_spatial_metrics(ckpt_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def _morans_i(errors: np.ndarray, positions: np.ndarray, k: int = 8) -> float:
    """Moran's I spatial autocorrelation of errors on a KNN graph."""
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=k + 1)   # idx[:, 0] is self
    neighbors = idx[:, 1:]                     # (N, k)

    n = len(errors)
    e = errors - errors.mean()
    e_var = float((e ** 2).sum())
    if e_var < 1e-10:
        return 0.0

    W = float(n * k)
    cross = float(sum(e[i] * e[neighbors[i]].sum() for i in range(n)))
    return (n / W) * (cross / e_var)


def compute_spatial_metrics(
    ckpt_dir: Path,
    k: int = 8,
    force: bool = False,
) -> dict:
    """
    Compute per-protein spatial error metrics for all test predictions.

    Reads every *_pred.npz from test_predictions/, computes Moran's I and
    ESP-error Spearman r, aggregates globally, and caches the result as
    test_spatial_metrics.json.

    Args:
        ckpt_dir: checkpoint directory containing test_predictions/
        k:        KNN neighbourhood size for Moran's I (default 8)
        force:    ignore cache and recompute

    Returns:
        Dict with keys:
            "global"      — mean/std of both metrics across proteins
            "per_protein" — {protein_id: {morans_i, esp_error_spearman, n_query}}
    """
    cache_path = ckpt_dir / "test_spatial_metrics.json"
    if cache_path.exists() and not force:
        print(f"  [spatial] Loading cached metrics ({cache_path.name})")
        with open(cache_path) as f:
            return json.load(f)

    pred_dir = ckpt_dir / "test_predictions"
    if not pred_dir.exists():
        print(f"  [spatial] test_predictions/ not found at {pred_dir}")
        return {}

    npz_files = sorted(pred_dir.glob("*_pred.npz"))
    if not npz_files:
        print(f"  [spatial] No *_pred.npz files in {pred_dir}")
        return {}

    per_protein: dict = {}
    for npz_path in npz_files:
        pid = npz_path.name.replace("_pred.npz", "")
        data = np.load(npz_path)
        query_pos = data["query_pos"].astype(np.float64)
        pred_esp  = data["pred_esp"].astype(np.float64)
        true_esp  = data["true_esp"].astype(np.float64)

        abs_err = np.abs(pred_esp - true_esp)

        mi = _morans_i(abs_err, query_pos, k=k)
        sr, _ = spearmanr(abs_err, np.abs(true_esp))
        sr = float(sr) if not np.isnan(sr) else 0.0

        per_protein[pid] = {
            "morans_i":           round(mi,    5),
            "esp_error_spearman": round(sr,    5),
            "n_query":            int(len(query_pos)),
        }
        print(f"  [spatial] {pid}  I={mi:.4f}  esp_r={sr:.4f}")

    mi_vals = [m["morans_i"]           for m in per_protein.values()]
    sr_vals = [m["esp_error_spearman"] for m in per_protein.values()]

    global_metrics = {
        "mean_morans_i":            round(float(np.mean(mi_vals)), 5),
        "std_morans_i":             round(float(np.std(mi_vals)),  5),
        "mean_esp_error_spearman":  round(float(np.mean(sr_vals)), 5),
        "std_esp_error_spearman":   round(float(np.std(sr_vals)),  5),
        "n_proteins":               len(per_protein),
    }

    result = {"global": global_metrics, "per_protein": per_protein}

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"  [spatial] Global — "
        f"Moran's I={global_metrics['mean_morans_i']:.4f}±{global_metrics['std_morans_i']:.4f}  "
        f"ESP-r={global_metrics['mean_esp_error_spearman']:.4f}±{global_metrics['std_esp_error_spearman']:.4f}  "
        f"({global_metrics['n_proteins']} proteins)"
    )
    print(f"  [spatial] Cached → {cache_path}")

    return result
