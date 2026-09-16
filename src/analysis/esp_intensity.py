"""
src/analysis/esp_intensity.py

Local ESP field statistics: how volatile is the ground-truth electrostatic
potential in the immediate neighborhood of a mesh vertex, independent of
any model prediction. Used to test whether prediction error is better
explained by locally unstable physics than by AlphaFold structural
confidence (pLDDT/PAE).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["local_esp_stats"]


def local_esp_stats(
    verts: np.ndarray, esp_verts: np.ndarray, query_idx: np.ndarray, k: int = 16
) -> dict[str, np.ndarray]:
    """
    Per-query-vertex local ESP field statistics, computed over each query
    vertex's k nearest mesh-vertex neighbors (ground truth only).

    Args:
        verts:     (n_verts, 3) all mesh vertex positions.
        esp_verts: (n_verts,) ground-truth ESP value at each mesh vertex.
        query_idx: (n_query,) indices into verts/esp_verts for query nodes.
        k:         neighborhood size (includes the vertex itself at k=0).

    Returns:
        {"local_esp_std": (n_query,), "local_esp_grad": (n_query,)} —
        local_esp_std is the neighborhood's ESP standard deviation (field
        volatility); local_esp_grad is the mean |Δesp| / distance over
        neighbors (a discrete local gradient magnitude), both float32.
    """
    tree = cKDTree(verts)
    query_pos = verts[query_idx]
    dist, nbr_idx = tree.query(query_pos, k=k)

    nbr_esp = esp_verts[nbr_idx]  # (n_query, k)
    local_esp_std = nbr_esp.std(axis=1)

    center_esp = esp_verts[query_idx][:, None]
    d_esp = np.abs(nbr_esp - center_esp)
    safe_dist = np.where(dist > 1e-6, dist, np.nan)
    grad = d_esp / safe_dist
    # first neighbor is the point itself (distance 0) -> nan, excluded from the mean
    local_esp_grad = np.nanmean(grad, axis=1)

    return {
        "local_esp_std":  local_esp_std.astype(np.float32),
        "local_esp_grad": local_esp_grad.astype(np.float32),
    }
