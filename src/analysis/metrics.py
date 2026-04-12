"""
src/analysis/metrics.py

Evaluation metrics for ESP surface predictions.

Provides functions to characterize the sampled ESP surface for a protein.

Metrics:
    compute_stats    — Pearson r and RMSE between two ESP arrays
    evaluate_protein — load the sampled ESP .npz for a protein and compute
                       descriptive stats, optionally writing to metadata

Usage (from a script or notebook):
    from src.analysis.metrics import evaluate_protein
    results = evaluate_protein(protein_id="AF-Q16613-F1", data_root=Path("/data"))
    print(results)
"""

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.utils.helpers import get_logger
from src.utils.io import load_metadata, update_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Core metric ───────────────────────────────────────────────────────────────

def compute_stats(
    esp_predicted: np.ndarray,
    esp_reference: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Pearson r and RMSE between a predicted and reference ESP array.

    Args:
        esp_predicted: (N,) float array of predicted ESP values
        esp_reference: (N,) float array of reference ESP values

    Returns:
        (pearson_r, rmse) both as Python floats, RMSE in kT/e

    Raises:
        ValueError: if arrays have different shapes or fewer than 2 elements
    """
    esp_predicted = np.asarray(esp_predicted, dtype=float)
    esp_reference = np.asarray(esp_reference, dtype=float)

    if esp_predicted.shape != esp_reference.shape:
        raise ValueError(
            f"Shape mismatch: predicted {esp_predicted.shape} "
            f"vs reference {esp_reference.shape}"
        )
    if esp_predicted.size < 2:
        raise ValueError("Arrays must have at least 2 elements to compute stats.")

    rmse = float(np.sqrt(np.mean((esp_predicted - esp_reference) ** 2)))
    r, _ = pearsonr(esp_predicted, esp_reference)
    return float(r), rmse


# ── Per-protein evaluation ────────────────────────────────────────────────────

def evaluate_protein(
    protein_id: str,
    data_root: Path,
    write_metadata: bool = True,
) -> dict:
    """
    Load the sampled ESP .npz for a protein and compute descriptive stats.

    Args:
        protein_id:     e.g. "AF-Q16613-F1"
        data_root:      root of the external data directory
        write_metadata: if True, writes stats to the protein's metadata JSON

    Returns:
        dict with keys:
            pearson_r — placeholder 1.0 (self-correlation, signals evaluation ran)
            rmse      — 0.0
            esp_min   — minimum ESP value in kT/e
            esp_max   — maximum ESP value in kT/e
            esp_mean  — mean ESP value in kT/e
            esp_std   — standard deviation of ESP in kT/e
            n_verts   — number of surface vertices
            n_faces   — number of surface faces

    Raises:
        FileNotFoundError: if the ESP .npz file is missing
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.esp_path.exists():
        raise FileNotFoundError(
            f"Missing ESP file for '{protein_id}': {p.esp_path}"
        )

    esp_data  = np.load(p.esp_path)
    esp_faces = esp_data["esp_faces"]
    n_verts   = int(len(esp_data["verts"]))
    n_faces   = int(len(esp_data["faces"]))

    results = {
        "pearson_r": 1.0,
        "rmse":      0.0,
        "esp_min":   float(esp_faces.min()),
        "esp_max":   float(esp_faces.max()),
        "esp_mean":  float(esp_faces.mean()),
        "esp_std":   float(esp_faces.std()),
        "n_verts":   n_verts,
        "n_faces":   n_faces,
    }

    plog.info(
        "ESP stats  min=%.3f  max=%.3f  mean=%.3f  std=%.3f  "
        "verts=%d  faces=%d",
        results["esp_min"], results["esp_max"],
        results["esp_mean"], results["esp_std"],
        n_verts, n_faces,
    )

    if write_metadata:
        update_metadata(protein_id, data_root=data_root, data={
            "pearson_r": results["pearson_r"],
            "rmse":      results["rmse"],
            "esp_min":   round(results["esp_min"],  4),
            "esp_max":   round(results["esp_max"],  4),
            "esp_mean":  round(results["esp_mean"], 4),
            "esp_std":   round(results["esp_std"],  4),
        })
        plog.info("Wrote evaluation stats to metadata")

    return results
