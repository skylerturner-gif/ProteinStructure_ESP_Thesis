"""
src/analysis/metrics.py

Evaluation metrics for ESP surface predictions.

Provides functions to compare predicted ESP (Laplacian reconstruction)
against a reference ESP (nearest-neighbor interpolation from APBS).

Metrics:
    compute_stats    — Pearson r and RMSE between two ESP arrays
    evaluate_protein — load sampled .npz files for a protein and compute
                       stats, optionally writing to metadata

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
        esp_predicted: (N,) float array of predicted ESP values (e.g. Laplacian)
        esp_reference: (N,) float array of reference ESP values (e.g. interp)

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
    Load the PQR sampled ESP .npz files for a protein, compute Laplacian vs
    interpolated stats, and optionally write results back to the metadata JSON.

    Args:
        protein_id:     e.g. "AF-Q16613-F1"
        data_root:      root of the external data directory
        write_metadata: if True, writes stats to the protein's metadata JSON

    Returns:
        dict with keys:
            pearson_r_pqr — Pearson r between Laplacian and interpolated ESP
            rmse_pqr      — RMSE in kT/e

    Raises:
        FileNotFoundError: if any expected .npz file is missing
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    missing = [str(f) for f in [p.pqr_interp_path, p.pqr_laplacian_path]
               if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing sampled files for '{protein_id}':\n" +
            "\n".join(f"  {p}" for p in missing)
        )

    interp_data = np.load(p.pqr_interp_path)
    lap_data    = np.load(p.pqr_laplacian_path)

    esp_faces_interp = interp_data["esp_faces"]
    esp_faces_lap    = lap_data["esp_faces"]

    pearson_r, rmse = compute_stats(esp_faces_lap, esp_faces_interp)

    results = {
        "pearson_r_pqr": pearson_r,
        "rmse_pqr":      rmse,
    }

    plog.info("[pqr] Pearson r = %.4f   RMSE = %.4f kT/e", pearson_r, rmse)

    if write_metadata:
        update_metadata(protein_id, data_root=data_root, data={
            "pearson_r_pqr": round(pearson_r, 6),
            "rmse_pqr":      round(rmse, 6),
        })
        plog.info("Wrote evaluation stats to metadata")

    return results
