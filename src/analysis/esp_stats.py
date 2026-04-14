"""
src/analysis/esp_stats.py

Ground-truth ESP surface characterisation and interpolation baseline.

Provides two functions:

  compute_stats       — Pearson r and RMSE between any two ESP arrays
  evaluate_protein    — Load the ground-truth ESP .npz for one protein,
                        run the RBF interpolation baseline from the saved
                        query_idx subset, and write interp_rmse /
                        interp_pearson_r to metadata.

The interpolation baseline answers: "how well can multiquadric RBF reconstruct
the full ESP surface from the curvature-sampled ~5% of vertices?"  This sets
the ceiling for model performance and is reported as interp_rmse /
interp_pearson_r in the per-protein metadata.

evaluate_protein always runs (no skip logic) — it is called by
pipelines/05_evaluate_esp.py at the end of every data-generation run to
ensure the query_idx and baseline metrics are current.

Usage:
    from src.analysis.esp_stats import evaluate_protein
    results = evaluate_protein("AF-Q16613-F1", data_root=Path("/data"))
"""

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.surface.esp_mapping import rbf_reconstruct
from src.utils.helpers import get_logger
from src.utils.io import update_metadata
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
    Load the ground-truth ESP .npz and compute the RBF interpolation baseline.

    Uses the canonical query_idx saved by sample_esp to reconstruct the full
    mesh from the sparse ~5% subset via multiquadric RBF, then compares the
    reconstruction to the ground-truth ESP at all vertices.

    This function always runs (no skip guard) — it is the authoritative source
    of interp_rmse and interp_pearson_r in the protein metadata.

    Args:
        protein_id:     e.g. "AF-Q16613-F1"
        data_root:      root of the external data directory
        write_metadata: if True, writes stats to the protein's metadata JSON

    Returns:
        dict with keys:
            interp_pearson_r — Pearson r of RBF reconstruction vs ground truth
            interp_rmse      — RMSE of RBF reconstruction vs ground truth (kT/e)
            esp_min          — minimum ESP value in kT/e
            esp_max          — maximum ESP value in kT/e
            esp_mean         — mean ESP value in kT/e
            esp_std          — standard deviation of ESP in kT/e
            n_verts          — number of surface vertices
            n_faces          — number of surface faces
            n_query          — number of query (curvature-sampled) vertices

    Raises:
        FileNotFoundError: if the ESP or mesh .npz file is missing
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.esp_path.exists():
        raise FileNotFoundError(
            f"Missing ESP file for '{protein_id}': {p.esp_path}"
        )
    if not p.mesh_path.exists():
        raise FileNotFoundError(
            f"Missing mesh file for '{protein_id}': {p.mesh_path}"
        )

    mesh_data = np.load(p.mesh_path)
    esp_data  = np.load(p.esp_path)

    verts     = mesh_data["verts"]
    faces     = mesh_data["faces"]
    esp_verts = esp_data["esp_verts"]
    esp_faces = esp_data["esp_faces"]
    query_idx = esp_data["query_idx"].astype(np.int64)

    n_verts = int(len(verts))
    n_faces = int(len(faces))
    n_query = int(len(query_idx))

    # RBF reconstruction from sparse subset → compare to full ground truth
    sparse_esp = esp_verts[query_idx]
    recon_esp  = rbf_reconstruct(verts, sparse_esp, query_idx, kernel="multiquadric")

    interp_r, _    = pearsonr(recon_esp.astype(float), esp_verts.astype(float))
    interp_rmse    = float(np.sqrt(np.mean((recon_esp - esp_verts) ** 2)))
    interp_pearson = float(interp_r)

    results = {
        "interp_pearson_r": round(interp_pearson, 5),
        "interp_rmse":      round(interp_rmse, 5),
        "esp_min":          float(esp_faces.min()),
        "esp_max":          float(esp_faces.max()),
        "esp_mean":         float(esp_faces.mean()),
        "esp_std":          float(esp_faces.std()),
        "n_verts":          n_verts,
        "n_faces":          n_faces,
        "n_query":          n_query,
    }

    plog.info(
        "Interp baseline  r=%.4f  rmse=%.4f kT/e  "
        "esp [%.3f, %.3f]  verts=%d  query=%d",
        interp_pearson, interp_rmse,
        results["esp_min"], results["esp_max"],
        n_verts, n_query,
    )

    if write_metadata:
        update_metadata(protein_id, data_root=data_root, data={
            "interp_pearson_r": results["interp_pearson_r"],
            "interp_rmse":      results["interp_rmse"],
            "esp_min":          round(results["esp_min"],  4),
            "esp_max":          round(results["esp_max"],  4),
            "esp_mean":         round(results["esp_mean"], 4),
            "esp_std":          round(results["esp_std"],  4),
        })
        plog.info("Wrote interpolation baseline stats to metadata")

    return results
