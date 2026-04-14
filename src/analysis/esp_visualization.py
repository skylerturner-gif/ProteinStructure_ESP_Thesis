"""
src/analysis/esp_visualization.py

PyVista-based ground-truth ESP surface visualization.

Renders a 3-panel view of the ground-truth APBS ESP surface for a single
protein:
  Left   — Ground truth (full mesh, all vertices)
  Centre — RBF interpolation reconstruction from the curvature-sampled subset
  Right  — Absolute difference |ground truth − reconstruction|

For post-training comparison of model predictions vs. ground truth, see
src/analysis/model_visualization.py.

Usage:
    from src.analysis.esp_visualization import plot_esp
    plot_esp(protein_id="AF-Q16613-F1", data_root=Path("/data"))
    plot_esp(protein_id="AF-Q16613-F1", data_root=Path("/data"), clim=(-5.0, 5.0))
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from src.surface.esp_mapping import rbf_reconstruct
from src.utils.helpers import get_logger
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_sampled(npz_file: Path, plog) -> tuple:
    """
    Load a ground-truth ESP .npz file.

    Returns (verts, faces, esp_verts, query_idx).
    Raises KeyError if query_idx is missing (run pipelines/04_sample_esp.py first).
    """
    data      = np.load(npz_file)
    verts     = data["verts"]
    faces     = data["faces"]
    esp_verts = data["esp_verts"]
    query_idx = data["query_idx"].astype(np.int64)
    plog.info(
        "Loaded %s: %d verts, %d faces, %d query verts  esp [%.3f, %.3f]",
        npz_file.name, len(verts), len(faces), len(query_idx),
        esp_verts.min(), esp_verts.max(),
    )
    return verts, faces, esp_verts, query_idx


# ── PyVista mesh builder ──────────────────────────────────────────────────────

def _make_pv_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    scalar_name: str,
) -> pv.PolyData:
    """Build a PyVista PolyData mesh with the given point scalars."""
    face_conn = np.hstack([np.full((len(faces), 1), 3), faces])
    mesh = pv.PolyData(verts, face_conn)
    mesh.point_data[scalar_name] = scalars
    return mesh


# ── Public API ────────────────────────────────────────────────────────────────

def plot_esp(
    protein_id: str,
    data_root: Path,
    clim: tuple[float, float] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Render a 3-panel ground-truth ESP surface for a protein.

    Panels:
      Left   — Ground truth (full mesh)
      Centre — RBF interpolation from curvature-sampled subset (query_idx)
      Right  — Absolute difference |ground truth − interpolation|

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory
        clim:       optional (min, max) colormap range in kT/e for the ESP
                    panels.  Defaults to the global min/max of ground truth.
        save_path:  if given, saves a screenshot PNG instead of showing
                    an interactive window.

    Raises:
        FileNotFoundError: if the ESP .npz file is missing
        KeyError: if query_idx is absent from the npz (re-run 04_sample_esp.py)
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.esp_path.exists():
        raise FileNotFoundError(
            f"Missing ESP file for '{protein_id}': {p.esp_path}"
        )

    verts, faces, esp_verts, query_idx = _load_sampled(p.esp_path, plog)

    # RBF reconstruction from sparse curvature sample
    sparse_esp = esp_verts[query_idx]
    recon_esp  = rbf_reconstruct(verts, sparse_esp, query_idx, kernel="multiquadric")
    abs_diff   = np.abs(esp_verts - recon_esp)

    if clim is None:
        clim = (float(esp_verts.min()), float(esp_verts.max()))
    diff_clim = (0.0, float(abs_diff.max()))

    plog.info(
        "ESP clim [%.3f, %.3f] kT/e  |diff| max %.3f kT/e",
        *clim, diff_clim[1],
    )

    gt_mesh    = _make_pv_mesh(verts, faces, esp_verts,  "ground_truth")
    recon_mesh = _make_pv_mesh(verts, faces, recon_esp,  "interpolation")
    diff_mesh  = _make_pv_mesh(verts, faces, abs_diff,   "abs_difference")

    n_query = len(query_idx)
    title = (
        f"{protein_id}  |  {len(verts):,} verts  "
        f"|  {n_query:,} query ({100 * n_query / len(verts):.1f}%)"
    )

    plotter = pv.Plotter(shape=(1, 3), window_size=(2400, 800))

    common = dict(cmap="coolwarm_r", clim=clim, show_edges=False, show_scalar_bar=False)

    plotter.subplot(0, 0)
    plotter.add_text("Ground truth", position="upper_edge", font_size=10)
    plotter.add_mesh(gt_mesh, scalars="ground_truth", **common)
    plotter.add_scalar_bar(title="ESP (kT/e)", n_labels=5)

    plotter.subplot(0, 1)
    plotter.add_text("RBF interpolation", position="upper_edge", font_size=10)
    plotter.add_mesh(recon_mesh, scalars="interpolation", **common)

    plotter.subplot(0, 2)
    plotter.add_text("|Ground truth − Interpolation|", position="upper_edge", font_size=10)
    plotter.add_mesh(diff_mesh, scalars="abs_difference",
                     cmap="Reds", clim=diff_clim, show_edges=False, show_scalar_bar=False)
    plotter.add_scalar_bar(title="|Δ ESP| (kT/e)", n_labels=5)

    plotter.add_text(title, position="upper_edge", font_size=9, viewport=True)

    plog.info("Launching PyVista viewer for %s", protein_id)
    if save_path is not None:
        plotter.screenshot(str(save_path))
        print(f"  Saved → {save_path}")
        plotter.close()
    else:
        plotter.show()
