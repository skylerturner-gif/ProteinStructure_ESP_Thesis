"""
src/analysis/visualization.py

PyVista-based ESP surface visualization.

Renders a 1×2 comparison of interpolated vs Laplacian ESP for the PQR
mesh variant of a given protein.

Layout:
    Left  — Interpolated ESP (nearest-neighbor from APBS)
    Right — Laplacian-reconstructed ESP

Usage (from a script or notebook):
    from src.analysis.visualization import plot_esp_comparison
    plot_esp_comparison(protein_id="AF-Q16613-F1", data_root=Path("/data"))
    plot_esp_comparison(protein_id="AF-Q16613-F1", data_root=Path("/data"),
                        clim=(-5.0, 5.0))
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from src.analysis.metrics import compute_stats
from src.utils.helpers import get_logger
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_sampled(npz_file: Path, plog) -> tuple:
    """Load a sampled ESP .npz file. Returns (verts, faces, esp_verts, esp_faces)."""
    data      = np.load(npz_file)
    verts     = data["verts"]
    faces     = data["faces"]
    esp_verts = data["esp_verts"]
    esp_faces = data["esp_faces"]
    plog.info(
        "Loaded %s: %d verts, %d faces  esp [%.3f, %.3f]",
        npz_file.name, len(verts), len(faces),
        esp_verts.min(), esp_verts.max(),
    )
    return verts, faces, esp_verts, esp_faces


# ── PyVista mesh builder ──────────────────────────────────────────────────────

def _make_pv_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    esp_verts: np.ndarray,
    esp_faces: np.ndarray,
) -> pv.PolyData:
    """Build a PyVista PolyData mesh with ESP as point and cell scalars."""
    face_conn = np.hstack([np.full((len(faces), 1), 3), faces])
    mesh = pv.PolyData(verts, face_conn)
    mesh.point_data["esp_verts"] = esp_verts
    mesh.cell_data["esp_faces"]  = esp_faces
    return mesh


# ── Public API ────────────────────────────────────────────────────────────────

def plot_esp_comparison(
    protein_id: str,
    data_root: Path,
    clim: tuple[float, float] = None,
) -> None:
    """
    Render a 1×2 PyVista window comparing interpolated vs Laplacian ESP
    for the PQR mesh.

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory
        clim:       optional (min, max) colormap range in kT/e.
                    Defaults to the global min/max across both panels.

    Raises:
        FileNotFoundError: if any required sampled .npz file is missing
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    missing = [f for f in [p.pqr_interp_path, p.pqr_laplacian_path] if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing sampled files for '{protein_id}':\n" +
            "\n".join(f"  {f}" for f in missing)
        )

    verts_i, faces_i, esp_verts_i, esp_faces_i = _load_sampled(p.pqr_interp_path,    plog)
    verts_l, faces_l, esp_verts_l, esp_faces_l = _load_sampled(p.pqr_laplacian_path, plog)

    r_pqr, rmse_pqr = compute_stats(esp_faces_l, esp_faces_i)

    plog.info("── Visualization stats ──")
    plog.info("  [pqr] Pearson r = %.4f   RMSE = %.4f kT/e", r_pqr, rmse_pqr)

    if clim is not None:
        plog.info("Colormap range: [%.3f, %.3f] kT/e", *clim)
    else:
        all_esp = np.concatenate([esp_faces_i, esp_faces_l])
        clim = (float(all_esp.min()), float(all_esp.max()))
        plog.info("Auto colormap range: [%.3f, %.3f] kT/e", *clim)

    mesh_i = _make_pv_mesh(verts_i, faces_i, esp_verts_i, esp_faces_i)
    mesh_l = _make_pv_mesh(verts_l, faces_l, esp_verts_l, esp_faces_l)

    plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 600))

    def _add_panel(col: int, mesh: pv.PolyData, title: str) -> None:
        plotter.subplot(0, col)
        plotter.add_text(title, position="upper_edge", font_size=11)
        plotter.add_mesh(
            mesh,
            scalars="esp_verts",
            preference="point",
            cmap="coolwarm_r",
            clim=clim,
            show_edges=False,
        )
        plotter.add_scalar_bar(title="ESP (kT/e)", n_labels=5)

    _add_panel(0, mesh_i, f"Interpolated ({len(verts_i):,} verts)")
    _add_panel(1, mesh_l, f"Laplacian  r={r_pqr:.3f}  RMSE={rmse_pqr:.3f}")

    plotter.link_views()
    plog.info("Launching PyVista viewer for %s", protein_id)
    plotter.show()
