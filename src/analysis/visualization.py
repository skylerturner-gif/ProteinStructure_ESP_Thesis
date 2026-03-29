"""
src/analysis/visualization.py

PyVista-based ESP surface visualization.

Renders a 2x2 comparison of interpolated vs Laplacian ESP for both
no-H (PDB) and with-H (PQR) mesh variants for a given protein.

Layout:
    Top-left     — Interpolated ESP, no H  (PDB mesh)
    Top-right    — Interpolated ESP, with H (PQR mesh)
    Bottom-left  — Laplacian ESP, no H     (PDB mesh)
    Bottom-right — Laplacian ESP, with H   (PQR mesh)

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
    Render a 2x2 PyVista window comparing interpolated vs Laplacian ESP
    for both PDB (no-H) and PQR (with-H) mesh variants.

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory
        clim:       optional (min, max) colormap range in kT/e.
                    Defaults to the global min/max across all four panels.

    Raises:
        FileNotFoundError: if any required sampled .npz file is missing
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    missing = [f for f in [
        p.pdb_interp_path, p.pdb_laplacian_path,
        p.pqr_interp_path, p.pqr_laplacian_path,
    ] if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing sampled files for '{protein_id}':\n" +
            "\n".join(f"  {f}" for f in missing)
        )

    verts_pi, faces_pi, esp_verts_pi, esp_faces_pi = _load_sampled(p.pdb_interp_path,    plog)
    verts_pl, faces_pl, esp_verts_pl, esp_faces_pl = _load_sampled(p.pdb_laplacian_path, plog)
    verts_qi, faces_qi, esp_verts_qi, esp_faces_qi = _load_sampled(p.pqr_interp_path,    plog)
    verts_ql, faces_ql, esp_verts_ql, esp_faces_ql = _load_sampled(p.pqr_laplacian_path, plog)

    r_pdb, rmse_pdb = compute_stats(esp_faces_pl, esp_faces_pi)
    r_pqr, rmse_pqr = compute_stats(esp_faces_ql, esp_faces_qi)

    plog.info("── Visualization stats ──")
    plog.info("  [pdb] Pearson r = %.4f   RMSE = %.4f kT/e", r_pdb, rmse_pdb)
    plog.info("  [pqr] Pearson r = %.4f   RMSE = %.4f kT/e", r_pqr, rmse_pqr)

    if clim is not None:
        plog.info("Colormap range: [%.3f, %.3f] kT/e", *clim)
    else:
        all_esp = np.concatenate([esp_faces_pi, esp_faces_qi,
                                   esp_faces_pl, esp_faces_ql])
        clim = (float(all_esp.min()), float(all_esp.max()))
        plog.info("Auto colormap range: [%.3f, %.3f] kT/e", *clim)

    mesh_pi = _make_pv_mesh(verts_pi, faces_pi, esp_verts_pi, esp_faces_pi)
    mesh_pl = _make_pv_mesh(verts_pl, faces_pl, esp_verts_pl, esp_faces_pl)
    mesh_qi = _make_pv_mesh(verts_qi, faces_qi, esp_verts_qi, esp_faces_qi)
    mesh_ql = _make_pv_mesh(verts_ql, faces_ql, esp_verts_ql, esp_faces_ql)

    plotter = pv.Plotter(shape=(2, 2), window_size=(1400, 1000))

    def _add_panel(row: int, col: int, mesh: pv.PolyData, title: str) -> None:
        plotter.subplot(row, col)
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

    _add_panel(0, 0, mesh_pi, f"no H — interp ({len(verts_pi)} verts)")
    _add_panel(0, 1, mesh_qi, f"with H — interp ({len(verts_qi)} verts)")
    _add_panel(1, 0, mesh_pl, f"no H — Laplacian  r={r_pdb:.3f}  RMSE={rmse_pdb:.3f}")
    _add_panel(1, 1, mesh_ql, f"with H — Laplacian  r={r_pqr:.3f}  RMSE={rmse_pqr:.3f}")

    plotter.link_views()
    plog.info("Launching PyVista viewer for %s", protein_id)
    plotter.show()