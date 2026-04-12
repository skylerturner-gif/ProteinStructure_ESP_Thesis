"""
src/analysis/visualization.py

PyVista-based ESP surface visualization.

Renders the sampled ESP surface for a protein.

Usage (from a script or notebook):
    from src.analysis.visualization import plot_esp
    plot_esp(protein_id="AF-Q16613-F1", data_root=Path("/data"))
    plot_esp(protein_id="AF-Q16613-F1", data_root=Path("/data"), clim=(-5.0, 5.0))
"""

from pathlib import Path

import numpy as np
import pyvista as pv

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

def plot_esp(
    protein_id: str,
    data_root: Path,
    clim: tuple[float, float] = None,
) -> None:
    """
    Render a PyVista window showing the ESP surface for a protein.

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory
        clim:       optional (min, max) colormap range in kT/e.
                    Defaults to the global min/max of the surface.

    Raises:
        FileNotFoundError: if the ESP .npz file is missing
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.esp_path.exists():
        raise FileNotFoundError(
            f"Missing ESP file for '{protein_id}': {p.esp_path}"
        )

    verts, faces, esp_verts, esp_faces = _load_sampled(p.esp_path, plog)

    if clim is not None:
        plog.info("Colormap range: [%.3f, %.3f] kT/e", *clim)
    else:
        clim = (float(esp_faces.min()), float(esp_faces.max()))
        plog.info("Auto colormap range: [%.3f, %.3f] kT/e", *clim)

    mesh = _make_pv_mesh(verts, faces, esp_verts, esp_faces)

    plotter = pv.Plotter(window_size=(900, 700))
    plotter.add_text(
        f"{protein_id}  ({len(verts):,} verts)",
        position="upper_edge", font_size=11,
    )
    plotter.add_mesh(
        mesh,
        scalars="esp_verts",
        preference="point",
        cmap="coolwarm_r",
        clim=clim,
        show_edges=False,
    )
    plotter.add_scalar_bar(title="ESP (kT/e)", n_labels=5)

    plog.info("Launching PyVista viewer for %s", protein_id)
    plotter.show()
