"""
src/analysis/model_visualization.py

PyVista side-by-side comparison of model predictions vs. ground truth.

Renders three panels per protein:
  1. Predicted ESP (model output, reconstructed to full mesh via RBF)
  2. Ground-truth ESP (APBS, full mesh)
  3. Absolute error (|predicted - ground truth|)

For ground-truth-only visualization (pre-training), see
src/analysis/esp_visualization.py.

Usage:
    from src.analysis.model_visualization import visualize_protein
    visualize_protein(
        "AF-Q16613-F1",
        pred_npz_path=Path("checkpoints/attention/test_predictions/AF-Q16613-F1_pred.npz"),
        data_root=Path("/data"),
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.surface.esp_mapping import reconstruct_full_mesh


def visualize_protein(
    protein_id: str,
    pred_npz_path: Path,
    data_root: Path,
    clim: tuple[float, float] | None = None,
    reconstruction: str = "multiquadric",
    save_path: Path | None = None,
) -> None:
    """
    Side-by-side PyVista view: predicted ESP | APBS truth | absolute error.

    The model prediction lives at sparse query points (~5% of mesh).
    Predictions are reconstructed to the full mesh using the chosen method
    before rendering.  Ground truth is taken from the APBS ESP .npz directly.

    Args:
        protein_id:     e.g. "AF-Q16613-F1"
        pred_npz_path:  path to <protein_id>_pred.npz from trainer.evaluate_test
        data_root:      protein data root
        clim:           optional (min, max) ESP colour range in kT/e.
                        Defaults to symmetric 99th-percentile of the truth.
        reconstruction: "multiquadric" (default) | "gaussian" | "nearest"
        save_path:      if given, saves a screenshot PNG instead of showing
                        an interactive window.
    """
    try:
        import pyvista as pv
        pv.global_theme.allow_empty_mesh = True
    except ImportError:
        print("  [visualize] pyvista not available — skipping.")
        return

    from src.utils.paths import ProteinPaths

    # Load prediction
    pred_data  = np.load(pred_npz_path)
    query_pos  = pred_data["query_pos"]   # (N_q, 3)
    pred_esp   = pred_data["pred_esp"]    # (N_q,) kT/e
    true_esp_q = pred_data["true_esp"]    # (N_q,) kT/e

    # Load full mesh + APBS ground-truth
    p = ProteinPaths(protein_id, data_root)
    if not p.mesh_path.exists():
        print(f"  [visualize] Mesh not found: {p.mesh_path}")
        return
    if not p.esp_path.exists():
        print(f"  [visualize] APBS ESP not found: {p.esp_path}")
        return

    mesh_data     = np.load(p.mesh_path)
    mesh_verts    = mesh_data["verts"]
    mesh_faces    = mesh_data["faces"]
    esp_data      = np.load(p.esp_path)
    true_esp_full = esp_data["esp_verts"]

    # Reconstruct full prediction from query points
    print(f"    Reconstructing full mesh ({reconstruction})...")
    pred_esp_full = reconstruct_full_mesh(query_pos, pred_esp, mesh_verts, method=reconstruction)
    error_full    = np.abs(pred_esp_full - true_esp_full)

    # Colour range
    if clim is None:
        p99  = float(np.percentile(np.abs(true_esp_full), 99))
        clim = (-p99, p99)

    # Build PyVista meshes
    face_conn = np.hstack([np.full((len(mesh_faces), 1), 3), mesh_faces])

    def _make_mesh(scalars):
        m = pv.PolyData(mesh_verts, face_conn)
        m["esp"] = scalars
        return m

    mesh_pred  = _make_mesh(pred_esp_full)
    mesh_true  = _make_mesh(true_esp_full)
    mesh_error = _make_mesh(error_full)

    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 650))

    _opts = dict(scalars="esp", cmap="coolwarm_r",
                 clim=clim, show_edges=False, show_scalar_bar=False)

    plotter.subplot(0, 0)
    plotter.add_text("Predicted (model)", position="upper_edge", font_size=10)
    plotter.add_mesh(mesh_pred, **_opts)
    plotter.add_scalar_bar(title="ESP (kT/e)", n_labels=5)

    plotter.subplot(0, 1)
    plotter.add_text("Ground truth (APBS)", position="upper_edge", font_size=10)
    plotter.add_mesh(mesh_true, **_opts)

    plotter.subplot(0, 2)
    err_max = float(np.percentile(error_full, 99))
    plotter.add_text("Absolute error", position="upper_edge", font_size=10)
    plotter.add_mesh(mesh_error, scalars="esp", cmap="hot_r",
                     clim=(0, err_max), show_edges=False, show_scalar_bar=False)
    plotter.add_scalar_bar(title="|error| (kT/e)", n_labels=5)

    plotter.link_views()

    # Title with stats
    r    = float(np.corrcoef(pred_esp, true_esp_q)[0, 1])
    rmse = float(np.sqrt(np.mean((pred_esp - true_esp_q) ** 2)))
    plotter.add_title(
        f"{protein_id}  |  Pearson r={r:.3f}  RMSE={rmse:.3f} kT/e  |  "
        f"{len(query_pos):,} query pts → {len(mesh_verts):,} verts ({reconstruction} recon)",
        font_size=9,
    )

    try:
        if save_path is not None:
            plotter.screenshot(str(save_path))
            print(f"    Saved → {save_path}")
        else:
            plotter.show()
    except Exception as exc:
        print(f"    [visualize] Render failed ({exc.__class__.__name__}: {exc})")
    finally:
        plotter.close()
