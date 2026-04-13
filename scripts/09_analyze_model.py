"""
scripts/09_analyze_model.py

Post-training analysis for ESP model predictions.

Modes (combine freely):
    --curves          Plot training/val loss, RMSE, Pearson r, and LR vs epoch.
    --distributions   Violin plots of per-protein RMSE and Pearson r, grouped
                      by sequence-length and pLDDT bins.
    --visualize       PyVista side-by-side view: reconstructed predicted ESP,
                      APBS ground-truth ESP, and absolute error on the full mesh.

Reconstruction:
    The model predicts ESP at ~5% of mesh vertices (query nodes).  This script
    uses KDTree nearest-neighbour interpolation to paint those predictions onto
    all mesh vertices for full-surface visualisation.  Ground-truth is taken
    directly from the APBS esp .npz (full coverage).

Usage:
    # All analyses, all test proteins
    python scripts/09_analyze_model.py --model attention \\
        --curves --distributions --visualize

    # Training curves only, save figures
    python scripts/09_analyze_model.py --model attention \\
        --curves --save-plots ~/thesis/figures

    # PyVista for one specific protein
    python scripts/09_analyze_model.py --model attention \\
        --visualize --protein-id AF-Q16613-F1

    # Explicit checkpoint and data-root
    python scripts/09_analyze_model.py \\
        --checkpoint-dir /path/to/checkpoints/attention \\
        --data-root /path/to/external_protein_data \\
        --curves --distributions --visualize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.surface.esp_mapping import rbf_reconstruct
from src.utils.config import get_data_root
from src.utils.io import load_metadata
from src.utils.paths import ProteinPaths


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct_full_mesh(
    query_pos: np.ndarray,
    query_esp: np.ndarray,
    mesh_verts: np.ndarray,
    method: str = "multiquadric",
) -> np.ndarray:
    """
    Reconstruct ESP at all mesh vertices from sparse query-point predictions.

    Query points are a curvature-prioritised 5% sample of mesh vertices.
    Their positions are matched back to vertex indices via KDTree (exact match
    since query_pos is a strict subset of mesh_verts), then passed to the
    chosen interpolation method.

    Methods:
        "multiquadric"  — RBF φ(r) = √(1+(ε·r)²), ε = 1/mean-nn-dist.
                          Best accuracy (r≈0.983, RMSE≈0.47 kT/e at 5%).
                          ~1s per protein.
        "gaussian"      — RBF φ(r) = exp(-(ε·r)²), same ε.
                          Slightly lower accuracy, similar speed.
        "nearest"       — 1-NN, instant but coarser (r≈0.954, RMSE≈0.77).

    Args:
        query_pos:  (N_q, 3) positions of model query nodes
        query_esp:  (N_q,)   predicted ESP at those positions (kT/e)
        mesh_verts: (N_v, 3) all mesh vertex positions
        method:     reconstruction method (default "multiquadric")

    Returns:
        (N_v,) float32 ESP at every mesh vertex
    """
    # Match query positions → vertex indices (query_pos ⊂ mesh_verts)
    tree = cKDTree(mesh_verts)
    _, sample_idx = tree.query(query_pos, k=1, workers=-1)
    sample_idx = sample_idx.astype(np.int64)

    if method == "nearest":
        # KDTree from query positions to all verts (fast path)
        q_tree = cKDTree(query_pos)
        _, nn_idx = q_tree.query(mesh_verts, k=1, workers=-1)
        return query_esp[nn_idx].astype(np.float32)

    return rbf_reconstruct(mesh_verts, query_esp, sample_idx, kernel=method)


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(ckpt_dir: Path, save_dir: Path | None = None) -> None:
    """
    Plot loss, RMSE, Pearson r, and LR from metrics.csv.

    Args:
        ckpt_dir: checkpoint directory containing metrics.csv
        save_dir: if given, saves training_curves.png here instead of showing
    """
    csv_path = ckpt_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"  [curves] metrics.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training history — {ckpt_dir.name}", fontsize=13)

    # Loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend()

    # RMSE
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["val_rmse"], color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (kT/e)")
    ax.set_title("Validation RMSE")

    # Pearson r
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_pearson_r"], color="tab:green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pearson r")
    ax.set_title("Validation Pearson r")
    ax.set_ylim(bottom=max(0, df["val_pearson_r"].min() - 0.05))

    # LR
    ax = axes[1, 1]
    ax.semilogy(df["epoch"], df["lr"], color="tab:red")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule")

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "training_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [curves] Saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Full-mesh metric computation ──────────────────────────────────────────────

def compute_full_mesh_metrics(
    ckpt_dir: Path,
    data_root: Path,
    reconstruction: str = "multiquadric",
    force: bool = False,
) -> dict:
    """
    Reconstruct every test protein prediction to the full mesh and compute
    metrics against the APBS ground truth at all vertices.

    Results are cached as test_metrics_fullmesh.json in ckpt_dir.
    Pass force=True to recompute even if the cache exists.

    Returns:
        Per-protein dict keyed by protein_id, each entry containing:
            rmse, mae, pearson_r, n_verts
    """
    cache_path = ckpt_dir / "test_metrics_fullmesh.json"
    if cache_path.exists() and not force:
        print(f"  [full-mesh] Loading cached metrics ({cache_path.name})")
        with open(cache_path) as f:
            return json.load(f)

    pred_dir = ckpt_dir / "test_predictions"
    if not pred_dir.exists():
        print(f"  [full-mesh] test_predictions/ not found at {pred_dir}")
        return {}

    npz_files = sorted(pred_dir.glob("*_pred.npz"))
    if not npz_files:
        print(f"  [full-mesh] No *_pred.npz files in {pred_dir}")
        return {}

    per_protein: dict = {}
    for npz_path in npz_files:
        pid = npz_path.name.replace("_pred.npz", "")
        p   = ProteinPaths(pid, data_root)

        if not p.mesh_path.exists():
            print(f"  [full-mesh] {pid}: mesh not found, skipping")
            continue
        if not p.esp_path.exists():
            print(f"  [full-mesh] {pid}: APBS ESP not found, skipping")
            continue

        pred_data     = np.load(npz_path)
        query_pos     = pred_data["query_pos"]
        pred_esp      = pred_data["pred_esp"]

        mesh_data     = np.load(p.mesh_path)
        mesh_verts    = mesh_data["verts"]

        esp_data      = np.load(p.esp_path)
        true_esp_full = esp_data["esp_verts"]

        print(f"  [full-mesh] {pid}  ({len(mesh_verts):,} verts, "
              f"{len(query_pos):,} query pts) ...", end=" ", flush=True)

        pred_esp_full = reconstruct_full_mesh(
            query_pos, pred_esp, mesh_verts, method=reconstruction,
        )

        diff = pred_esp_full - true_esp_full
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae  = float(np.mean(np.abs(diff)))
        r    = float(np.corrcoef(
            pred_esp_full.astype(float), true_esp_full.astype(float)
        )[0, 1])

        per_protein[pid] = {
            "rmse":      round(rmse, 5),
            "mae":       round(mae,  5),
            "pearson_r": round(r,    5),
            "n_verts":   int(len(mesh_verts)),
        }
        print(f"r={r:.4f}  rmse={rmse:.4f}")

    with open(cache_path, "w") as f:
        json.dump(per_protein, f, indent=2)
    print(f"  [full-mesh] Cached → {cache_path}")

    return per_protein


# ── Error distributions ───────────────────────────────────────────────────────

def _make_dist_figure(
    df: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """
    2×2 violin+scatter figure: RMSE and Pearson r by sequence-length and pLDDT.
    """
    def _bin(series: pd.Series) -> pd.Series:
        clean = series.dropna()
        if len(clean) < 2 or clean.nunique() < 2:
            return pd.Series("unknown", index=series.index)
        if len(clean) < 4:
            return pd.cut(series, bins=2, precision=0, duplicates="drop").astype(str)
        return pd.qcut(series, q=4, duplicates="drop", precision=0).astype(str)

    df = df.copy()
    df["seq_len_bin"] = _bin(df["sequence_length"])
    df["plddt_bin"]   = _bin(df["plddt"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=12)

    def _violin(ax, group_col, metric_col, ylabel):
        groups = sorted(df[group_col].dropna().unique())
        data   = [df.loc[df[group_col] == g, metric_col].dropna().values
                  for g in groups]
        data, groups = zip(*[(d, g) for d, g in zip(data, groups) if len(d) > 0]) \
            if any(len(d) > 0 for d in data) else ([], [])
        if not data:
            ax.set_title(f"{metric_col} by {group_col} (no data)"); return
        parts = ax.violinplot(list(data), showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        for i, d in enumerate(data, start=1):
            ax.scatter(np.full(len(d), i, dtype=float), d,
                       alpha=0.6, s=25, zorder=3, color="tab:blue")

    _violin(axes[0, 0], "seq_len_bin", "rmse",      "RMSE (kT/e)")
    axes[0, 0].set_title("RMSE by sequence length")
    _violin(axes[0, 1], "plddt_bin",   "rmse",      "RMSE (kT/e)")
    axes[0, 1].set_title("RMSE by pLDDT")
    _violin(axes[1, 0], "seq_len_bin", "pearson_r", "Pearson r")
    axes[1, 0].set_title("Pearson r by sequence length")
    _violin(axes[1, 1], "plddt_bin",   "pearson_r", "Pearson r")
    axes[1, 1].set_title("Pearson r by pLDDT")

    plt.tight_layout()
    return fig


def plot_distributions(
    ckpt_dir: Path,
    data_root: Path,
    reconstruction: str = "multiquadric",
    force_recompute: bool = False,
    save_dir: Path | None = None,
) -> None:
    """
    Produce two distribution figures for all test proteins:

      1. Sparse (query-point) metrics — model predictions at the sampled ~5%
         of vertices vs. APBS ground truth at those same vertices.
         Source: test_metrics.json (written by evaluate_test during training).

      2. Full-mesh metrics — predictions reconstructed to all mesh vertices
         via RBF, compared against the full APBS esp_verts array.
         Source: test_metrics_fullmesh.json (computed and cached here).

    Both figures use 2×2 violin plots grouped by sequence-length and pLDDT.
    A side-by-side summary table is printed to the terminal.

    Args:
        ckpt_dir:        checkpoint directory
        data_root:       protein data root
        reconstruction:  RBF kernel for full-mesh reconstruction
        force_recompute: ignore cache and recompute full-mesh metrics
        save_dir:        save figures to this directory instead of showing
    """
    n_proteins = len(list((ckpt_dir / "test_predictions").glob("*_pred.npz"))) \
        if (ckpt_dir / "test_predictions").exists() else 0

    # ── 1. Sparse (query-point) metrics from test_metrics.json ───────────────
    sparse_per_protein: dict = {}
    metrics_path = ckpt_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            sparse_per_protein = json.load(f).get("per_protein", {})
    else:
        print(f"  [distributions] test_metrics.json not found — skipping sparse figure")

    # ── 2. Full-mesh metrics (reconstruct + compare) ──────────────────────────
    print("  Computing full-mesh metrics (RBF reconstruction for each test protein)...")
    full_per_protein = compute_full_mesh_metrics(
        ckpt_dir, data_root,
        reconstruction=reconstruction,
        force=force_recompute,
    )

    if not sparse_per_protein and not full_per_protein:
        print("  [distributions] No metrics available.")
        return

    # ── Terminal summary table ────────────────────────────────────────────────
    all_pids = sorted(
        set(sparse_per_protein) | set(full_per_protein),
        key=lambda pid: full_per_protein.get(pid, {}).get("pearson_r", 0),
    )
    print(
        f"\n  {'Protein':<30}  "
        f"{'Sparse r':>9}  {'Sparse RMSE':>12}  "
        f"{'Full r':>7}  {'Full RMSE':>10}"
    )
    print("  " + "-" * 80)
    for pid in all_pids:
        s = sparse_per_protein.get(pid, {})
        f = full_per_protein.get(pid, {})
        print(
            f"  {pid:<30}  "
            f"{s.get('pearson_r', float('nan')):>9.4f}  "
            f"{s.get('rmse', float('nan')):>12.4f}  "
            f"{f.get('pearson_r', float('nan')):>7.4f}  "
            f"{f.get('rmse', float('nan')):>10.4f}"
        )

    # ── Build DataFrames with metadata ────────────────────────────────────────
    def _build_df(per_protein: dict) -> pd.DataFrame:
        rows = []
        for pid, m in per_protein.items():
            try:
                meta = load_metadata(pid, data_root)
            except FileNotFoundError:
                meta = {}
            rows.append({
                "protein_id":      pid,
                "rmse":            m["rmse"],
                "mae":             m["mae"],
                "pearson_r":       m["pearson_r"],
                "sequence_length": meta.get("sequence_length", np.nan),
                "plddt":           meta.get("plddt_mean",      np.nan),
                "surface_area":    meta.get("surface_area",    np.nan),
            })
        return pd.DataFrame(rows)

    figs = []

    if sparse_per_protein:
        df_sparse = _build_df(sparse_per_protein)
        figs.append((
            _make_dist_figure(
                df_sparse,
                f"Sparse (query-point) metrics — {ckpt_dir.name}  "
                f"({len(df_sparse)} proteins, ~5% of surface)",
            ),
            "error_distributions_sparse.png",
        ))

    if full_per_protein:
        df_full = _build_df(full_per_protein)
        figs.append((
            _make_dist_figure(
                df_full,
                f"Full-mesh metrics ({reconstruction} RBF reconstruction) — "
                f"{ckpt_dir.name}  ({len(df_full)} proteins)",
            ),
            "error_distributions_fullmesh.png",
        ))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for fig, fname in figs:
            out = save_dir / fname
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  [distributions] Saved → {out}")
            plt.close(fig)
    else:
        for fig, _ in figs:
            plt.figure(fig.number)
            plt.show()


# ── PyVista visualisation ─────────────────────────────────────────────────────

def visualize_protein(
    protein_id: str,
    pred_npz_path: Path,
    data_root: Path,
    clim: tuple[float, float] | None = None,
    reconstruction: str = "multiquadric",
) -> None:
    """
    Side-by-side PyVista view: predicted ESP | APBS truth | absolute error.

    The model prediction lives at sparse query points (~5% of mesh).
    Predictions are reconstructed to the full mesh using the chosen method
    before rendering.  Ground truth is taken from the APBS esp .npz directly.

    Args:
        protein_id:     e.g. "AF-Q16613-F1"
        pred_npz_path:  path to <protein_id>_pred.npz from evaluate_test
        data_root:      protein data root
        clim:           optional (min, max) for ESP colour range (kT/e).
                        Defaults to symmetric 99th-percentile of the truth.
        reconstruction: "multiquadric" (default) | "gaussian" | "nearest"
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  [visualize] pyvista not available — skipping.")
        return

    # Load prediction
    pred_data = np.load(pred_npz_path)
    query_pos = pred_data["query_pos"]   # (N_q, 3)
    pred_esp  = pred_data["pred_esp"]    # (N_q,) kT/e
    true_esp_q = pred_data["true_esp"]   # (N_q,) kT/e

    # Load full mesh + APBS ground-truth
    p = ProteinPaths(protein_id, data_root)
    if not p.mesh_path.exists():
        print(f"  [visualize] Mesh not found: {p.mesh_path}")
        return
    if not p.esp_path.exists():
        print(f"  [visualize] APBS ESP not found: {p.esp_path}")
        return

    mesh_data   = np.load(p.mesh_path)
    mesh_verts  = mesh_data["verts"]      # (N_v, 3)
    mesh_faces  = mesh_data["faces"]      # (F, 3)

    esp_data    = np.load(p.esp_path)
    true_esp_full = esp_data["esp_verts"] # (N_v,) APBS truth at all vertices

    # Reconstruct full prediction from query points to all mesh vertices
    print(f"    Reconstructing full mesh ({reconstruction})...")
    pred_esp_full = reconstruct_full_mesh(query_pos, pred_esp, mesh_verts, method=reconstruction)
    error_full    = np.abs(pred_esp_full - true_esp_full)

    # Colour range: symmetric 99th-percentile of APBS truth
    if clim is None:
        p99 = float(np.percentile(np.abs(true_esp_full), 99))
        clim = (-p99, p99)

    # Build PyVista meshes
    face_conn = np.hstack([np.full((len(mesh_faces), 1), 3), mesh_faces])

    def _make_mesh(verts, faces, scalars):
        m = pv.PolyData(verts, faces)
        m["esp"] = scalars
        return m

    mesh_pred  = _make_mesh(mesh_verts, face_conn, pred_esp_full)
    mesh_true  = _make_mesh(mesh_verts, face_conn, true_esp_full)
    mesh_error = _make_mesh(mesh_verts, face_conn, error_full)

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

    # Stats line
    r = float(np.corrcoef(pred_esp, true_esp_q)[0, 1])
    rmse = float(np.sqrt(np.mean((pred_esp - true_esp_q) ** 2)))
    title = (
        f"{protein_id}  |  "
        f"Pearson r={r:.3f}  RMSE={rmse:.3f} kT/e  |  "
        f"{len(query_pos):,} query pts → {len(mesh_verts):,} verts ({reconstruction} recon)"
    )
    plotter.add_title(title, font_size=9)

    plotter.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training analysis: training curves, error distributions, and PyVista visualisation."
    )

    # ── Location ──────────────────────────────────────────────────────────────
    parser.add_argument("--model", choices=["distance", "attention"], default=None,
                        help="Model type — used to infer checkpoint dir if "
                             "--checkpoint-dir is not set.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Checkpoint directory (contains metrics.csv, "
                             "test_metrics.json, test_predictions/).")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml.")

    # ── Modes ─────────────────────────────────────────────────────────────────
    parser.add_argument("--curves",        action="store_true",
                        help="Plot training curves from metrics.csv.")
    parser.add_argument("--distributions", action="store_true",
                        help="Plot error distributions from test_metrics.json.")
    parser.add_argument("--visualize",     action="store_true",
                        help="PyVista visualisation of test proteins.")

    # ── Options ───────────────────────────────────────────────────────────────
    parser.add_argument("--protein-id", type=str, default=None,
                        help="Limit --visualize to one protein ID.")
    parser.add_argument("--clim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=None,
                        help="Colour range for PyVista plots (kT/e).")
    parser.add_argument("--reconstruction",
                        choices=["multiquadric", "gaussian", "nearest"],
                        default="multiquadric",
                        help="RBF kernel for full-mesh reconstruction used by "
                             "--distributions and --visualize "
                             "(default: multiquadric). 'nearest' is fast but coarser.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore cached test_metrics_fullmesh.json and "
                             "recompute full-mesh metrics from scratch.")
    parser.add_argument("--save-plots", type=Path, default=None,
                        help="Save matplotlib figures to this directory instead "
                             "of displaying them interactively.")

    args = parser.parse_args()

    if not (args.curves or args.distributions or args.visualize):
        parser.error("Specify at least one of --curves, --distributions, --visualize.")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    data_root = args.data_root or get_data_root()

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    elif args.model:
        ckpt_dir = Path(data_root).parent / "checkpoints" / args.model
    else:
        parser.error("Provide --model or --checkpoint-dir.")

    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return

    pred_dir = ckpt_dir / "test_predictions"

    # ── Training curves ───────────────────────────────────────────────────────
    if args.curves:
        print("Plotting training curves...")
        plot_training_curves(ckpt_dir, save_dir=args.save_plots)

    # ── Error distributions ───────────────────────────────────────────────────
    if args.distributions:
        print("Plotting error distributions...")
        plot_distributions(
            ckpt_dir, data_root,
            reconstruction=args.reconstruction,
            force_recompute=args.force_recompute,
            save_dir=args.save_plots,
        )

    # ── PyVista visualisation ─────────────────────────────────────────────────
    if args.visualize:
        if not pred_dir.exists():
            print(f"  [visualize] test_predictions/ not found at {pred_dir}")
            print("  Run training to completion first (evaluate_test saves predictions).")
            return

        clim = tuple(args.clim) if args.clim else None

        if args.protein_id:
            npz = pred_dir / f"{args.protein_id}_pred.npz"
            if not npz.exists():
                print(f"  [visualize] {npz.name} not found in {pred_dir}")
                return
            proteins = [(args.protein_id, npz)]
        else:
            npz_files = sorted(pred_dir.glob("*_pred.npz"))
            if not npz_files:
                print(f"  [visualize] No *_pred.npz files in {pred_dir}")
                return
            proteins = [(f.name.replace("_pred.npz", ""), f) for f in npz_files]

        print(f"Visualising {len(proteins)} test protein(s)...")
        for protein_id, npz_path in proteins:
            print(f"  {protein_id}")
            visualize_protein(protein_id, npz_path, data_root,
                              clim=clim, reconstruction=args.reconstruction)


if __name__ == "__main__":
    main()
