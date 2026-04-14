"""
src/analysis/model_plots.py

Matplotlib figures for post-training ESP model analysis.

Functions
---------
plot_training_curves
    4-panel figure: train/val loss, validation RMSE, validation Pearson r,
    and learning-rate schedule vs epoch.  Source: metrics.csv.

plot_distributions
    Violin plots of per-protein RMSE and Pearson r grouped by sequence-length
    and pLDDT bins, for both sparse (query-point) and full-mesh metrics.

Usage:
    from src.analysis.model_plots import plot_training_curves, plot_distributions
    plot_training_curves(ckpt_dir, save_dir=Path("~/figures"), model_name="attention")
    plot_distributions(ckpt_dir, data_root, save_dir=Path("~/figures"))
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.model_metrics import compute_full_mesh_metrics, _build_metrics_df


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    ckpt_dir: Path,
    save_dir: Path | None = None,
    model_name: str = "",
) -> None:
    """
    Plot loss, RMSE, Pearson r, and LR from metrics.csv.

    Args:
        ckpt_dir:   checkpoint directory containing metrics.csv
        save_dir:   if given, saves training_curves.png here instead of showing
        model_name: optional prefix for the saved filename
    """
    csv_path = ckpt_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"  [curves] metrics.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training history — {ckpt_dir.name}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend()

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["val_rmse"], color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (kT/e)")
    ax.set_title("Validation RMSE")

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_pearson_r"], color="tab:green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pearson r")
    ax.set_title("Validation Pearson r")
    ax.set_ylim(bottom=max(0, df["val_pearson_r"].min() - 0.05))

    ax = axes[1, 1]
    ax.semilogy(df["epoch"], df["lr"], color="tab:red")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule")

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{model_name}_" if model_name else ""
        out = save_dir / f"{prefix}training_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [curves] Saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Distribution figure helper ────────────────────────────────────────────────

def _make_dist_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    """2×2 violin+scatter figure: RMSE and Pearson r by sequence-length and pLDDT."""

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
        data   = [df.loc[df[group_col] == g, metric_col].dropna().values for g in groups]
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


# ── Error distributions ───────────────────────────────────────────────────────

def plot_distributions(
    ckpt_dir: Path,
    data_root: Path,
    reconstruction: str = "multiquadric",
    force_recompute: bool = False,
    save_dir: Path | None = None,
    model_name: str = "",
) -> None:
    """
    Produce two violin-plot figures for all test proteins:

      1. Sparse (query-point) metrics — model predictions at the sampled ~5%
         of vertices vs. APBS ground truth at those same vertices.
         Source: test_metrics.json written by trainer.evaluate_test.

      2. Full-mesh (complete) metrics — predictions reconstructed to all mesh
         vertices via RBF, compared against the full APBS esp_verts array.
         Source: test_metrics_fullmesh.json (computed and cached here).

    A side-by-side summary table is printed to the terminal.

    Args:
        ckpt_dir:        checkpoint directory
        data_root:       protein data root
        reconstruction:  RBF kernel for full-mesh reconstruction
        force_recompute: ignore cache and recompute full-mesh metrics
        save_dir:        save figures here instead of showing interactively
        model_name:      prefix for saved filenames
    """
    # ── 1. Sparse metrics from test_metrics.json ──────────────────────────────
    sparse_per_protein: dict = {}
    metrics_path = ckpt_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            sparse_per_protein = json.load(f).get("per_protein", {})
    else:
        print("  [distributions] test_metrics.json not found — skipping sparse figure")

    # ── 2. Full-mesh (complete) metrics ───────────────────────────────────────
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
        key=lambda pid: full_per_protein.get(pid, {}).get("complete_pearson_r", 0),
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
            f"{s.get('pearson_r',         float('nan')):>9.4f}  "
            f"{s.get('rmse',              float('nan')):>12.4f}  "
            f"{f.get('complete_pearson_r', float('nan')):>7.4f}  "
            f"{f.get('complete_rmse',      float('nan')):>10.4f}"
        )

    # ── Build figures ─────────────────────────────────────────────────────────
    figs = []
    if sparse_per_protein:
        df_sparse = _build_metrics_df(sparse_per_protein, data_root)
        figs.append((
            _make_dist_figure(
                df_sparse,
                f"Sparse (query-point) metrics — {ckpt_dir.name}  "
                f"({len(df_sparse)} proteins, ~5% of surface)",
            ),
            f"{model_name}_error_distributions_sparse.png" if model_name
            else "error_distributions_sparse.png",
        ))

    if full_per_protein:
        df_full = _build_metrics_df(full_per_protein, data_root)
        figs.append((
            _make_dist_figure(
                df_full,
                f"Full-mesh metrics ({reconstruction} RBF) — "
                f"{ckpt_dir.name}  ({len(df_full)} proteins)",
            ),
            f"{model_name}_error_distributions_fullmesh.png" if model_name
            else "error_distributions_fullmesh.png",
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
