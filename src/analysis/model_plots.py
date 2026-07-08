"""
src/analysis/model_plots.py

Matplotlib figures for post-training ESP model analysis.

Functions
---------
plot_training_curves
    4-panel figure: train/val loss, validation RMSE, validation Pearson r,
    and learning-rate schedule vs epoch.  Source: metrics.csv.

plot_distributions
    Three-panel figure per model run:
      [0] Vertex parity scatter (ground truth vs predicted ESP, subsampled
          query vertices), marker size ∝ sequence length, colour = net charge,
          OLS regression line and R² annotation.
      [1] Protein parity scatter (mean true vs mean predicted ESP, one point
          per protein), same size/colour encoding, OLS line and R².
      [2] Sequence length vs full-mesh RMSE, colour = net charge.

plot_error_by_esp
    Two-panel figure per model run (SUMMER_PLAN.md "Error Distribution by
    ESP Value"):
      [0] Residual (pred − true) vs ground-truth ESP value, all query
          vertices, colour = net charge. A binned median |residual| trend
          line shows whether error grows at ESP extremes.
      [1] Residual histograms split by |ESP| tertile (low/mid/high
          magnitude) — a widening distribution at high magnitude means the
          model struggles more with extreme ESP values.

Usage:
    from src.analysis.model_plots import plot_training_curves, plot_distributions, plot_error_by_esp
    plot_training_curves(ckpt_dir, save_dir=Path("~/figures"), model_name="attention")
    plot_distributions(ckpt_dir, data_root, save_dir=Path("~/figures"))
    plot_error_by_esp(ckpt_dir, data_root, save_dir=Path("~/figures"))
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
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


# ── Sparse prediction loader ──────────────────────────────────────────────────

def _load_sparse_preds(
    pids: list[str],
    pred_dir: Path,
    rng: np.random.Generator,
    max_pts: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load (true_esp, pred_esp) arrays from *_pred.npz for each protein.

    Returns a dict keyed by protein_id.  Proteins without a pred npz are
    silently skipped.  When max_pts is set, arrays are randomly subsampled
    so the parity scatter stays readable; pass None to load all vertices.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid in pids:
        path = pred_dir / f"{pid}_pred.npz"
        if not path.exists():
            continue
        d        = np.load(path)
        true_esp = d["true_esp"].astype(np.float32)
        pred_esp = d["pred_esp"].astype(np.float32)
        if max_pts is not None and len(true_esp) > max_pts:
            idx      = rng.choice(len(true_esp), max_pts, replace=False)
            true_esp = true_esp[idx]
            pred_esp = pred_esp[idx]
        result[pid] = (true_esp, pred_esp)
    return result


# ── Shared helpers ────────────────────────────────────────────────────────────

def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot."""
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _annotate_parity(
    ax: plt.Axes,
    all_true: np.ndarray,
    all_pred: np.ndarray,
    pad_frac: float = 0.03,
) -> None:
    """
    Add identity line, OLS regression line, and R² annotation to a parity axis.
    Axis limits are set to a square range covering both arrays.
    """
    lo  = min(all_true.min(), all_pred.min())
    hi  = max(all_true.max(), all_pred.max())
    pad = (hi - lo) * pad_frac
    lim = (lo - pad, hi + pad)

    # Identity line
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y = x")

    # OLS regression line
    m, b   = np.polyfit(all_true.astype(float), all_pred.astype(float), 1)
    xs     = np.array(lim)
    ax.plot(xs, m * xs + b, color="tab:red", lw=1.2, alpha=0.8,
            label=f"fit  (slope={m:.2f})")

    # R² annotation
    r2_val = _r2(all_true, all_pred)
    ax.text(
        0.04, 0.96, f"$R^2 = {r2_val:.4f}$",
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"),
    )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")


# ── Analysis figure ───────────────────────────────────────────────────────────

def _make_analysis_figure(
    df: pd.DataFrame,
    full_per_protein: dict,
    pred_dir: Path,
    title: str,
    max_pts_per_protein: int = 400,
) -> plt.Figure:
    """
    1×3 figure:
      [0] Vertex parity  — ground truth vs predicted ESP, subsampled to
          max_pts_per_protein vertices per protein.  Marker size ∝ sequence
          length; colour = net charge.  Includes OLS regression line and R².
      [1] Protein parity — one point per protein (mean true vs mean pred ESP).
          Same size/colour encoding.  Includes OLS regression line and R².
      [2] Size vs error  — sequence length vs full-mesh RMSE per protein;
          colour = net charge.
    """
    rng = np.random.default_rng(0)
    meta = df.set_index("protein_id")

    # ── Shared colour norm (diverging, centred at 0) ──────────────────────────
    charges = meta["net_charge"].dropna()
    abs_max = max(abs(charges.min()), abs(charges.max()), 1.0)
    norm    = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    cmap    = plt.cm.coolwarm

    # ── Shared marker size (seq_len → [20, 200] pt²) ─────────────────────────
    sl               = meta["sequence_length"].dropna()
    sl_min, sl_max   = float(sl.min()), float(sl.max())

    def _marker_size(pid: str) -> float:
        v = meta.at[pid, "sequence_length"] if pid in meta.index else sl_min
        if np.isnan(v) or sl_max == sl_min:
            return 80.0
        return 20.0 + 180.0 * (v - sl_min) / (sl_max - sl_min)

    def _net_charge(pid: str) -> float:
        return float(meta.at[pid, "net_charge"]) if pid in meta.index else 0.0

    # ── Load sparse predictions (full arrays + subsampled) ────────────────────
    pids        = df["protein_id"].tolist()
    sparse_full = _load_sparse_preds(pids, pred_dir, rng, max_pts=None)
    sparse_sub: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid, (true_, pred_) in sparse_full.items():
        if len(true_) > max_pts_per_protein:
            idx = rng.choice(len(true_), max_pts_per_protein, replace=False)
            sparse_sub[pid] = (true_[idx], pred_[idx])
        else:
            sparse_sub[pid] = (true_, pred_)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=12)
    ax_vert, ax_prot, ax_size = axes

    # ── [0] Vertex-level parity ───────────────────────────────────────────────
    cat_true_v: list[np.ndarray] = []
    cat_pred_v: list[np.ndarray] = []
    sc = None
    for pid, (true_esp, pred_esp) in sparse_sub.items():
        sc = ax_vert.scatter(
            true_esp, pred_esp,
            c=np.full(len(true_esp), _net_charge(pid)),
            s=_marker_size(pid),
            cmap=cmap, norm=norm,
            alpha=0.35, linewidths=0, rasterized=True,
        )
        cat_true_v.append(true_esp)
        cat_pred_v.append(pred_esp)

    if cat_true_v:
        _annotate_parity(ax_vert,
                         np.concatenate(cat_true_v),
                         np.concatenate(cat_pred_v))
        ax_vert.legend(fontsize=8, loc="lower right")

    ax_vert.set_xlabel("Ground truth ESP (kT/e)")
    ax_vert.set_ylabel("Predicted ESP (kT/e)")
    ax_vert.set_title(f"Vertex parity  (~{max_pts_per_protein}/protein)")

    if sc is not None:
        cb = plt.colorbar(sc, ax=ax_vert, shrink=0.8)
        cb.set_label("Net charge (e)")

    # Size legend — lower left so it doesn't overlap the R² box (top-left)
    size_handles = [
        plt.scatter([], [], s=20 + 180 * f, color="gray", alpha=0.6,
                    label=f"{int(sl_min + f * (sl_max - sl_min))} aa")
        for f in [0.0, 0.5, 1.0]
    ]
    ax_vert.add_artist(
        ax_vert.legend(handles=size_handles, title="Seq length",
                       fontsize=8, loc="lower left", framealpha=0.7)
    )

    # ── [1] Protein-level parity (one point per protein = mean ESP) ───────────
    mean_true: list[float] = []
    mean_pred: list[float] = []
    prot_pids: list[str]   = []
    for pid, (true_esp, pred_esp) in sparse_full.items():
        mean_true.append(float(np.mean(true_esp)))
        mean_pred.append(float(np.mean(pred_esp)))
        prot_pids.append(pid)

    if mean_true:
        arr_true = np.array(mean_true, dtype=float)
        arr_pred = np.array(mean_pred, dtype=float)

        sc2 = ax_prot.scatter(
            arr_true, arr_pred,
            c=[_net_charge(p) for p in prot_pids],
            s=[_marker_size(p) for p in prot_pids],
            cmap=cmap, norm=norm,
            alpha=0.85, linewidths=0.6, edgecolors="k",
            zorder=3,
        )
        _annotate_parity(ax_prot, arr_true, arr_pred)
        ax_prot.legend(fontsize=8, loc="lower right")

        # Size legend — upper right; R² is top-left, identity/fit is lower right
        prot_size_handles = [
            plt.scatter([], [], s=20 + 180 * f, color="gray", alpha=0.6,
                        label=f"{int(sl_min + f * (sl_max - sl_min))} aa")
            for f in [0.0, 0.5, 1.0]
        ]
        ax_prot.add_artist(
            ax_prot.legend(handles=prot_size_handles, title="Seq length",
                           fontsize=8, loc="upper right", framealpha=0.7)
        )

        cb2 = plt.colorbar(sc2, ax=ax_prot, shrink=0.8)
        cb2.set_label("Net charge (e)")

    ax_prot.set_xlabel("Mean ground truth ESP (kT/e)")
    ax_prot.set_ylabel("Mean predicted ESP (kT/e)")
    ax_prot.set_title(f"Protein parity  ({len(mean_true)} proteins)")

    # ── [2] Sequence length vs full-mesh RMSE ─────────────────────────────────
    for _, row in df.iterrows():
        pid  = row["protein_id"]
        rmse = full_per_protein.get(pid, {}).get("complete_rmse", float("nan"))
        if np.isnan(row["sequence_length"]) or np.isnan(rmse):
            continue
        ax_size.scatter(
            row["sequence_length"], rmse,
            c=[row["net_charge"]], cmap=cmap, norm=norm,
            s=_marker_size(pid), alpha=0.85, linewidths=0.5, edgecolors="k",
        )

    ax_size.set_xlabel("Sequence length (aa)")
    ax_size.set_ylabel("Full-mesh RMSE (kT/e)")
    ax_size.set_title("Protein size vs prediction error")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb3 = plt.colorbar(sm, ax=ax_size, shrink=0.8)
    cb3.set_label("Net charge (e)")

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
    Produce a 1×2 analysis figure for all test proteins:

      Left  — Parity scatter (ground truth vs predicted ESP at query vertices),
               marker size ∝ sequence length, colour = net charge.
      Right — Sequence length vs full-mesh RMSE, colour = net charge.

    A side-by-side summary table is also printed to the terminal.

    Args:
        ckpt_dir:        checkpoint directory (contains test_predictions/)
        data_root:       protein data root
        reconstruction:  RBF kernel for full-mesh reconstruction
        force_recompute: ignore cache and recompute full-mesh metrics
        save_dir:        save figure here instead of showing interactively
        model_name:      prefix for saved filename
    """
    # ── Sparse metrics from test_metrics.json ─────────────────────────────────
    sparse_per_protein: dict = {}
    metrics_path = ckpt_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            sparse_per_protein = json.load(f).get("per_protein", {})
    else:
        print("  [distributions] test_metrics.json not found")

    # ── Full-mesh (complete) metrics ──────────────────────────────────────────
    print("  Computing full-mesh metrics (RBF reconstruction for each test protein)...")
    full_result      = compute_full_mesh_metrics(
        ckpt_dir, data_root,
        reconstruction=reconstruction,
        force=force_recompute,
    )
    full_per_protein = full_result.get("per_protein", {})

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
            f"{s.get('pearson_r',          float('nan')):>9.4f}  "
            f"{s.get('rmse',               float('nan')):>12.4f}  "
            f"{f.get('complete_pearson_r',  float('nan')):>7.4f}  "
            f"{f.get('complete_rmse',       float('nan')):>10.4f}"
        )

    # ── Build figure ──────────────────────────────────────────────────────────
    # Use full-mesh metrics for the df (has rmse/pearson_r + metadata).
    # Fall back to sparse if full-mesh didn't run.
    source = full_per_protein if full_per_protein else sparse_per_protein
    df = _build_metrics_df(source, data_root)

    prefix = f"{model_name}_" if model_name else ""
    fname  = f"{prefix}error_distributions.png"
    title  = (
        f"Model analysis — {ckpt_dir.name}  "
        f"({len(all_pids)} test proteins, {reconstruction} RBF)"
    )

    fig = _make_analysis_figure(
        df, full_per_protein,
        pred_dir = ckpt_dir / "test_predictions",
        title    = title,
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [distributions] Saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Error distribution by ESP value ─────────────────────────────────────────────

def _net_charge_lookup(pids: list[str], data_root: Path) -> dict[str, float]:
    """Load net_charge for each protein_id, defaulting to 0.0 if missing."""
    from src.utils.io import load_metadata

    out: dict[str, float] = {}
    for pid in pids:
        try:
            out[pid] = float(load_metadata(pid, data_root).get("net_charge", 0.0))
        except FileNotFoundError:
            out[pid] = 0.0
    return out


def plot_error_by_esp(
    ckpt_dir: Path,
    data_root: Path,
    save_dir: Path | None = None,
    model_name: str = "",
    max_pts_per_protein: int = 2000,
    n_bins: int = 20,
) -> None:
    """
    Two-panel error-by-ESP-value diagnostic (SUMMER_PLAN.md "Error
    Distribution by ESP Value"): does the model struggle more at extreme
    positive/negative ESP values, and does that vary with a protein's net
    charge?

      Left  — Residual (pred − true ESP) vs ground-truth ESP, all query
              vertices (subsampled per protein for readability), colour =
              net charge. A black line shows the median |residual| within
              each ESP bin, making any error-vs-magnitude trend visible
              against the point cloud.
      Right — Residual histograms split by |ESP| tertile (low/mid/high
              magnitude, thresholds computed globally across all loaded
              vertices). A widening or off-centre distribution in the
              "high" tertile means the model is systematically worse at
              extreme ESP values, not just noisier.

    Args:
        ckpt_dir:             checkpoint directory (contains test_predictions/)
        data_root:            protein data root
        save_dir:             save figure here instead of showing interactively
        model_name:           prefix for saved filename
        max_pts_per_protein:  subsample cap per protein for the scatter panel
        n_bins:                number of ESP bins for the median-error trend line
    """
    pred_dir = ckpt_dir / "test_predictions"
    if not pred_dir.exists():
        print(f"  [error_by_esp] test_predictions/ not found at {pred_dir}")
        return

    pids = sorted(p.name.replace("_pred.npz", "") for p in pred_dir.glob("*_pred.npz"))
    if not pids:
        print(f"  [error_by_esp] No *_pred.npz files in {pred_dir}")
        return

    rng     = np.random.default_rng(0)
    charges = _net_charge_lookup(pids, data_root)
    sparse  = _load_sparse_preds(pids, pred_dir, rng, max_pts=None)

    all_true: list[np.ndarray] = []
    all_resid: list[np.ndarray] = []
    all_charge: list[np.ndarray] = []
    for pid, (true_esp, pred_esp) in sparse.items():
        n = len(true_esp)
        if n > max_pts_per_protein:
            idx = rng.choice(n, max_pts_per_protein, replace=False)
            true_esp = true_esp[idx]
            pred_esp = pred_esp[idx]
        all_true.append(true_esp)
        all_resid.append(pred_esp - true_esp)
        all_charge.append(np.full(len(true_esp), charges[pid]))

    if not all_true:
        print("  [error_by_esp] No predictions loaded.")
        return

    true_v  = np.concatenate(all_true)
    resid_v = np.concatenate(all_resid)
    charge_v = np.concatenate(all_charge)

    abs_max = max(abs(charge_v.min()), abs(charge_v.max()), 1.0)
    norm    = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    cmap    = plt.cm.coolwarm

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(15, 6))
    prefix = f"{model_name}_" if model_name else ""
    fig.suptitle(
        f"Error distribution by ESP value — {ckpt_dir.name}  ({len(pids)} test proteins)",
        fontsize=12,
    )

    # ── Left: residual vs ESP, coloured by net charge ────────────────────────
    sc = ax_scatter.scatter(
        true_v, resid_v, c=charge_v, cmap=cmap, norm=norm,
        s=8, alpha=0.25, linewidths=0, rasterized=True,
    )
    ax_scatter.axhline(0.0, color="k", lw=1, alpha=0.5)

    # Median |residual| trend, plotted on the same axis/units as the residual
    # scatter (both kT/e) — avoids a second y-axis and the label collisions
    # that come with it.
    edges = np.linspace(true_v.min(), true_v.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(true_v, edges) - 1, 0, n_bins - 1)
    bin_centers, bin_medians = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        bin_centers.append(0.5 * (edges[b] + edges[b + 1]))
        bin_medians.append(np.median(np.abs(resid_v[mask])))
    if bin_centers:
        ax_scatter.plot(bin_centers, bin_medians, color="black", lw=2, marker="o",
                        markersize=4, label="median |residual| (binned)", zorder=5)
        ax_scatter.legend(fontsize=8, loc="upper center")

    ax_scatter.set_xlabel("Ground truth ESP (kT/e)")
    ax_scatter.set_ylabel("Residual  (pred − true, kT/e)")
    ax_scatter.set_title("Residual vs ESP value")
    cb = plt.colorbar(sc, ax=ax_scatter, shrink=0.8, pad=0.12)
    cb.set_label("Net charge (e)")

    # ── Right: residual histograms by |ESP| tertile ──────────────────────────
    abs_true = np.abs(true_v)
    t1, t2 = np.percentile(abs_true, [100 / 3, 200 / 3])
    tiers = [
        ("low |ESP|",  abs_true <  t1,               "#4e79a7"),
        ("mid |ESP|",  (abs_true >= t1) & (abs_true < t2), "#f28e2b"),
        ("high |ESP|", abs_true >= t2,               "#e15759"),
    ]
    lo, hi = np.percentile(resid_v, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    for label, mask, color in tiers:
        if mask.sum() == 0:
            continue
        r = resid_v[mask]
        ax_hist.hist(
            r, bins=bins, color=color, alpha=0.5, density=True,
            label=f"{label}  (n={mask.sum():,}, median|r|={np.median(np.abs(r)):.3f})",
        )
    ax_hist.axvline(0.0, color="k", lw=1, alpha=0.5)
    ax_hist.set_xlabel("Residual  (pred − true, kT/e)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual distribution by |ESP| tertile")
    ax_hist.legend(fontsize=8)

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{prefix}error_by_esp.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [error_by_esp] Saved → {out}")
        plt.close(fig)
    else:
        plt.show()
