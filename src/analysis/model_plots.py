"""
src/analysis/model_plots.py

Matplotlib figures for post-training ESP model analysis.

Functions
---------
plot_training_curves
    4-panel figure: train/val loss, validation RMSE, validation Pearson r,
    and learning-rate schedule vs epoch.  Source: metrics.csv.

plot_parity_hexbin
    Pooled hexbin density parity plot (pred vs true ESP) across all query
    vertices, with marginal histograms and R²/slope annotation.

plot_sorted_predictions
    Sorts all query vertices by true ESP and overlays predicted values —
    over-smoothing shows as the predicted line collapsing toward zero at
    the extremes.

plot_delta_predictions
    Per-ESP-bin MAE-improvement bar chart comparing two checkpoint dirs.

Usage:
    from src.analysis.model_plots import plot_training_curves, plot_parity_hexbin, plot_sorted_predictions
    plot_training_curves(ckpt_dir, save_dir=Path("~/figures"), model_name="attention")
    plot_parity_hexbin(ckpt_dir, save_dir=Path("~/figures"))
    plot_sorted_predictions(ckpt_dir, save_dir=Path("~/figures"))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot."""
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ── Option A: parity hexbin ──────────────────────────────────────────────────

def plot_parity_hexbin(
    ckpt_dir: Path,
    save_dir: Path | None = None,
    model_name: str = "",
    gridsize: int = 80,
) -> None:
    """
    Hexbin parity scatter of predicted vs true ESP at all query vertices.

    Unlike the existing vertex parity plot (which subsamples per protein and
    colours by net charge), this pools every query vertex across all test
    proteins into a 2-D density map.  Over-smoothing shows as a horizontal
    band of density near pred ≈ 0 that does not track the identity line.

    Annotates: R², OLS slope (compression factor), and marginal histograms on
    each axis so you can compare the predicted vs true distributions directly.
    """
    pred_dir = ckpt_dir / "test_predictions"
    if not pred_dir.exists():
        print(f"  [parity_hexbin] test_predictions/ not found at {pred_dir}")
        return

    pids = sorted(p.name.replace("_pred.npz", "") for p in pred_dir.glob("*_pred.npz"))
    if not pids:
        print(f"  [parity_hexbin] No *_pred.npz files in {pred_dir}")
        return

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for pid in pids:
        d = np.load(pred_dir / f"{pid}_pred.npz")
        all_true.append(d["true_esp"].astype(np.float32))
        all_pred.append(d["pred_esp"].astype(np.float32))

    true_v = np.concatenate(all_true)
    pred_v = np.concatenate(all_pred)

    r2_val   = _r2(true_v, pred_v)
    slope, b = np.polyfit(true_v.astype(float), pred_v.astype(float), 1)

    # Figure with marginal histograms via gridspec
    fig = plt.figure(figsize=(8, 8))
    gs  = fig.add_gridspec(
        2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
        hspace=0.05, wspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    prefix = f"{model_name}_" if model_name else ""
    title  = f"Parity hexbin — {ckpt_dir.name}  ({len(pids)} proteins, {len(true_v):,} vertices)"
    fig.suptitle(title, fontsize=11, y=0.98)

    # ── Main hexbin ───────────────────────────────────────────────────────────
    lo  = min(float(true_v.min()), float(pred_v.min()))
    hi  = max(float(true_v.max()), float(pred_v.max()))
    pad = (hi - lo) * 0.03
    lim = (lo - pad, hi + pad)

    hb = ax_main.hexbin(
        true_v, pred_v,
        gridsize=gridsize, mincnt=1,
        cmap="Blues", bins="log",
        extent=[lim[0], lim[1], lim[0], lim[1]],
    )
    cb = plt.colorbar(hb, ax=ax_main, shrink=0.8, pad=0.02)
    cb.set_label("log₁₀(count)")

    ax_main.plot(lim, lim, "k--", lw=1.2, alpha=0.6, label="y = x  (perfect)")
    xs = np.array(lim, dtype=float)
    ax_main.plot(xs, slope * xs + b, color="tab:red", lw=1.5, alpha=0.85,
                 label=f"OLS  slope={slope:.3f}")
    ax_main.set_xlim(lim); ax_main.set_ylim(lim)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlabel("True ESP (kT/e)", fontsize=10)
    ax_main.set_ylabel("Predicted ESP (kT/e)", fontsize=10)
    ax_main.legend(fontsize=8, loc="lower right")
    ax_main.text(
        0.04, 0.96, f"$R^2 = {r2_val:.4f}$\nslope = {slope:.3f}",
        transform=ax_main.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"),
    )

    # ── Top marginal: true distribution ──────────────────────────────────────
    bins = np.linspace(lim[0], lim[1], 120)
    ax_top.hist(true_v, bins=bins, color="steelblue", alpha=0.7, density=True, label="true")
    ax_top.hist(pred_v, bins=bins, color="tab:orange", alpha=0.5, density=True, label="pred")
    ax_top.legend(fontsize=7, loc="upper right")
    ax_top.set_ylabel("Density", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    # ── Right marginal: predicted distribution ────────────────────────────────
    ax_right.hist(pred_v, bins=bins, color="tab:orange", alpha=0.7, density=True,
                  orientation="horizontal")
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel("Density", fontsize=8)

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{prefix}parity_hexbin.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [parity_hexbin] Saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Option B: sorted prediction plot ─────────────────────────────────────────

def plot_sorted_predictions(
    ckpt_dir: Path,
    save_dir: Path | None = None,
    model_name: str = "",
    n_bins: int = 300,
) -> None:
    """
    Sort all query vertices by true ESP value and overlay the predicted values.

    The true ESP line is monotonically increasing (S-curve from negative to
    positive); the predicted line shows the model's behaviour at each ESP
    magnitude.  Over-smoothing manifests as the predicted line collapsing
    toward zero at the extremes — the two lines diverge at the tails.

    Vertices are binned into n_bins quantile buckets so both lines are smooth
    and readable regardless of dataset size.  The error band (|pred − true|)
    is shaded between them.
    """
    pred_dir = ckpt_dir / "test_predictions"
    if not pred_dir.exists():
        print(f"  [sorted_pred] test_predictions/ not found at {pred_dir}")
        return

    pids = sorted(p.name.replace("_pred.npz", "") for p in pred_dir.glob("*_pred.npz"))
    if not pids:
        print(f"  [sorted_pred] No *_pred.npz files in {pred_dir}")
        return

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for pid in pids:
        d = np.load(pred_dir / f"{pid}_pred.npz")
        all_true.append(d["true_esp"].astype(np.float32))
        all_pred.append(d["pred_esp"].astype(np.float32))

    true_v = np.concatenate(all_true)
    pred_v = np.concatenate(all_pred)

    # Sort by true ESP
    order  = np.argsort(true_v)
    true_s = true_v[order]
    pred_s = pred_v[order]

    # Bin into n_bins quantile buckets (mean per bucket)
    splits    = np.array_split(np.arange(len(true_s)), n_bins)
    bin_true  = np.array([true_s[idx].mean() for idx in splits])
    bin_pred  = np.array([pred_s[idx].mean() for idx in splits])
    bin_pred_std = np.array([pred_s[idx].std()  for idx in splits])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    prefix = f"{model_name}_" if model_name else ""
    fig.suptitle(
        f"Sorted prediction plot — {ckpt_dir.name}  "
        f"({len(pids)} proteins, {len(true_v):,} vertices, {n_bins} bins)",
        fontsize=11,
    )

    ax_pred, ax_err = axes

    # ── Top: true vs predicted ────────────────────────────────────────────────
    ax_pred.plot(bin_true, bin_true, color="steelblue", lw=2, label="True ESP")
    ax_pred.plot(bin_true, bin_pred, color="tab:orange", lw=2, linestyle="--",
                 label="Predicted ESP")
    ax_pred.fill_between(
        bin_true,
        bin_pred - bin_pred_std,
        bin_pred + bin_pred_std,
        color="tab:orange", alpha=0.15, label="±1 σ (within bin)",
    )
    ax_pred.axhline(0.0, color="k", lw=0.8, alpha=0.4, linestyle=":")
    ax_pred.set_ylabel("ESP (kT/e)", fontsize=10)
    ax_pred.legend(fontsize=9, loc="upper left")

    # ── Bottom: signed error (pred - true) ───────────────────────────────────
    error = bin_pred - bin_true
    ax_err.plot(bin_true, error, color="tab:red", lw=1.5)
    ax_err.fill_between(bin_true, error, 0.0, where=(error >= 0),
                        color="tab:red", alpha=0.3)
    ax_err.fill_between(bin_true, error, 0.0, where=(error < 0),
                        color="tab:blue", alpha=0.3)
    ax_err.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax_err.set_xlabel("True ESP (kT/e, sorted)", fontsize=10)
    ax_err.set_ylabel("Pred − True (kT/e)", fontsize=10)

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{prefix}sorted_predictions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [sorted_pred] Saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Option C: delta sorted prediction plot ────────────────────────────────────

def _load_delta_bins(
    input_ckpt_dir: Path,
    compared_ckpt_dir: Path,
    n_bins: int = 300,
) -> tuple[list[str], np.ndarray, np.ndarray, int] | tuple[None, None, None, None]:
    """Shared data loading + binning for delta-vs-baseline analysis.

    For each quantile bin of true ESP value (sorted, split into n_bins equal-
    count groups), computes mean(|input_error|) − mean(|compared_error|).

    Returns (shared_pids, bin_true, bin_delta) where bin_delta[i] > 0 means the
    compared model has lower MAE than the input model in that bin, or
    (None, None, None) if the two checkpoint dirs have no usable overlap.
    Used by both plot_delta_predictions (single net total) and
    range_delta_breakdown (net total per named true-ESP range) so both stay
    consistent with each other.
    """
    input_pred_dir    = input_ckpt_dir    / "test_predictions"
    compared_pred_dir = compared_ckpt_dir / "test_predictions"

    if not input_pred_dir.exists() or not compared_pred_dir.exists():
        print(f"  [delta_pred] Missing test_predictions/ in one or both dirs")
        return None, None, None, None

    input_pids    = {p.name.replace("_pred.npz", "") for p in input_pred_dir.glob("*_pred.npz")}
    compared_pids = {p.name.replace("_pred.npz", "") for p in compared_pred_dir.glob("*_pred.npz")}
    shared_pids   = sorted(input_pids & compared_pids)

    if not shared_pids:
        print(f"  [delta_pred] No shared proteins between dirs")
        return None, None, None, None

    all_true:         list[np.ndarray] = []
    all_input_pred:   list[np.ndarray] = []
    all_compared_pred: list[np.ndarray] = []

    for pid in shared_pids:
        di = np.load(input_pred_dir    / f"{pid}_pred.npz")
        dc = np.load(compared_pred_dir / f"{pid}_pred.npz")
        all_true.append(di["true_esp"].astype(np.float32))
        all_input_pred.append(di["pred_esp"].astype(np.float32))
        all_compared_pred.append(dc["pred_esp"].astype(np.float32))

    true_v     = np.concatenate(all_true)
    input_v    = np.concatenate(all_input_pred)
    compared_v = np.concatenate(all_compared_pred)

    order      = np.argsort(true_v)
    true_s     = true_v[order]
    input_s    = input_v[order]
    compared_s = compared_v[order]

    splits    = np.array_split(np.arange(len(true_s)), n_bins)
    bin_true  = np.array([true_s[idx].mean()  for idx in splits])
    bin_delta = np.array([
        np.abs(input_s[idx]    - true_s[idx]).mean()
        - np.abs(compared_s[idx] - true_s[idx]).mean()
        for idx in splits
    ])
    return shared_pids, bin_true, bin_delta, len(true_s)


def range_delta_breakdown(
    input_ckpt_dir: Path,
    compared_ckpt_dir: Path,
    ranges: tuple[tuple[float, float], ...] = (
        (-15, -8), (-8, -2), (-2, 2), (2, 8), (8, 15),
    ),
    n_bins: int = 300,
) -> "pd.DataFrame | None":
    """Aggregate the per-bin MAE delta into named true-ESP ranges.

    A single net-delta total (as in plot_delta_predictions) can look like a
    clear win while hiding a real loss in the range that matters most — e.g.
    a config can win big at the extremes and lose in the middle, netting
    positive overall. This breaks the same per-bin delta down by range so
    that trade-off is visible directly, instead of requiring a one-off script
    per sweep.

    Returns a DataFrame with one row per range: range label, n_bins, net
    delta (kT/e), mean delta per bin, and % of bins favoring the compared
    model. Bins whose mean true-ESP falls outside every given range are
    dropped (ranges should be contiguous/covering if that's not desired).
    """
    _, bin_true, bin_delta, _ = _load_delta_bins(input_ckpt_dir, compared_ckpt_dir, n_bins)
    if bin_true is None:
        return None

    rows = []
    for lo, hi in ranges:
        mask = (bin_true >= lo) & (bin_true < hi)
        if not mask.any():
            continue
        rows.append({
            "range":              f"[{lo}, {hi})",
            "n_bins":             int(mask.sum()),
            "net_delta_kT_e":     float(bin_delta[mask].sum()),
            "mean_delta_kT_e":    float(bin_delta[mask].mean()),
            "pct_bins_favor_compared": float((bin_delta[mask] > 0).mean() * 100),
        })
    return pd.DataFrame(rows)


def plot_delta_predictions(
    input_ckpt_dir: Path,
    compared_ckpt_dir: Path,
    input_label: str = "baseline",
    compared_label: str = "compared",
    save_path: Path | None = None,
    n_bins: int = 300,
    win_color: str = "#4878b0",
    loss_color: str = "#d65f5f",
) -> float | None:
    """Per-bin MAE-improvement bar chart: compared model vs. input model.

    For each quantile bin of true ESP value, plots:
        mean(|input_error|) − mean(|compared_error|)

    Positive (blue) bars = compared model has lower MAE in that bin.
    Negative (red)  bars = input model has lower MAE in that bin.
    Returns the sum across all bins (positive = compared wins overall).

    See also range_delta_breakdown, which aggregates this same per-bin delta
    into named true-ESP ranges — the single net total here can look like a
    clean win while masking a loss in the range that matters most.
    """
    from matplotlib.patches import Patch

    shared_pids, bin_true, bin_delta, n_vertices = _load_delta_bins(
        input_ckpt_dir, compared_ckpt_dir, n_bins
    )
    if bin_true is None:
        return None

    net      = float(bin_delta.sum())
    pct_wins = float((bin_delta > 0).mean() * 100)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Δ MAE per ESP bin: {compared_label}  vs  {input_label}\n"
        f"({len(shared_pids)} proteins, {n_vertices:,} vertices, {n_bins} bins)",
        fontsize=11,
    )

    bar_width = (bin_true[-1] - bin_true[0]) / n_bins * 0.9
    colors    = np.where(bin_delta >= 0, win_color, loss_color)
    ax.bar(bin_true, bin_delta, width=bar_width, color=colors, zorder=3, edgecolor="none")
    ax.axhline(0, color="k", lw=0.9, alpha=0.7)

    ax.set_xlabel("True ESP (kT/e, sorted)", fontsize=10)
    ax.set_ylabel("|input error| − |compared error|  (kT/e)", fontsize=10)

    legend_elements = [
        Patch(facecolor=win_color,  label=f"{compared_label} wins"),
        Patch(facecolor=loss_color, label=f"{input_label} wins"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    sign = "+" if net >= 0 else ""
    ax.annotate(
        f"Net: {sign}{net:.1f} kT/e  ({pct_wins:.0f}% of bins favor {compared_label})",
        xy=(0.98, 0.96), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )

    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [delta_pred] Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return net
