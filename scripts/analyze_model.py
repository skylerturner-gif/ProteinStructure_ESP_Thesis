"""
scripts/analyze_model.py

Post-training analysis for ESP model predictions.

Modes (combine freely):
    --curves          Plot training/val loss, RMSE, Pearson r, and LR vs epoch.
    --parity          Hexbin parity scatter (pred vs true) with marginal
                      histograms — shows over-smoothing as density near pred≈0.
    --sorted-pred     Sort all query vertices by true ESP; overlay predicted
                      values — compression toward zero is immediately visible.
    --delta-vs DIR    Per-ESP-bin MAE-improvement bar chart comparing this run
                      against another checkpoint dir (e.g. the phase baseline).
    --spatial-metrics Compute Moran's I (spatial error autocorrelation) and
                      ESP-error Spearman r per protein.
    --visualize       PyVista three-panel view for test proteins: predicted ESP,
                      APBS ground truth, and absolute error on the full mesh.

All analysis functions live in src/analysis/model_*.py and can be imported
directly for use in notebooks or other scripts.

Usage:
    # All analyses, all test proteins
    python scripts/analyze_model.py --model attention \\
        --curves --parity --sorted-pred --visualize

    # Training curves only, save figures
    python scripts/analyze_model.py --model attention \\
        --curves --save-plots ~/thesis/figures

    # Compare against the phase baseline
    python scripts/analyze_model.py \\
        --checkpoint-dir checkpoints/phase_a/attention_pw05 \\
        --delta-vs checkpoints/baseline_attention --save-plots ~/thesis/figures

    # PyVista for one specific protein
    python scripts/analyze_model.py --model attention \\
        --visualize --protein-id AF-Q16613-F1

    # Explicit checkpoint and data-root
    python scripts/analyze_model.py \\
        --checkpoint-dir /path/to/checkpoints/attention \\
        --data-root /path/to/external_protein_data \\
        --curves --parity --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.model_plots import (
    plot_delta_predictions, plot_training_curves,
    plot_parity_hexbin, plot_sorted_predictions,
)
from src.analysis.model_spatial import compute_spatial_metrics
from src.analysis.model_visualization import visualize_protein
from src.utils.config import get_data_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training analysis: training curves, parity/sorted-prediction "
                    "plots, and PyVista visualisation."
    )

    # ── Location ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", choices=["distance", "attention"], default=None,
        help="Model type — used to infer checkpoint dir if --checkpoint-dir is not set.",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix appended to checkpoint dir name, e.g. 'base' → checkpoints/attention_base. "
             "Ignored when --checkpoint-dir is set explicitly.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Checkpoint directory (contains metrics.csv, test_metrics.json, "
             "test_predictions/).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data_root from config.yaml.",
    )

    # ── Modes ─────────────────────────────────────────────────────────────────
    parser.add_argument("--curves",        action="store_true",
                        help="Plot training curves from metrics.csv.")
    parser.add_argument("--visualize",     action="store_true",
                        help="PyVista visualisation of test proteins.")
    parser.add_argument("--spatial-metrics", action="store_true",
                        help="Compute Moran's I (spatial error autocorrelation) and "
                             "ESP-error Spearman r per protein. Saved to "
                             "test_spatial_metrics.json in the checkpoint dir.")
    parser.add_argument("--parity", action="store_true",
                        help="Hexbin parity scatter (pred vs true) with marginal "
                             "histograms — shows over-smoothing as density near pred≈0.")
    parser.add_argument("--sorted-pred", action="store_true",
                        help="Sort all query vertices by true ESP; overlay predicted "
                             "values — compression toward zero is immediately visible.")
    parser.add_argument("--delta-vs", type=Path, default=None, metavar="CHECKPOINT_DIR",
                        help="Per-ESP-bin MAE-improvement bar chart comparing this run "
                             "against another checkpoint dir (e.g. the phase baseline).")

    # ── Options ───────────────────────────────────────────────────────────────
    parser.add_argument("--protein-id", type=str, default=None,
                        help="Limit --visualize to one protein ID.")
    parser.add_argument("--clim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=None, help="Colour range for PyVista plots (kT/e).")
    parser.add_argument(
        "--reconstruction",
        choices=["multiquadric", "gaussian", "nearest"],
        default="multiquadric",
        help="RBF kernel for full-mesh reconstruction (default: multiquadric).",
    )
    parser.add_argument("--save-plots", type=Path, default=None,
                        help="Save matplotlib figures to this directory instead of "
                             "displaying them interactively.")

    args = parser.parse_args()

    if not (args.curves or args.visualize or args.spatial_metrics
            or args.parity or args.sorted_pred or args.delta_vs):
        parser.error(
            "Specify at least one of --curves, --parity, --sorted-pred, "
            "--delta-vs, --visualize, --spatial-metrics."
        )

    # ── Resolve paths ─────────────────────────────────────────────────────────
    data_root = args.data_root or get_data_root()

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    elif args.model:
        base_name = f"{args.model}_{args.suffix}" if args.suffix else args.model
        ckpt_dir = Path(data_root).parent / "checkpoints" / base_name
    else:
        parser.error("Provide --model or --checkpoint-dir.")

    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return

    # ── Training curves ───────────────────────────────────────────────────────
    if args.curves:
        print("Plotting training curves...")
        plot_training_curves(ckpt_dir, save_dir=args.save_plots, model_name=args.model or "")

    # ── Parity hexbin ─────────────────────────────────────────────────────────
    if args.parity:
        print("Plotting parity hexbin...")
        plot_parity_hexbin(ckpt_dir, save_dir=args.save_plots, model_name=args.model or "")

    # ── Sorted prediction plot ────────────────────────────────────────────────
    if args.sorted_pred:
        print("Plotting sorted predictions...")
        plot_sorted_predictions(ckpt_dir, save_dir=args.save_plots, model_name=args.model or "")

    # ── Per-bin delta comparison ──────────────────────────────────────────────
    if args.delta_vs:
        print(f"Plotting per-bin delta vs {args.delta_vs}...")
        save_path = None
        if args.save_plots is not None:
            prefix = f"{args.model}_" if args.model else ""
            save_path = args.save_plots / f"{prefix}delta_vs_{args.delta_vs.name}.png"
        plot_delta_predictions(
            args.delta_vs, ckpt_dir,
            input_label=args.delta_vs.name,
            compared_label=ckpt_dir.name,
            save_path=save_path,
        )

    # ── Spatial error metrics ─────────────────────────────────────────────────
    if args.spatial_metrics:
        print("Computing spatial error metrics...")
        compute_spatial_metrics(ckpt_dir)

    # ── PyVista visualisation ─────────────────────────────────────────────────
    if args.visualize:
        pred_dir = ckpt_dir / "test_predictions"
        if not pred_dir.exists():
            print(f"  [visualize] test_predictions/ not found at {pred_dir}")
            print("  Run training to completion first.")
            return

        clim = tuple(args.clim) if args.clim else None
        save_dir = args.save_plots

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
            save_path = None
            if save_dir is not None:
                prefix = f"{args.model}_" if args.model else ""
                save_path = save_dir / f"{prefix}{protein_id}_esp.png"
            visualize_protein(
                protein_id, npz_path, data_root,
                clim=clim,
                reconstruction=args.reconstruction,
                save_path=save_path,
            )


if __name__ == "__main__":
    main()
