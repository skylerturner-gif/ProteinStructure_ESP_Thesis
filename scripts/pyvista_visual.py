"""
scripts/pyvista_visual.py

Interactive PyVista visualization utility for ESP surfaces.

Two modes, combinable freely:

  --ground-truth PROTEIN_ID [...]
      Render the APBS ground-truth ESP on the full mesh for each protein.
      Uses src.analysis.esp_visualization.plot_esp.

  --model-compare PROTEIN_ID [...]
      Render a three-panel comparison: predicted ESP | ground truth | absolute
      error for each protein.  Reads <checkpoint-dir>/test_predictions/.
      Uses src.analysis.model_visualization.visualize_protein.

For both modes, proteins are processed one at a time in sequence.

Usage:
    # Ground-truth surface for one protein (interactive)
    python scripts/pyvista_visual.py --ground-truth AF-Q16613-F1

    # Ground-truth surface for multiple proteins, saved as PNGs
    python scripts/pyvista_visual.py \\
        --ground-truth AF-Q16613-F1 AF-P12345-F1 \\
        --save ~/figures

    # Model comparison for all test proteins
    python scripts/pyvista_visual.py \\
        --model-compare AF-Q16613-F1 AF-P12345-F1 \\
        --checkpoint-dir ~/checkpoints/attention

    # Both modes together, save all output
    python scripts/pyvista_visual.py \\
        --ground-truth AF-Q16613-F1 \\
        --model-compare AF-Q16613-F1 \\
        --checkpoint-dir ~/checkpoints/attention \\
        --save ~/figures

    # Override data root and colour range
    python scripts/pyvista_visual.py \\
        --ground-truth AF-Q16613-F1 \\
        --data-root /path/to/data \\
        --clim -5 5
"""

import argparse
from pathlib import Path

from src.analysis.esp_visualization import plot_esp
from src.analysis.model_visualization import visualize_protein
from src.utils.config import get_data_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyVista ESP surface viewer — ground-truth and/or model comparison."
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ground-truth", nargs="+", metavar="PROTEIN_ID", default=None,
        help="One or more protein IDs to render as ground-truth ESP surfaces.",
    )
    parser.add_argument(
        "--model-compare", nargs="+", metavar="PROTEIN_ID", default=None,
        help="One or more protein IDs to render as predicted vs ground-truth "
             "three-panel comparisons.  Requires --checkpoint-dir.",
    )

    # ── Options ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Checkpoint directory containing test_predictions/. "
             "Required for --model-compare.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data_root from config.yaml.",
    )
    parser.add_argument(
        "--clim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="Colour range for ESP plots in kT/e.",
    )
    parser.add_argument(
        "--reconstruction",
        choices=["multiquadric", "gaussian", "nearest"],
        default="multiquadric",
        help="RBF kernel for model-compare reconstruction (default: multiquadric).",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Save PNG screenshots to this directory instead of showing "
             "interactive windows.",
    )

    args = parser.parse_args()

    if not args.ground_truth and not args.model_compare:
        parser.error("Specify at least one of --ground-truth or --model-compare.")

    if args.model_compare and not args.checkpoint_dir:
        parser.error("--model-compare requires --checkpoint-dir.")

    data_root = args.data_root or get_data_root()
    clim      = tuple(args.clim) if args.clim else None

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    # ── Ground-truth mode ─────────────────────────────────────────────────────
    if args.ground_truth:
        print(f"Ground-truth visualization — {len(args.ground_truth)} protein(s)")
        for protein_id in args.ground_truth:
            print(f"  {protein_id}")
            save_path = args.save / f"{protein_id}_gt.png" if args.save else None
            try:
                plot_esp(protein_id, data_root, clim=clim, save_path=save_path)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")

    # ── Model-compare mode ────────────────────────────────────────────────────
    if args.model_compare:
        pred_dir = args.checkpoint_dir / "test_predictions"
        if not pred_dir.exists():
            print(
                f"[model-compare] test_predictions/ not found at {pred_dir}\n"
                "Run training to completion first (evaluate_test saves predictions)."
            )
        else:
            print(f"Model comparison — {len(args.model_compare)} protein(s)")
            for protein_id in args.model_compare:
                npz = pred_dir / f"{protein_id}_pred.npz"
                if not npz.exists():
                    print(f"  [skip] {npz.name} not found in {pred_dir}")
                    continue
                print(f"  {protein_id}")
                save_path = (
                    args.save / f"{protein_id}_model_compare.png" if args.save else None
                )
                visualize_protein(
                    protein_id, npz, data_root,
                    clim=clim,
                    reconstruction=args.reconstruction,
                    save_path=save_path,
                )


if __name__ == "__main__":
    main()
