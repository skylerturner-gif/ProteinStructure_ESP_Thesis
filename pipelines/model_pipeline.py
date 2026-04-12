"""
pipelines/model_pipeline.py

Model pipeline for the ProteinStructure ESP project.

Runs the two model-side steps sequentially for a filtered set of proteins:
    7. Build PyG HeteroData graphs (delegates to scripts/07_build_graphs.py)
    8. Train and evaluate the ESP prediction model (delegates to scripts/08_train.py)

Model architecture and training hyperparameters are read from the `model:` and
`training:` sections of config.yaml. The most commonly changed values can be
overridden via CLI flags.

Requires the model training conda environment (PyTorch, PyG).

Usage:
    # Build graphs + train on all proteins using config.yaml defaults:
    python pipelines/model_pipeline.py --all

    # Filter proteins, override model type and epochs:
    python pipelines/model_pipeline.py --filter --min-plddt 70 \\
        --model distance --epochs 50

    # Only build graphs (skip training):
    python pipelines/model_pipeline.py --all --skip-training

    # Only train (graphs already built):
    python pipelines/model_pipeline.py --all --skip-graphs

    # Rebuild graphs + resume training from a checkpoint:
    python pipelines/model_pipeline.py --all --force \\
        --resume checkpoints/attention/latest_model.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect_filter_args(args) -> list[str]:
    """Convert parsed filter args back into CLI flag strings for subprocess calls."""
    out = []
    if getattr(args, "select_all", False):
        out.append("--all")
    elif getattr(args, "use_filter", False):
        out.append("--filter")
        for attr, flag in [
            ("min_sequence_length", "--min-sequence-length"),
            ("max_sequence_length", "--max-sequence-length"),
            ("min_plddt",           "--min-plddt"),
            ("max_plddt",           "--max-plddt"),
            ("min_surface_area",    "--min-surface-area"),
            ("max_surface_area",    "--max-surface-area"),
        ]:
            val = getattr(args, attr, None)
            if val is not None:
                out += [flag, str(val)]
    return out


def _run_script(script: str, extra_args: list[str]) -> bool:
    """Run a script in the current Python interpreter. Returns True on success."""
    cmd = [sys.executable, str(_PROJECT_ROOT / script)] + extra_args
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Model pipeline: build PyG graphs (step 7) then train the ESP model (step 8). "
            "Hyperparameter defaults are read from the model: and training: sections of config.yaml."
        )
    )

    # ── Protein selection ─────────────────────────────────────────────────────
    add_filter_args(parser)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml.")

    # ── Step control ──────────────────────────────────────────────────────────
    parser.add_argument("--skip-graphs",   action="store_true",
                        help="Skip graph building (step 7); assume graphs are already cached.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (step 8); only build graphs.")

    # ── Graph building ────────────────────────────────────────────────────────
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for graph building (default: 1).")
    parser.add_argument("--force",   action="store_true",
                        help="Rebuild cached graphs from scratch.")

    # ── Key training overrides ────────────────────────────────────────────────
    parser.add_argument("--model",  type=str,  default=None,
                        choices=["distance", "attention"],
                        help="Model type. Overrides config model.type.")
    parser.add_argument("--epochs", type=int,  default=None,
                        help="Training epochs. Overrides config training.epochs.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Directory for checkpoints. "
                             "Defaults to <data_root>/../checkpoints/<model>.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume training from.")

    args = parser.parse_args()

    cfg       = get_config()
    data_root = args.data_root or get_data_root()
    model_cfg = cfg.get("model",    {})
    train_cfg = cfg.get("training", {})

    filter_args = _collect_filter_args(args)
    data_root_args = ["--data-root", str(data_root)]

    # ── Step 7: Build graphs ──────────────────────────────────────────────────
    if not args.skip_graphs:
        print("[model_pipeline] Step 7: Building graphs...")
        graph_args = (
            filter_args
            + data_root_args
            + ["--sample-frac", str(train_cfg.get("sample_frac", 0.05))]
            + ["--workers", str(args.workers)]
        )
        if args.force:
            graph_args.append("--force")

        ok = _run_script("scripts/07_build_graphs.py", graph_args)
        if not ok:
            print("[model_pipeline] Step 7 failed — aborting.")
            sys.exit(1)
        print("[model_pipeline] Step 7: Graphs built.")
    else:
        print("[model_pipeline] Step 7: Skipped (--skip-graphs).")

    # ── Step 8: Train ─────────────────────────────────────────────────────────
    if not args.skip_training:
        print("[model_pipeline] Step 8: Training model...")

        model_type = args.model or model_cfg.get("type", "attention")

        train_args = (
            filter_args
            + data_root_args
            + [
                "--model",               model_type,
                "--hidden-dim",          str(model_cfg.get("hidden_dim",           256)),
                "--n-rbf",               str(model_cfg.get("n_rbf",                16)),
                "--n-heads",             str(model_cfg.get("n_heads",              4)),
                "--n-bond-radial-rounds",str(model_cfg.get("n_bond_radial_rounds", 2)),
                "--n-aq-rounds",         str(model_cfg.get("n_aq_rounds",          3)),
                "--n-qq-rounds",         str(model_cfg.get("n_qq_rounds",          2)),
                "--sample-frac",         str(train_cfg.get("sample_frac",          0.05)),
                "--epochs",              str(args.epochs or train_cfg.get("epochs", 100)),
                "--max-edges-per-batch", str(train_cfg.get("max_edges_per_batch",  1_000_000)),
                "--lr",                  str(train_cfg.get("lr",                   3e-4)),
                "--weight-decay",        str(train_cfg.get("weight_decay",         1e-4)),
                "--pearson-weight",      str(train_cfg.get("pearson_weight",       0.1)),
                "--clip-grad",           str(train_cfg.get("clip_grad",            1.0)),
                "--lr-patience",         str(train_cfg.get("lr_patience",          15)),
                "--train-frac",          str(train_cfg.get("train_frac",           0.8)),
                "--val-frac",            str(train_cfg.get("val_frac",             0.1)),
                "--split-seed",          str(train_cfg.get("split_seed",           42)),
            ]
        )
        if args.checkpoint_dir:
            train_args += ["--checkpoint-dir", str(args.checkpoint_dir)]
        if args.resume:
            train_args += ["--resume", str(args.resume)]

        ok = _run_script("scripts/08_train.py", train_args)
        if not ok:
            print("[model_pipeline] Step 8 failed.")
            sys.exit(1)
        print("[model_pipeline] Step 8: Training complete.")
    else:
        print("[model_pipeline] Step 8: Skipped (--skip-training).")


if __name__ == "__main__":
    main()
