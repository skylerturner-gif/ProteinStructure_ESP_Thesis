"""
pipelines/model_pipeline.py

Model pipeline for the ProteinStructure ESP project.

Runs the two model-side steps sequentially for a filtered set of proteins:
    6. Build PyG HeteroData graphs (delegates to pipelines/06_build_graphs.py)
    7. Train and evaluate the ESP prediction model (delegates to pipelines/07_train.py)

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


def _run_script(script: str, extra_args: list[str], nproc: int = 1) -> bool:
    """Run a script in the current Python interpreter. Returns True on success."""
    if nproc > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            str(_PROJECT_ROOT / script),
        ] + extra_args
    else:
        cmd = [sys.executable, str(_PROJECT_ROOT / script)] + extra_args
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Model pipeline: build PyG graphs (step 6) then train the ESP model (step 7). "
            "Hyperparameter defaults are read from the model: and training: sections of config.yaml."
        )
    )

    # ── Protein selection (only required when building graphs) ───────────────
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--all",    action="store_true", dest="select_all",
                       help="Build graphs for all proteins with valid metadata.")
    group.add_argument("--filter", action="store_true", dest="use_filter",
                       help="Build graphs for proteins matching the filter criteria.")
    parser.add_argument("--min-sequence-length", type=int,   default=None)
    parser.add_argument("--max-sequence-length", type=int,   default=None)
    parser.add_argument("--min-plddt",           type=float, default=None)
    parser.add_argument("--max-plddt",           type=float, default=None)
    parser.add_argument("--min-surface-area",    type=float, default=None)
    parser.add_argument("--max-surface-area",    type=float, default=None)
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
                        choices=["distance", "attention", "both"],
                        help="Model type. Use 'both' to train distance then attention "
                             "sequentially. Overrides config model.type.")
    parser.add_argument("--epochs", type=int,  default=None,
                        help="Training epochs. Overrides config training.epochs.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR scheduler (default: cosine).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Directory for checkpoints. "
                             "Defaults to <data_root>/../checkpoints/<model>[_suffix]. "
                             "Ignored when --model both (each model uses its own subdir).")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Label appended to the checkpoint directory name, e.g. "
                             "'base' → checkpoints/attention_base. "
                             "Has no effect when --checkpoint-dir is set explicitly.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume training from. "
                             "Ignored when --model both.")
    parser.add_argument("--protein-weighted", action="store_true", default=False,
                        help="Weight MSE equally per protein to reduce large-protein "
                             "dominance in greedy batches.")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="Gradient accumulation steps (1 = disabled).")

    args = parser.parse_args()

    cfg       = get_config()
    data_root = args.data_root or get_data_root()
    model_cfg = cfg.get("model",    {})
    train_cfg = cfg.get("training", {})

    filter_args = _collect_filter_args(args)
    data_root_args = ["--data-root", str(data_root)]

    # ── Validate filter args when graph building is needed ───────────────────
    if not args.skip_graphs and not args.select_all and not args.use_filter:
        parser.error("--all or --filter is required when building graphs (omit with --skip-graphs).")

    # ── Step 6: Build graphs ──────────────────────────────────────────────────
    if not args.skip_graphs:
        print("[model_pipeline] Step 6: Building graphs...")
        graph_args = (
            filter_args
            + data_root_args
            + ["--workers", str(args.workers)]
        )
        if args.force:
            graph_args.append("--force")

        ok = _run_script("pipelines/06_build_graphs.py", graph_args)
        if not ok:
            print("[model_pipeline] Step 6 failed — aborting.")
            sys.exit(1)
        print("[model_pipeline] Step 6: Graphs built.")
    else:
        print("[model_pipeline] Step 6: Skipped (--skip-graphs).")

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    if not args.skip_training:
        try:
            import torch as _torch
            n_gpus = _torch.cuda.device_count()
        except Exception:
            n_gpus = 0

        raw_model = args.model or model_cfg.get("type", "attention")
        model_types = ["distance", "attention"] if raw_model == "both" else [raw_model]

        for model_type in model_types:
            print(f"[model_pipeline] Step 8: Training {model_type} model...")

            # Resolve checkpoint dir: explicit > suffix-derived > default
            if args.checkpoint_dir and len(model_types) == 1:
                ckpt_dir = args.checkpoint_dir
            else:
                base_name = f"{model_type}_{args.suffix}" if args.suffix else model_type
                ckpt_dir  = Path(data_root).parent / "checkpoints" / base_name

            train_args = (
                data_root_args
                + [
                    "--model",               model_type,
                    "--checkpoint-dir",      str(ckpt_dir),
                    "--hidden-dim",          str(model_cfg.get("hidden_dim",           256)),
                    "--n-rbf",               str(model_cfg.get("n_rbf",                16)),
                    "--n-heads",             str(model_cfg.get("n_heads",              4)),
                    "--n-bond-radial-rounds",str(model_cfg.get("n_bond_radial_rounds", 2)),
                    "--n-aq-rounds",         str(model_cfg.get("n_aq_rounds",          3)),
                    "--n-qq-rounds",         str(model_cfg.get("n_qq_rounds",          2)),
                    "--epochs",              str(args.epochs or train_cfg.get("epochs", 100)),
                    "--max-edges-per-batch", str(train_cfg.get("max_edges_per_batch",  2_000_000)),
                    "--lr",                  str(train_cfg.get("lr",                   3e-4)),
                    "--weight-decay",        str(train_cfg.get("weight_decay",         1e-4)),
                    "--pearson-weight",      str(train_cfg.get("pearson_weight",       0.1)),
                    "--clip-grad",           str(train_cfg.get("clip_grad",            1.0)),
                    "--lr-patience",         str(train_cfg.get("lr_patience",          15)),
                ]
            )
            if args.resume and len(model_types) == 1:
                train_args += ["--resume", str(args.resume)]
            if args.protein_weighted or train_cfg.get("protein_weighted", False):
                train_args.append("--protein-weighted")
            accum = args.grad_accum_steps or train_cfg.get("grad_accum_steps", 1)
            if accum and accum > 1:
                train_args += ["--grad-accum-steps", str(accum)]

            ok = _run_script("pipelines/07_train.py", train_args, nproc=max(1, n_gpus))
            if not ok:
                print(f"[model_pipeline] Step 7 ({model_type}) failed — aborting.")
                sys.exit(1)
            print(f"[model_pipeline] Step 7: {model_type} training complete.")
    else:
        print("[model_pipeline] Step 7: Skipped (--skip-training).")


if __name__ == "__main__":
    main()
