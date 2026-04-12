"""
pipelines/full_pipeline.py

Top-level orchestrator — runs the data generation pipeline then the model
pipeline, each optionally inside its own conda environment.

Sub-pipelines:
    data_gen_pipeline.py  — download → PDB2PQR → APBS → mesh → ESP → evaluate
    model_pipeline.py     — build graphs → train ESP prediction model

Conda environment names default to the `environments:` section of config.yaml.
Override them with --data-env / --model-env. If an env name is empty or absent,
the pipeline runs in the current Python interpreter without conda switching.

Usage:
    # Run both pipelines (single conda env):
    python pipelines/full_pipeline.py --id-file data/protein_ids.txt

    # Run both pipelines in separate conda environments:
    python pipelines/full_pipeline.py --id-file data/protein_ids.txt \\
        --data-env protein_esp_data --model-env protein_esp_model

    # Data generation only:
    python pipelines/full_pipeline.py --id-file data/protein_ids.txt --skip-model

    # Model pipeline only (data already generated):
    python pipelines/full_pipeline.py --id-file data/protein_ids.txt --skip-data-gen

    # Pass extra options to each sub-pipeline:
    python pipelines/full_pipeline.py --id-file data/protein_ids.txt \\
        --data-env protein_esp_data --model-env protein_esp_model \\
        --workers 4 --keep-dx \\
        --model attention --epochs 150 --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.config import get_config, get_data_root

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_pipeline(script: Path, extra_args: list[str], env_name: str | None) -> bool:
    """
    Run a pipeline script as a subprocess.

    If env_name is set, wraps the call with `conda run --no-capture-output -n <env>`
    so the script runs inside that conda environment. Returns True on success.
    """
    cmd = [sys.executable, str(script)] + extra_args
    if env_name:
        cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full ESP pipeline orchestrator. "
            "Runs data_gen_pipeline.py then model_pipeline.py, "
            "each optionally in its own conda environment."
        )
    )

    # ── Common ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--id-file", type=Path, required=True,
        help="File of UniProt IDs passed to data_gen_pipeline.py.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data_root from config.yaml (passed to both sub-pipelines).",
    )

    # ── Environment selection ─────────────────────────────────────────────────
    parser.add_argument(
        "--data-env", type=str, default=None,
        help="Conda env for data_gen_pipeline.py. "
             "Defaults to config environments.data_gen (skips conda if empty).",
    )
    parser.add_argument(
        "--model-env", type=str, default=None,
        help="Conda env for model_pipeline.py. "
             "Defaults to config environments.model (skips conda if empty).",
    )

    # ── Step skipping ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-data-gen", action="store_true",
        help="Skip the data generation pipeline.",
    )
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip the model pipeline.",
    )

    # ── Data-gen pass-through ─────────────────────────────────────────────────
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker processes for data_gen_pipeline.py (default: 1).",
    )
    parser.add_argument(
        "--keep-dx", action="store_true",
        help="Keep APBS .dx files on disk (passed to data_gen_pipeline.py).",
    )

    # ── Model pass-through ────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default=None, choices=["distance", "attention"],
        help="Model type. Overrides config model.type (passed to model_pipeline.py).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs. Overrides config training.epochs.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Checkpoint directory (passed to model_pipeline.py).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild cached graphs (passed to model_pipeline.py).",
    )
    parser.add_argument(
        "--skip-graphs", action="store_true",
        help="Skip graph building inside model_pipeline.py.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training inside model_pipeline.py.",
    )
    parser.add_argument(
        "--model-workers", type=int, default=1,
        help="Worker processes for graph building inside model_pipeline.py (default: 1).",
    )

    args = parser.parse_args()

    cfg       = get_config()
    data_root = args.data_root or get_data_root()
    env_cfg   = cfg.get("environments", {})

    # Resolve env names: CLI > config > None (run in current interpreter)
    data_env  = args.data_env  or (env_cfg.get("data_gen") or None)
    model_env = args.model_env or (env_cfg.get("model")    or None)

    data_root_args = ["--data-root", str(data_root)]
    ok = True

    # ── Data generation pipeline ──────────────────────────────────────────────
    if not args.skip_data_gen:
        env_label = f"env='{data_env}'" if data_env else "current env"
        print(f"[full_pipeline] Running data_gen_pipeline.py  ({env_label})")

        data_gen_args = (
            ["--id-file", str(args.id_file)]
            + data_root_args
            + ["--workers", str(args.workers)]
        )
        if args.keep_dx:
            data_gen_args.append("--keep-dx")

        ok = _run_pipeline(
            _PROJECT_ROOT / "pipelines" / "data_gen_pipeline.py",
            data_gen_args,
            data_env,
        )
        if not ok:
            print("[full_pipeline] data_gen_pipeline.py exited with errors.")
    else:
        print("[full_pipeline] Skipping data gen pipeline (--skip-data-gen).")

    # ── Model pipeline ────────────────────────────────────────────────────────
    if not args.skip_model and ok:
        env_label = f"env='{model_env}'" if model_env else "current env"
        print(f"[full_pipeline] Running model_pipeline.py  ({env_label})")

        model_args = ["--all"] + data_root_args

        if args.model:
            model_args += ["--model", args.model]
        if args.epochs is not None:
            model_args += ["--epochs", str(args.epochs)]
        if args.checkpoint_dir:
            model_args += ["--checkpoint-dir", str(args.checkpoint_dir)]
        if args.resume:
            model_args += ["--resume", str(args.resume)]
        if args.force:
            model_args.append("--force")
        if args.skip_graphs:
            model_args.append("--skip-graphs")
        if args.skip_training:
            model_args.append("--skip-training")
        if args.model_workers != 1:
            model_args += ["--workers", str(args.model_workers)]

        ok = _run_pipeline(
            _PROJECT_ROOT / "pipelines" / "model_pipeline.py",
            model_args,
            model_env,
        )
        if not ok:
            print("[full_pipeline] model_pipeline.py exited with errors.")
    elif args.skip_model:
        print("[full_pipeline] Skipping model pipeline (--skip-model).")

    print("[full_pipeline] Done." if ok else "[full_pipeline] Finished with errors.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
