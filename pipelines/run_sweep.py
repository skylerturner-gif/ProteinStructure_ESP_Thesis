"""
pipelines/run_sweep.py

Run consecutive training experiments defined in a sweep YAML file.

Each entry in the YAML specifies feature overrides, model overrides, and
training args.  Between runs the script patches config.yaml with the run's
settings, calls model_pipeline.py as a subprocess, then restores the original
config.yaml — even if a run fails or the process is interrupted.

Graph rebuilding is auto-detected: whenever query_curvature or query_normal
change between consecutive runs, --force is added automatically.  You can
override this per-run with `rebuild_graphs: true/false`.

Usage
-----
    # Dry run — print commands without executing
    python pipelines/run_sweep.py sweeps/ablation_example.yaml --all --dry-run

    # Run sweep on all proteins
    nohup python -u pipelines/run_sweep.py sweeps/ablation_example.yaml --all \\
        > nohup_sweep.out 2>&1 &

    # Filter proteins
    python pipelines/run_sweep.py sweeps/ablation_example.yaml \\
        --filter --min-plddt 70

    # Skip the first N runs (resume after partial completion)
    python pipelines/run_sweep.py sweeps/ablation_example.yaml --all --skip 2

    # Auto-run analysis (training curves, parity hexbin, sorted predictions) after each run
    nohup python -u pipelines/run_sweep.py sweeps/ablation_example.yaml --all --analyze \\
        > nohup_sweep.out 2>&1 &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH  = _PROJECT_ROOT / "config.yaml"


# ── Config patching ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _write_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def _patch_config(run: dict) -> None:
    """Merge run's feature/model/training overrides into config.yaml (in-place)."""
    cfg = _load_config()
    cfg.setdefault("features", {}).update(run.get("features", {}))
    cfg.setdefault("model",    {}).update(run.get("model",    {}))
    cfg.setdefault("training", {}).update(run.get("training", {}))
    _write_config(cfg)


# ── Rebuild detection ─────────────────────────────────────────────────────────

_GRAPH_FEATURE_KEYS = ("query_curvature", "query_normal", "true_radial")


def _features_changed(prev: dict | None, curr: dict) -> bool:
    """Return True if any graph-affecting feature flag differs between runs."""
    if prev is None:
        return False
    prev_f = prev.get("features", {})
    curr_f = curr.get("features", {})
    return any(prev_f.get(k) != curr_f.get(k) for k in _GRAPH_FEATURE_KEYS)


# ── Command building ──────────────────────────────────────────────────────────

def _build_cmd(run: dict, filter_args: list[str], rebuild: bool, workers: int = 1,
               phase: str | None = None) -> list[str]:
    """Build the model_pipeline.py subprocess command for one run."""
    train    = run.get("train",    {})
    training = run.get("training", {})
    cmd = [
        sys.executable, str(_PROJECT_ROOT / "pipelines" / "model_pipeline.py"),
    ]
    cmd += filter_args
    cmd += ["--suffix", run["suffix"]]
    if phase:
        cmd += ["--phase", phase]

    if train.get("model"):
        cmd += ["--model", train["model"]]
    if train.get("epochs"):
        cmd += ["--epochs", str(train["epochs"])]
    if train.get("lr_scheduler"):
        cmd += ["--lr-scheduler", train["lr_scheduler"]]

    # protein-weighted loss and EMA are always on (see ESPLoss/Trainer) — no
    # longer forwarded as flags.
    accum = training.get("grad_accum_steps", 1)
    if accum and accum > 1:
        cmd += ["--grad-accum-steps", str(accum)]
    if "early_stopping_patience" in training:
        cmd += ["--early-stopping-patience", str(training["early_stopping_patience"])]
    if training.get("bf16", False):
        cmd.append("--bf16")
    if "ema_decay" in training:
        cmd += ["--ema-decay", str(training["ema_decay"])]
    if "seed" in training:
        cmd += ["--seed", str(training["seed"])]
    if workers > 1:
        cmd += ["--workers", str(workers)]

    if rebuild:
        cmd.append("--force")

    return cmd


# ── Analysis command ─────────────────────────────────────────────────────────

def _build_analysis_cmd(run: dict, phase: str | None = None) -> list[str] | None:
    """Build the analyze_model.py command for a completed run, or None if the
    run doesn't specify a model type (can't resolve checkpoint dir)."""
    model = run.get("train", {}).get("model")
    if not model:
        return None
    suffix   = run["suffix"]
    cfg      = _load_config()
    data_root = Path(cfg["paths"]["data_root"])
    ckpt_base = data_root.parent / "checkpoints" / phase if phase else data_root.parent / "checkpoints"
    plot_base = data_root.parent / "model_eval"  / phase if phase else data_root.parent / "model_eval"
    ckpt_dir  = ckpt_base / f"{model}_{suffix}"
    plot_dir  = plot_base / f"{model}_{suffix}"
    return [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "analyze_model.py"),
        "--checkpoint-dir", str(ckpt_dir),
        "--save-plots",     str(plot_dir),
        "--curves", "--parity", "--sorted-pred", "--spatial-metrics",
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run consecutive training experiments from a sweep YAML."
    )
    parser.add_argument(
        "sweep_file", type=Path,
        help="Path to the sweep YAML file (e.g. sweeps/ablation_example.yaml).",
    )

    # ── Protein selection (forwarded to model_pipeline.py) ────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",    action="store_true", dest="select_all",
                       help="Process all proteins with valid metadata.")
    group.add_argument("--filter", action="store_true", dest="use_filter",
                       help="Filter proteins by the criteria below.")
    parser.add_argument("--min-sequence-length", type=int,   default=None)
    parser.add_argument("--max-sequence-length", type=int,   default=None)
    parser.add_argument("--min-plddt",           type=float, default=None)
    parser.add_argument("--max-plddt",           type=float, default=None)
    parser.add_argument("--min-surface-area",    type=float, default=None)
    parser.add_argument("--max-surface-area",    type=float, default=None)

    parser.add_argument(
        "--skip", type=int, default=0, metavar="N",
        help="Skip the first N runs (useful for resuming a partial sweep).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analyze_model.py --curves --parity --sorted-pred --spatial-metrics "
             "after each successful training run. Plots are saved to "
             "<data_root>/../model_eval/[<phase>/]<model>_<suffix>/.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Graph-build worker processes forwarded to model_pipeline.py (default: 1).",
    )

    args = parser.parse_args()

    # ── Load sweep definition ─────────────────────────────────────────────────
    if not args.sweep_file.exists():
        print(f"[sweep] ERROR: sweep file not found: {args.sweep_file}")
        sys.exit(1)
    with open(args.sweep_file) as f:
        sweep = yaml.safe_load(f)
    runs: list[dict] = sweep.get("runs", [])
    if not runs:
        print("[sweep] ERROR: sweep YAML has no 'runs' entries.")
        sys.exit(1)
    phase: str | None = sweep.get("phase")

    # ── Build filter args to forward ──────────────────────────────────────────
    filter_args: list[str] = []
    if args.select_all:
        filter_args.append("--all")
    elif args.use_filter:
        filter_args.append("--filter")
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
                filter_args += [flag, str(val)]

    # ── Read original config so we can restore it ─────────────────────────────
    original_config = _CONFIG_PATH.read_text()

    total   = len(runs)
    skipped = args.skip
    if skipped:
        print(f"[sweep] Skipping first {skipped} run(s).")

    print(f"[sweep] {total} runs defined in {args.sweep_file.name}  "
          f"({'dry run' if args.dry_run else 'live'})\n")

    pending_analysis: list[subprocess.Popen] = []

    try:
        prev_run: dict | None = runs[skipped - 1] if skipped > 0 else None

        for i, run in enumerate(runs):
            run_num = i + 1

            if i < skipped:
                print(f"[sweep] [{run_num}/{total}] {run['suffix']} — skipped")
                continue

            # Determine if graphs need rebuilding
            explicit = run.get("rebuild_graphs")
            if explicit is not None:
                rebuild = bool(explicit)
                reason  = "explicit"
            else:
                rebuild = _features_changed(prev_run, run)
                reason  = "feature change detected" if rebuild else "features unchanged"

            cmd = _build_cmd(run, filter_args, rebuild, workers=args.workers, phase=phase)
            rebuild_tag = f"  [REBUILD: {reason}]" if rebuild else ""

            print(f"[sweep] [{run_num}/{total}] {run['suffix']}{rebuild_tag}")
            print(f"         {' '.join(cmd)}\n")

            if not args.dry_run:
                _patch_config(run)
                result = subprocess.run(cmd, start_new_session=True)
                if result.returncode != 0:
                    print(
                        f"[sweep] Run {run_num} ({run['suffix']}) failed "
                        f"(exit code {result.returncode}) — aborting sweep."
                    )
                    sys.exit(result.returncode)

                if args.analyze:
                    analysis_cmd = _build_analysis_cmd(run, phase=phase)
                    if analysis_cmd:
                        print(f"[sweep] [{run_num}/{total}] analyzing (background): {' '.join(analysis_cmd)}\n")
                        pending_analysis.append(subprocess.Popen(analysis_cmd))

            prev_run = run

    finally:
        if not args.dry_run:
            _CONFIG_PATH.write_text(original_config)
            print("\n[sweep] config.yaml restored.")

    if pending_analysis:
        print(f"\n[sweep] Waiting for {len(pending_analysis)} background analysis job(s)...")
        for proc in pending_analysis:
            proc.wait()
        print("[sweep] All analysis jobs complete.")

    print(f"\n[sweep] All {total - skipped} run(s) complete.")


if __name__ == "__main__":
    main()
