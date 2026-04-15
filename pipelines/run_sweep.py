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
    """Merge run's feature/model overrides into config.yaml (in-place)."""
    cfg = _load_config()
    cfg.setdefault("features", {}).update(run.get("features", {}))
    cfg.setdefault("model",    {}).update(run.get("model",    {}))
    _write_config(cfg)


# ── Rebuild detection ─────────────────────────────────────────────────────────

_GRAPH_FEATURE_KEYS = ("query_curvature", "query_normal")


def _features_changed(prev: dict | None, curr: dict) -> bool:
    """Return True if any graph-affecting feature flag differs between runs."""
    if prev is None:
        return False
    prev_f = prev.get("features", {})
    curr_f = curr.get("features", {})
    return any(prev_f.get(k) != curr_f.get(k) for k in _GRAPH_FEATURE_KEYS)


# ── Command building ──────────────────────────────────────────────────────────

def _build_cmd(run: dict, filter_args: list[str], rebuild: bool) -> list[str]:
    """Build the model_pipeline.py subprocess command for one run."""
    train = run.get("train", {})
    cmd = [
        sys.executable, str(_PROJECT_ROOT / "pipelines" / "model_pipeline.py"),
    ]
    cmd += filter_args
    cmd += ["--suffix", run["suffix"]]

    if train.get("model"):
        cmd += ["--model", train["model"]]
    if train.get("epochs"):
        cmd += ["--epochs", str(train["epochs"])]
    if train.get("lr_scheduler"):
        cmd += ["--lr-scheduler", train["lr_scheduler"]]

    if rebuild:
        cmd.append("--force")

    return cmd


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

            cmd = _build_cmd(run, filter_args, rebuild)
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

            prev_run = run

    finally:
        if not args.dry_run:
            _CONFIG_PATH.write_text(original_config)
            print("\n[sweep] config.yaml restored.")

    print(f"\n[sweep] All {total - skipped} run(s) complete.")


if __name__ == "__main__":
    main()
