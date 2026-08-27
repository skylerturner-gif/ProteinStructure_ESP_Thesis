"""
scripts/fix_esp_artifacts.py

Repairs the ESP interpolation artifacts found by scripts/audit_esp_artifacts.py:
mesh vertices whose offset ESP-sampling point still falls inside a nearby
atom's PQR radius, where the APBS grid represents the unphysical near-field
of a point charge rather than a genuine solvent-side potential.

The fix (src.surface.esp_mapping.resolve_singularities, wired permanently
into sample_esp) repositions each offending sample point along the
atom-to-point vector until it clears the atom's radius, then re-runs
trilinear interpolation at the corrected location — a real value from the
real APBS field, not a clipped or dropped one. Since the .dx grid is not
persisted to disk by default (see config.yaml esp_mapping / electrostatics
docs), fixing an already-processed protein means re-running APBS to
regenerate the grid in memory.

Run in the `protein_esp` conda environment (needs the `apbs` binary and no
PyTorch). Graph caches must be rebuilt separately afterwards in `pyg_env` —
see scripts/rebuild_graphs_for_ids.py — because query.y for any protein
whose flagged vertex was also a query node is now stale.

Usage:
    conda activate protein_esp
    python scripts/fix_esp_artifacts.py --audit-csv outputs/esp_artifact_audit.csv
    python scripts/fix_esp_artifacts.py --audit-csv outputs/esp_artifact_audit.csv \\
        --workers 4 --ram-limit-gb 20

Writes:
    outputs/esp_artifact_repaired_ids.txt — protein IDs successfully repaired,
        for scripts/rebuild_graphs_for_ids.py to pick up in pyg_env.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.esp_stats import evaluate_protein
from src.electrostatics.run_apbs import process_apbs
from src.surface.esp_mapping import sample_esp
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.parallel import run_parallel


def _repair_one(protein_id: str, data_root: str, ram_limit_gb: float) -> str:
    """
    Worker: re-run APBS (grid in memory only) + sample_esp (now with
    singularity resolution) + the RBF baseline eval, for one protein.
    Returns "ok", "apbs_failed", "sample_failed", or "eval_failed".
    """
    data_root = Path(data_root)

    grid_data = process_apbs(protein_id, data_root, keep_dx=False, ram_limit_gb=ram_limit_gb)
    if grid_data is None:
        return "apbs_failed"

    ok = sample_esp(protein_id, data_root, grid_data=grid_data)
    if not ok:
        return "sample_failed"

    try:
        evaluate_protein(protein_id, data_root, write_metadata=True)
    except Exception:
        return "eval_failed"

    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run APBS + fixed ESP sampling for proteins flagged by "
                     "scripts/audit_esp_artifacts.py."
    )
    parser.add_argument("--audit-csv", type=str, default="outputs/esp_artifact_audit.csv",
                         help="Per-protein CSV from audit_esp_artifacts.py.")
    parser.add_argument("--data-root", type=str, default=None,
                         help="Override data_root from config.yaml.")
    parser.add_argument("--output-ids", type=str,
                         default="outputs/esp_artifact_repaired_ids.txt",
                         help="Where to write the list of successfully repaired IDs "
                              "(input for rebuild_graphs_for_ids.py in pyg_env).")
    parser.add_argument("--workers", type=int, default=4,
                         help="Parallel APBS runs. Each APBS solve can be memory-heavy — "
                              "keep this modest relative to available RAM.")
    parser.add_argument("--ram-limit-gb", type=float, default=40.0,
                         help="Per-protein APBS memory cap passed to process_apbs.")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    audit_path = Path(args.audit_csv)
    with open(audit_path) as f:
        reader = csv.DictReader(f)
        affected = [row["protein_id"] for row in reader if int(row["n_violations"]) > 0]

    print(f"Repairing {len(affected):,} proteins flagged in {audit_path}  "
          f"(workers={args.workers}, ram_limit={args.ram_limit_gb} GB)")
    log.info("Repairing %d ESP-artifact proteins", len(affected))

    results = run_parallel(
        _repair_one,
        [(pid, str(data_root), args.ram_limit_gb) for pid in affected],
        n_workers=args.workers,
        label="repair",
    )

    ok_ids: list[str] = []
    counts = {"ok": 0, "apbs_failed": 0, "sample_failed": 0, "eval_failed": 0, "exception": 0}
    for protein_id, outcome in results:
        if isinstance(outcome, Exception):
            counts["exception"] += 1
            log.error("[%s] Repair raised: %s", protein_id, outcome)
            notify(protein_id, "failed", f"esp repair exception: {outcome}")
            continue
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "ok":
            ok_ids.append(protein_id)
            notify(protein_id, "complete", "esp artifact repair")
        else:
            notify(protein_id, "failed", f"esp repair: {outcome}")

    print(f"\nDone — ok: {counts['ok']}  apbs_failed: {counts['apbs_failed']}  "
          f"sample_failed: {counts['sample_failed']}  eval_failed: {counts['eval_failed']}  "
          f"exception: {counts['exception']}")
    log.info("ESP artifact repair complete: %s", counts)

    output_path = Path(args.output_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ok_ids) + ("\n" if ok_ids else ""))
    print(f"\nRepaired protein IDs written to: {output_path}")
    print("Next step (in pyg_env — cached graphs for these proteins are now stale):")
    print(f"  conda activate pyg_env")
    print(f"  python scripts/rebuild_graphs_for_ids.py --id-file {output_path}")


if __name__ == "__main__":
    main()
