"""
scripts/repair_graph_stats.py

Repair graph-build-stage metadata fields (num_atom_nodes, num_query_nodes,
num_nodes_total, num_bond_edges, num_radial_edges, num_aq_edges,
num_qq_edges, num_edges_total) corrupted by the resample_query_density.py /
rebuild_graphs_for_ids.py symlink bug: that (since-deleted) mesh-density
experiment symlinked metadata.json from an isolated dest-root back to the
source data_root, so its graph-rebuild step's update_metadata() calls wrote
graph stats through the symlink into the shared source metadata instead of
the isolated dest-root. The actual cached graph.pt files were never
touched by that bug — only these descriptive fields drifted. This script
recomputes them directly from each affected protein's existing (correct)
graph.pt; no graph rebuild is needed.

Detection is via src.utils.io.is_graph_stats_contaminated() — live-scans
metadata rather than trusting any stale ID list.

Run in the `pyg_env` conda environment (needs PyTorch + PyTorch Geometric
to load .pt files).

Usage:
    conda activate pyg_env
    python scripts/repair_graph_stats.py --dry-run
    python scripts/repair_graph_stats.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.io import is_graph_stats_contaminated, load_metadata, update_metadata
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths

_FIELD_KEYS = [
    "num_atom_nodes", "num_query_nodes", "num_nodes_total",
    "num_bond_edges", "num_radial_edges", "num_aq_edges",
    "num_qq_edges", "num_edges_total",
]


def _find_contaminated(data_root: Path, restrict_to: set[str] | None) -> list[str]:
    """Live-scan protein dirs under data_root for the contamination signature."""
    ids = []
    for protein_dir in sorted(data_root.iterdir()):
        if not protein_dir.is_dir():
            continue
        if restrict_to is not None and protein_dir.name not in restrict_to:
            continue
        meta_path = protein_dir / f"{protein_dir.name}_metadata.json"
        if not meta_path.exists():
            continue
        meta = load_metadata(protein_dir.name, data_root)
        if is_graph_stats_contaminated(meta):
            ids.append(protein_dir.name)
    return ids


def _recompute_fields(graph_path: Path) -> dict:
    """Load a cached graph.pt and recompute the 8 graph-stat metadata fields."""
    data = torch.load(graph_path, weights_only=False)
    return {
        "num_atom_nodes":   int(data["atom"].num_nodes),
        "num_query_nodes":  int(data["query"].num_nodes),
        "num_nodes_total":  int(data.num_nodes),
        "num_bond_edges":   int(data["atom", "bond",   "atom"].num_edges),
        "num_radial_edges": int(data["atom", "radial", "atom"].num_edges),
        "num_aq_edges":     int(data["atom", "aq",     "query"].num_edges),
        "num_qq_edges":     int(data["query", "qq",    "query"].num_edges),
        "num_edges_total":  int(data.num_edges),
    }


def _repair_one(protein_id: str, data_root: Path, dry_run: bool, log) -> tuple[str, dict | None]:
    """Repair one protein's graph-stat metadata. Returns (status, audit_row)."""
    p = ProteinPaths(protein_id, data_root)
    if not p.graph_path().exists():
        log.error("[%s] No cached graph.pt — cannot repair", protein_id)
        return "fail", None

    meta = load_metadata(protein_id, data_root)
    old_fields = {k: meta.get(k) for k in _FIELD_KEYS}

    try:
        new_fields = _recompute_fields(p.graph_path())
    except Exception as e:
        log.error("[%s] Failed to load graph.pt: %s", protein_id, e)
        return "fail", None

    if new_fields["num_query_nodes"] != meta.get("n_query_nodes"):
        log.warning(
            "[%s] Recomputed num_query_nodes=%d still disagrees with n_query_nodes=%s "
            "— writing recomputed value anyway, flag for manual review",
            protein_id, new_fields["num_query_nodes"], meta.get("n_query_nodes"),
        )

    if not dry_run:
        update_metadata(protein_id, data_root=data_root, data=new_fields)

    log.info("[%s] Repaired graph stats: %s", protein_id, new_fields)
    audit_row = {"protein_id": protein_id}
    for k in _FIELD_KEYS:
        audit_row[f"{k}_old"] = old_fields[k]
        audit_row[f"{k}_new"] = new_fields[k]
    return "ok", audit_row


def _repair_one_worker(
    protein_id: str, data_root_str: str, dry_run: bool
) -> tuple[str, dict | None]:
    """Process-pool-safe wrapper. Returns (status, audit_row) — no logger argument."""
    from pathlib import Path as _Path

    from src.utils.config import get_config as _get_config
    from src.utils.helpers import get_pipeline_logger as _get_pipeline_logger

    data_root = _Path(data_root_str)
    log       = _get_pipeline_logger(_Path(_get_config()["paths"]["log_file"]))
    return _repair_one(protein_id, data_root, dry_run, log)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair graph-build-stage metadata fields corrupted by the "
                     "resample_query_density.py / rebuild_graphs_for_ids.py symlink bug."
    )
    parser.add_argument("--data-root", type=Path, default=None,
                         help="Override data_root from config.yaml.")
    parser.add_argument(
        "--id-file", type=Path, default=None,
        help="Restrict to protein IDs in this file (one per line), still "
             "skipping any that are not actually contaminated. Default: "
             "live-scan every protein under data_root.",
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="Detect and report without writing any changes.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). Recommend "
             "8 for a full repair run (~2,700 proteins, ~1.3s per graph.pt "
             "load — serial takes ~1 hour, --workers 8 takes ~10-15 min).",
    )
    parser.add_argument("--audit-csv", type=Path,
                         default=Path("/home/student/thesis/outputs/graph_stats_repair_audit.csv"),
                         help="Where to write the before/after audit CSV.")
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    restrict_to = None
    if args.id_file is not None:
        restrict_to = {
            line.strip() for line in args.id_file.read_text().splitlines() if line.strip()
        }

    print(f"Scanning {data_root} for contaminated graph-stat metadata...")
    protein_ids = _find_contaminated(data_root, restrict_to)
    print(f"Found {len(protein_ids):,} contaminated proteins"
          f"{' (dry run — no changes will be written)' if args.dry_run else ''}")
    if not protein_ids:
        print("Nothing to repair.")
        return

    n_ok = n_fail = 0
    audit_rows = []

    if args.workers == 1:
        for protein_id in protein_ids:
            status, audit_row = _repair_one(protein_id, data_root, args.dry_run, log)
            if status == "ok":
                n_ok += 1
                if audit_row:
                    audit_rows.append(audit_row)
                notify(protein_id, "complete", "graph stats repair")
            else:
                n_fail += 1
                notify(protein_id, "failed", "graph stats repair")
    else:
        results = run_parallel(
            _repair_one_worker,
            [(pid, str(data_root), args.dry_run) for pid in protein_ids],
            n_workers=args.workers,
            label="repair",
        )
        for protein_id, outcome in results:
            if isinstance(outcome, Exception):
                n_fail += 1
                notify(protein_id, "failed", f"graph stats repair exception: {outcome}")
                log.error("[%s] Worker exception: %s", protein_id, outcome)
                continue
            status, audit_row = outcome
            if status == "ok":
                n_ok += 1
                if audit_row:
                    audit_rows.append(audit_row)
                notify(protein_id, "complete", "graph stats repair")
            else:
                n_fail += 1
                notify(protein_id, "failed", "graph stats repair")

    print(f"Done — ok: {n_ok}  failed: {n_fail}")
    log.info("Graph stats repair complete — ok: %d  failed: %d", n_ok, n_fail)

    if audit_rows and not args.dry_run:
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["protein_id"] + [
            f"{k}_{suffix}" for k in _FIELD_KEYS for suffix in ("old", "new")
        ]
        with open(args.audit_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)
        print(f"Audit trail written to {args.audit_csv}")


if __name__ == "__main__":
    main()
