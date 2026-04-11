"""
scripts/07_build_graphs.py

Pre-build and cache PyG HeteroData graphs for all proteins.

Each graph is saved as a .pt file to:
    <data_root>/<protein_id>/graph/<protein_id>_graph_interp.pt

Graph node/edge counts are written to the protein metadata JSON so that
DynamicBatchSampler can bin proteins by total edge budget without loading
graphs from disk.

Usage:
    python scripts/07_build_graphs.py --all
    python scripts/07_build_graphs.py --filter --min-plddt 70
    python scripts/07_build_graphs.py --all --force      # rebuild cached
    python scripts/07_build_graphs.py --all --sample-frac 0.05
"""

import argparse
import os
from pathlib import Path

import torch

from src.data.graph_builder import build_graph
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger, notify, timer
from src.utils.io import update_metadata
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths


def _build_one(
    protein_id: str,
    data_root: Path,
    sample_frac: float,
    force: bool,
    log,
) -> str:
    """
    Build and cache the graph for one protein.

    Returns "ok", "skip", or "fail".
    """
    p = ProteinPaths(protein_id, data_root)
    graph_path = p.graph_path("interp")

    if graph_path.exists() and not force:
        log.info("[%s] Graph cached — skipping", protein_id)
        return "skip"

    missing = [f for f in [p.pqr_path, p.pqr_mesh_path, p.pqr_interp_path] if not f.exists()]
    if missing:
        for f in missing:
            log.error("[%s] Missing input: %s", protein_id, f.name)
        return "fail"

    p.ensure_dirs()

    try:
        with timer() as t:
            data = build_graph(protein_id, data_root, sample_frac=sample_frac)

        torch.save(data, graph_path)

        update_metadata(protein_id, data_root=data_root, data={
            "num_atom_nodes":       int(data["atom"].num_nodes),
            "num_query_nodes":      int(data["query"].num_nodes),
            "num_nodes_total":      int(data.num_nodes),
            "num_bond_edges":       int(data["atom", "bond",   "atom"].num_edges),
            "num_radial_edges":     int(data["atom", "radial", "atom"].num_edges),
            "num_aq_edges":         int(data["atom", "aq",     "query"].num_edges),
            "num_qq_edges":         int(data["query", "qq",    "query"].num_edges),
            "num_edges_total":      int(data.num_edges),
            "time_graph_build_sec": t.rounded,
        })

        log.info(
            "[%s] Graph built: %d nodes  %d edges  (%.2f s)",
            protein_id, data.num_nodes, data.num_edges, t.seconds,
        )
        return "ok"

    except Exception as e:
        log.error("[%s] Graph build failed: %s", protein_id, e)
        return "fail"


def _build_one_worker(protein_id: str, data_root_str: str, sample_frac: float, force: bool) -> tuple[str, str]:
    """
    Process-pool-safe wrapper around _build_one.

    Takes only picklable primitive arguments and creates its own logger
    so it can be submitted to ProcessPoolExecutor.

    Returns (protein_id, status) where status is "ok", "skip", or "fail".
    """
    from pathlib import Path
    from src.utils.config import get_config
    from src.utils.helpers import get_pipeline_logger

    data_root = Path(data_root_str)
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    status    = _build_one(protein_id, data_root, sample_frac, force, log)
    return protein_id, status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build and cache PyG graphs for a filtered set of proteins."
    )
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml.")
    parser.add_argument(
        "--sample-frac", type=float, default=0.05,
        help="Fraction of mesh vertices used as query nodes (default: 0.05).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild and overwrite existing cached graphs.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). "
             "Set >1 to build graphs concurrently across proteins.",
    )
    add_filter_args(parser)
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    protein_ids = get_protein_ids_from_args(args, data_root)
    if not protein_ids:
        log.warning("No proteins selected. Exiting.")
        return

    log.info(
        "Building graphs for %d proteins  sample_frac=%.2f  force=%s  workers=%d",
        len(protein_ids), args.sample_frac, args.force, args.workers,
    )

    n_ok = n_skip = n_fail = 0

    if args.workers == 1:
        # Sequential path — simpler stack traces when debugging.
        for protein_id in protein_ids:
            status = _build_one(protein_id, data_root, args.sample_frac, args.force, log)
            if status == "ok":
                n_ok   += 1; notify(protein_id, "complete", "graph build")
            elif status == "skip":
                n_skip += 1; notify(protein_id, "skipped",  "graph build")
            else:
                n_fail += 1; notify(protein_id, "failed",   "graph build")
    else:
        # Parallel path — one process per protein.
        results = run_parallel(
            _build_one_worker,
            [(pid, str(data_root), args.sample_frac, args.force) for pid in protein_ids],
            n_workers=args.workers,
            label="graphs",
        )
        for protein_id, outcome in results:
            if isinstance(outcome, Exception):
                n_fail += 1
                notify(protein_id, "failed", f"graph build exception: {outcome}")
                log.error("[%s] Worker exception: %s", protein_id, outcome)
                continue
            _, status = outcome
            if status == "ok":
                n_ok   += 1; notify(protein_id, "complete", "graph build")
            elif status == "skip":
                n_skip += 1; notify(protein_id, "skipped",  "graph build")
            else:
                n_fail += 1; notify(protein_id, "failed",   "graph build")

    log.info("Done — ok: %d  skipped: %d  failed: %d", n_ok, n_skip, n_fail)
    print(f"Done — ok: {n_ok}  skipped: {n_skip}  failed: {n_fail}")


if __name__ == "__main__":
    main()
