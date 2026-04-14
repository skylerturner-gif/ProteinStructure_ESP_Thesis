"""
pipelines/06_build_graphs.py

Pre-build and cache PyG HeteroData graphs for a filtered set of proteins.

Each graph is saved as a .pt file to:
    <data_root>/<protein_id>/graph/<protein_id>_graph.pt

Query nodes are loaded from the canonical query_idx stored in the protein's
ESP .npz (written by 04_sample_esp.py).  This guarantees the same vertex
subset is used across graph building, interpolation evaluation, and training.

Graph node/edge counts are written to the protein metadata JSON so that
DynamicBatchSampler can bin proteins by total edge budget without loading
graphs from disk.

Usage:
    python pipelines/06_build_graphs.py --all
    python pipelines/06_build_graphs.py --filter --min-plddt 70
    python pipelines/06_build_graphs.py --all --force      # rebuild cached
"""

import argparse
from pathlib import Path

import torch

from src.data.dataset import ProteinGraphDataset, write_split_manifest
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
    force: bool,
    log,
) -> str:
    """Build and cache the graph for one protein. Returns "ok", "skip", or "fail"."""
    p          = ProteinPaths(protein_id, data_root)
    graph_path = p.graph_path()

    if graph_path.exists() and not force:
        log.info("[%s] Graph cached — skipping", protein_id)
        return "skip"

    missing = [f for f in [p.pqr_path, p.mesh_path, p.esp_path] if not f.exists()]
    if missing:
        for f in missing:
            log.error("[%s] Missing input: %s", protein_id, f.name)
        return "fail"

    p.ensure_dirs()

    try:
        with timer() as t:
            data = build_graph(protein_id, data_root)

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


def _build_one_worker(protein_id: str, data_root_str: str, force: bool) -> tuple[str, str]:
    """Process-pool-safe wrapper. Returns (protein_id, status)."""
    from pathlib import Path
    from src.utils.config import get_config
    from src.utils.helpers import get_pipeline_logger

    data_root = Path(data_root_str)
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    status    = _build_one(protein_id, data_root, force, log)
    return protein_id, status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build and cache PyG graphs for a filtered set of proteins."
    )
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml.")
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild and overwrite existing cached graphs.",
    )
    parser.add_argument(
        "--resplit", action="store_true",
        help="Regenerate split_manifest.json even if one already exists.",
    )
    parser.add_argument("--train-frac",  type=float, default=0.8)
    parser.add_argument("--val-frac",    type=float, default=0.1)
    parser.add_argument("--split-seed",  type=int,   default=42)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1).",
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
        "Building graphs for %d proteins  force=%s  workers=%d",
        len(protein_ids), args.force, args.workers,
    )

    n_ok = n_skip = n_fail = 0

    if args.workers == 1:
        for protein_id in protein_ids:
            status = _build_one(protein_id, data_root, args.force, log)
            if status == "ok":
                n_ok   += 1; notify(protein_id, "complete", "graph build")
            elif status == "skip":
                n_skip += 1; notify(protein_id, "skipped",  "graph build")
            else:
                n_fail += 1; notify(protein_id, "failed",   "graph build")
    else:
        results = run_parallel(
            _build_one_worker,
            [(pid, str(data_root), args.force) for pid in protein_ids],
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

    # ── Split manifest ────────────────────────────────────────────────────────
    from src.data.dataset import SPLIT_MANIFEST_NAME
    manifest_path = data_root / SPLIT_MANIFEST_NAME

    if manifest_path.exists() and not args.resplit:
        print(f"Split manifest already exists — skipping (use --resplit to regenerate).")
        log.info("Split manifest exists at %s — skipping", manifest_path)
    else:
        # Only proteins that have a successfully built graph are eligible.
        from src.utils.paths import ProteinPaths as _PP
        built_ids = [
            pid for pid in protein_ids
            if _PP(pid, data_root).graph_path().exists()
        ]
        full_ds = ProteinGraphDataset(built_ids, data_root)
        path = write_split_manifest(
            full_ds,
            train=args.train_frac,
            val=args.val_frac,
            seed=args.split_seed,
        )
        log.info("Split manifest written to %s", path)
        print(f"Split manifest written → {path}")


if __name__ == "__main__":
    main()
