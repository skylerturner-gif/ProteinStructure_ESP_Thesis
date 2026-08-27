"""
scripts/rebuild_graphs_for_ids.py

Force-rebuild cached PyG graphs for an explicit list of protein IDs, reading
the same query_idx and esp_verts that graph_builder.build_graph always reads
from disk. Unlike pipelines/06_build_graphs.py (which selects proteins via
--all/--filter on metadata), this targets an exact ID list — written for
scripts/fix_esp_artifacts.py, whose repaired proteins have a stale cached
graph.pt (query.y was baked in from the pre-fix ESP values) but don't
necessarily match any single metadata filter.

Run in the `pyg_env` conda environment (needs PyTorch + PyTorch Geometric).

Usage:
    conda activate pyg_env
    python scripts/rebuild_graphs_for_ids.py --id-file outputs/esp_artifact_repaired_ids.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import build_graph
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, notify, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths


def _rebuild_one(protein_id: str, data_root: Path, log) -> str:
    """Build and overwrite the cached graph for one protein. Returns "ok" or "fail"."""
    p = ProteinPaths(protein_id, data_root)

    missing = [f for f in [p.pqr_path, p.mesh_path, p.esp_path] if not f.exists()]
    if missing:
        for f in missing:
            log.error("[%s] Missing input: %s", protein_id, f.name)
        return "fail"

    p.ensure_dirs()
    try:
        with timer() as t:
            data = build_graph(protein_id, data_root)
        torch.save(data, p.graph_path())

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
        log.info("[%s] Graph rebuilt: %d nodes  %d edges  (%.2f s)",
                  protein_id, data.num_nodes, data.num_edges, t.seconds)
        return "ok"
    except Exception as e:
        log.error("[%s] Graph rebuild failed: %s", protein_id, e)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force-rebuild cached graphs for an explicit list of protein IDs."
    )
    parser.add_argument("--id-file", type=str, required=True,
                         help="Text file with one protein ID per line.")
    parser.add_argument("--data-root", type=str, default=None,
                         help="Override data_root from config.yaml.")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    protein_ids = [
        line.strip() for line in Path(args.id_file).read_text().splitlines() if line.strip()
    ]
    print(f"Rebuilding {len(protein_ids):,} cached graphs from {args.id_file}")

    n_ok = n_fail = 0
    for protein_id in protein_ids:
        status = _rebuild_one(protein_id, data_root, log)
        if status == "ok":
            n_ok += 1
            notify(protein_id, "complete", "graph rebuild")
        else:
            n_fail += 1
            notify(protein_id, "failed", "graph rebuild")

    print(f"Done — ok: {n_ok}  failed: {n_fail}")
    log.info("Graph rebuild for explicit ID list complete — ok: %d  failed: %d", n_ok, n_fail)


if __name__ == "__main__":
    main()
