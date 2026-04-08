"""
scripts/03_generate_surface.py

Generate PQR SES mesh surfaces for a filtered set of proteins.

Usage:
    python scripts/03_generate_surface.py --all
    python scripts/03_generate_surface.py --filter --min-sequence-length 100
    python scripts/03_generate_surface.py --filter --min-plddt 70 --max-plddt 90
"""

import argparse
from pathlib import Path

from src.surface.mesh import build_mesh
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.paths import ProteinPaths


def main():
    parser = argparse.ArgumentParser(
        description="Generate PQR SES mesh surfaces for a filtered set of proteins."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    add_filter_args(parser)
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    protein_ids = get_protein_ids_from_args(args, data_root)
    if not protein_ids:
        log.warning("No proteins selected. Exiting.")
        return

    log.info("Surface generation for %d proteins", len(protein_ids))

    for protein_id in protein_ids:
        p = ProteinPaths(protein_id, data_root)

        if p.pqr_mesh_path.exists():
            log.info("[%s] PQR mesh exists — skipping", protein_id)
            notify(protein_id, "skipped", "surface generation")
            continue

        if not p.pqr_path.exists():
            log.error("[%s] PQR missing — cannot build mesh", protein_id)
            notify(protein_id, "failed", "pqr missing")
            continue

        try:
            build_mesh(p.pqr_path, protein_id, data_root)
            notify(protein_id, "complete", "surface generation")
        except Exception as e:
            log.error("[%s] PQR mesh failed: %s", protein_id, e)
            notify(protein_id, "failed", "pqr mesh")


if __name__ == "__main__":
    main()
