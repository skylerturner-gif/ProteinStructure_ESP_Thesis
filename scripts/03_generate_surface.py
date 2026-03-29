"""
scripts/03_generate_surface.py

Generate SES mesh surfaces for a filtered set of proteins.

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
        description="Generate SES mesh surfaces for a filtered set of proteins."
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
        p           = ProteinPaths(protein_id, data_root)
        protein_ok  = True
        failed_step = ""

        # PDB mesh
        if p.pdb_mesh_path.exists():
            log.info("[%s] PDB mesh exists — skipping", protein_id)
        elif not p.pdb_path.exists():
            log.error("[%s] PDB missing — skipping PDB mesh", protein_id)
            protein_ok  = False
            failed_step = "pdb missing"
        else:
            try:
                build_mesh(p.pdb_path, protein_id, data_root)
            except Exception as e:
                log.error("[%s] PDB mesh failed: %s", protein_id, e)
                protein_ok  = False
                failed_step = "pdb mesh"

        # PQR mesh
        if p.pqr_mesh_path.exists():
            log.info("[%s] PQR mesh exists — skipping", protein_id)
        elif not p.pqr_path.exists():
            log.error("[%s] PQR missing — skipping PQR mesh", protein_id)
            protein_ok  = False
            failed_step = failed_step or "pqr missing"
        else:
            try:
                build_mesh(p.pqr_path, protein_id, data_root)
            except Exception as e:
                log.error("[%s] PQR mesh failed: %s", protein_id, e)
                protein_ok  = False
                failed_step = failed_step or "pqr mesh"

        if protein_ok:
            notify(protein_id, "complete", "surface generation")
        else:
            notify(protein_id, "failed", failed_step)


if __name__ == "__main__":
    main()