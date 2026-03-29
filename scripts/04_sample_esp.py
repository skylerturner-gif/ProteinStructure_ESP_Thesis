"""
scripts/04_sample_esp.py

Sample ESP values onto surface meshes for a filtered set of proteins.

Usage:
    python scripts/04_sample_esp.py --all
    python scripts/04_sample_esp.py --filter --min-sequence-length 100
    python scripts/04_sample_esp.py --filter --min-plddt 70 --max-surface-area 5000
"""

import argparse
from pathlib import Path

from src.surface.esp_mapping import sample_esp
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.paths import ProteinPaths


def main():
    parser = argparse.ArgumentParser(
        description="Sample ESP onto surface meshes for a filtered set of proteins."
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

    log.info("ESP sampling for %d proteins", len(protein_ids))

    for protein_id in protein_ids:
        p = ProteinPaths(protein_id, data_root)

        if p.all_sampled_exist():
            log.info("[%s] All sampled files exist — skipping", protein_id)
            notify(protein_id, "skipped", "esp sampling")
            continue

        missing = [f for f in [p.pdb_mesh_path, p.pqr_mesh_path, p.dx_path]
                   if not f.exists()]
        if missing:
            for f in missing:
                log.error("[%s] Missing: %s", protein_id, f.name)
            notify(protein_id, "failed", "missing inputs")
            continue

        ok = sample_esp(protein_id, data_root)
        if ok:
            notify(protein_id, "complete", "esp sampling")
        else:
            notify(protein_id, "failed", "esp sampling")


if __name__ == "__main__":
    main()