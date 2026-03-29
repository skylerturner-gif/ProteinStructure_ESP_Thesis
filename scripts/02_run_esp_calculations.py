"""
scripts/02_run_esp_calculations.py

Run PDB2PQR and APBS for a filtered set of proteins.

Usage:
    python scripts/02_run_esp_calculations.py --all
    python scripts/02_run_esp_calculations.py --filter --min-sequence-length 100
    python scripts/02_run_esp_calculations.py --filter --min-plddt 70 --max-plddt 90
"""

import argparse
from pathlib import Path

from src.electrostatics.run_pdb2pqr import process_pdb2pqr
from src.electrostatics.run_apbs import process_apbs
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.paths import ProteinPaths


def main():
    parser = argparse.ArgumentParser(
        description="Run PDB2PQR and APBS for a filtered set of proteins."
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

    log.info("ESP calculations for %d proteins", len(protein_ids))

    for protein_id in protein_ids:
        p          = ProteinPaths(protein_id, data_root)
        protein_ok = True
        failed_step = ""

        if p.pqr_path.exists():
            log.info("[%s] PQR exists — skipping PDB2PQR", protein_id)
        elif not p.pdb_path.exists():
            log.error("[%s] PDB missing — skipping", protein_id)
            notify(protein_id, "failed", "pdb missing")
            continue
        else:
            ok = process_pdb2pqr(protein_id, data_root)
            if not ok:
                protein_ok  = False
                failed_step = "pdb2pqr"

        if protein_ok:
            if p.dx_path.exists():
                log.info("[%s] DX exists — skipping APBS", protein_id)
            else:
                ok = process_apbs(protein_id, data_root)
                if not ok:
                    protein_ok  = False
                    failed_step = "apbs"

        if protein_ok:
            notify(protein_id, "complete", "esp calculations")
        else:
            notify(protein_id, "failed", failed_step)


if __name__ == "__main__":
    main()