"""
scripts/05_evaluate.py

Compute ESP prediction metrics for a filtered set of proteins.

Usage:
    python scripts/05_evaluate.py --all
    python scripts/05_evaluate.py --filter --min-sequence-length 100
    python scripts/05_evaluate.py --filter --min-plddt 70
"""

import argparse
from pathlib import Path

from src.analysis.metrics import evaluate_protein
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.paths import ProteinPaths


def main():
    parser = argparse.ArgumentParser(
        description="Compute ESP metrics for a filtered set of proteins."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute metrics even if they already exist.")
    add_filter_args(parser)
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    protein_ids = get_protein_ids_from_args(args, data_root)
    if not protein_ids:
        log.warning("No proteins selected. Exiting.")
        return

    log.info("Evaluation for %d proteins", len(protein_ids))

    for protein_id in protein_ids:
        p = ProteinPaths(protein_id, data_root)
        if not args.force and p.is_evaluated():
            log.info("[%s] Already evaluated — skipping", protein_id)
            notify(protein_id, "skipped", "already evaluated")
            continue

        try:
            evaluate_protein(protein_id, data_root, write_metadata=True)
            notify(protein_id, "complete", "evaluation")
        except FileNotFoundError as e:
            log.error("[%s] Missing files: %s", protein_id, e)
            notify(protein_id, "failed", "missing sampled files")
        except Exception as e:
            log.error("[%s] Evaluation failed: %s", protein_id, e)
            notify(protein_id, "failed", "evaluation error")


if __name__ == "__main__":
    main()