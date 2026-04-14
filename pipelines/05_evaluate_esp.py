"""
pipelines/05_evaluate_esp.py

Compute the RBF interpolation baseline for a filtered set of proteins.

For each protein, loads the curvature-sampled query_idx (saved by
04_sample_esp.py), reconstructs the full ESP surface from those sparse
vertices via multiquadric RBF, and compares against the APBS ground truth.
Results are written to metadata as interp_rmse and interp_pearson_r.

This script always runs (no skip-if-evaluated logic) — it is the authoritative
step for setting interp_rmse / interp_pearson_r in protein metadata and must
run after any rebuild of the ESP .npz.

Usage:
    python pipelines/05_evaluate_esp.py --all
    python pipelines/05_evaluate_esp.py --filter --min-sequence-length 100
    python pipelines/05_evaluate_esp.py --filter --min-plddt 70
"""

import argparse
from pathlib import Path

from src.analysis.esp_stats import evaluate_protein
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger, notify


def main():
    parser = argparse.ArgumentParser(
        description="Compute RBF interpolation baseline metrics for a filtered set of proteins."
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

    log.info("Interpolation baseline evaluation for %d proteins", len(protein_ids))

    for protein_id in protein_ids:
        try:
            evaluate_protein(protein_id, data_root, write_metadata=True)
            notify(protein_id, "complete", "evaluate_esp")
        except FileNotFoundError as e:
            log.error("[%s] Missing files: %s", protein_id, e)
            notify(protein_id, "failed", "missing files")
        except Exception as e:
            log.error("[%s] Evaluation failed: %s", protein_id, e)
            notify(protein_id, "failed", "evaluation error")


if __name__ == "__main__":
    main()
