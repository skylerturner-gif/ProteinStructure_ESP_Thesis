"""
scripts/01_download_structures.py

Download AlphaFold structures for a list of UniProt IDs.

Usage:
    python scripts/01_download_structures.py --id-file data/protein_ids.txt
    python scripts/01_download_structures.py --id-file data/protein_ids.txt --data-root /path/to/data
"""

import argparse
from pathlib import Path

from src.structure.af_api import download_structures, find_downloaded_protein_id, read_uniprot_ids
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.paths import ProteinPaths


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold structures for a list of UniProt IDs."
    )
    parser.add_argument("--id-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    if not args.id_file.exists():
        log.error("ID file not found: %s", args.id_file)
        return

    all_ids = read_uniprot_ids(args.id_file)
    pending = [uid for uid in all_ids if not find_downloaded_protein_id(uid, data_root)]
    skipped = len(all_ids) - len(pending)

    if skipped:
        log.info("Skipping %d already-downloaded proteins.", skipped)
    if not pending:
        log.info("All proteins already downloaded.")
        return

    tmp_id_file = data_root / "_pending_ids.txt"
    tmp_id_file.write_text("\n".join(pending))

    try:
        results = download_structures(id_file=tmp_id_file, data_root=data_root)
    finally:
        tmp_id_file.unlink(missing_ok=True)

    for uid in results["success"]:
        notify(uid, "complete", "download")
    for uid in results["failed"]:
        notify(uid, "failed", "download")

    log.info("Download done. %d succeeded, %d failed, %d skipped.",
             len(results["success"]), len(results["failed"]), skipped)


if __name__ == "__main__":
    main()