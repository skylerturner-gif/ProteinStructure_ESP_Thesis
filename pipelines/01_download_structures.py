"""
scripts/01_download_structures.py

Download AlphaFold structures for a list of UniProt IDs.

Downloads all available fragments (F1, F2, ...) for each UniProt ID.

Usage:
    python scripts/01_download_structures.py --id-file data/protein_ids.txt
    python scripts/01_download_structures.py --id-file data/protein_ids.txt --data-root /path/to/data
    python scripts/01_download_structures.py --id-file data/protein_ids.txt --workers 6
"""

import argparse
from pathlib import Path

from src.structure.af_api import download_protein, find_downloaded_protein_ids, read_uniprot_ids
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, notify
from src.utils.parallel import run_parallel


def _download_worker(uniprot_id: str, data_root_str: str) -> bool:
    from pathlib import Path
    from src.structure.af_api import download_protein
    return download_protein(uniprot_id, Path(data_root_str))


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold structures for a list of UniProt IDs."
    )
    parser.add_argument("--id-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel download threads (default: 1).")
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    if not args.id_file.exists():
        log.error("ID file not found: %s", args.id_file)
        return

    all_ids = read_uniprot_ids(args.id_file)
    pending = [uid for uid in all_ids if not find_downloaded_protein_ids(uid, data_root)]
    skipped = len(all_ids) - len(pending)

    if skipped:
        log.info("Skipping %d already-downloaded UniProt IDs.", skipped)
    if not pending:
        log.info("All proteins already downloaded.")
        return

    succeeded, failed = 0, 0

    if args.workers == 1:
        for uid in pending:
            ok = download_protein(uid, data_root)
            if ok:
                notify(uid, "complete", "download")
                succeeded += 1
            else:
                notify(uid, "failed", "download")
                failed += 1
    else:
        results = run_parallel(
            _download_worker,
            [(uid, str(data_root)) for uid in pending],
            n_workers=args.workers,
            label=f"downloading (workers={args.workers})",
            use_threads=True,
        )
        for uid, outcome in results:
            if isinstance(outcome, Exception) or not outcome:
                notify(uid, "failed", "download")
                failed += 1
            else:
                notify(uid, "complete", "download")
                succeeded += 1

    log.info("Download done. %d succeeded, %d failed, %d skipped.",
             succeeded, failed, skipped)


if __name__ == "__main__":
    main()
