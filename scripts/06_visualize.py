"""
scripts/06_visualize.py

Visualize ESP surface comparison for a single protein.

Usage:
    python scripts/06_visualize.py AF-Q16613-F1
    python scripts/06_visualize.py AF-Q16613-F1 --clim -5.0 5.0
    python scripts/06_visualize.py AF-Q16613-F1 --data-root /path/to/data
"""

import argparse
from pathlib import Path

from src.analysis.visualization import plot_esp_comparison
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ESP surface comparison for a single protein."
    )
    parser.add_argument("protein_id", help="Protein ID (e.g. AF-Q16613-F1).")
    parser.add_argument("--clim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    clim      = tuple(args.clim) if args.clim else None

    log.info("Visualizing %s", args.protein_id)
    try:
        plot_esp_comparison(args.protein_id, data_root, clim=clim)
    except FileNotFoundError as e:
        log.error("Missing files: %s", e)
        print(f"✗  {args.protein_id}  —  missing sampled files (run scripts 01-04 first)")


if __name__ == "__main__":
    main()