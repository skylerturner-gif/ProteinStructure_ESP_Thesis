"""
scripts/build_dataset_csv.py

Aggregate all per-protein metadata JSONs into a single CSV for dataset
curation and analysis.

Usage:
    python scripts/build_dataset_csv.py
    python scripts/build_dataset_csv.py --data-root /path/to/data
    python scripts/build_dataset_csv.py --output /path/to/output.csv
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_root
from src.utils.filter import get_protein_ids
from src.utils.io import load_metadata

COLUMNS = [
    "protein_id", "uniprot_id", "protein_name", "organism",
    "sequence_length", "plddt_mean", "plddt_median",
    "n_heavy_atoms", "net_charge",
    "n_vertices", "n_vertices_pqr",
    "ses_area", "ses_area_pqr",
    "dipole_magnitude", "dipole_x", "dipole_y", "dipole_z",
    "radius_of_gyration", "asphericity", "acylindricity",
    "hydrophobic_fraction", "charged_fraction", "polar_fraction",
    "volume_A3", "surface_to_volume", "volume_per_atom", "area_per_atom", "normalized_area",
    "esp_min", "esp_max", "esp_mean", "esp_std",
    "interp_pearson_r", "interp_rmse",
    "pearson_r_pqr", "rmse_pqr",
    "pipeline_complete",
]

SUMMARY_COLS = [
    "sequence_length", "plddt_mean", "net_charge",
    "dipole_magnitude", "radius_of_gyration", "asphericity", "normalized_area",
]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _print_summary(rows):
    print()
    print(f"{'Field':<25}  {'n':>6}  {'min':>10}  {'mean':>10}  {'max':>10}")
    print("-" * 68)
    for col in SUMMARY_COLS:
        vals = []
        for r in rows:
            v = r.get(col)
            if v not in (None, ""):
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if not vals:
            continue
        n    = len(vals)
        mean = sum(vals) / n
        print(f"{col:<25}  {n:>6,}  {min(vals):>10.2f}  {mean:>10.2f}  {max(vals):>10.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-protein metadata into a master CSV."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <data_root>/../dataset_summary.csv)",
    )
    parser.add_argument(
        "--exclude", type=Path, default=None,
        help="File of protein IDs to exclude (one per line). "
             "Pass data/comparison_af_ids.txt to keep comparison proteins out of the CSV.",
    )
    args = parser.parse_args()

    data_root = args.data_root or Path(get_data_root())
    out_path  = args.output or data_root.parent / "dataset_summary.csv"

    excluded: set[str] = set()
    if args.exclude and args.exclude.exists():
        excluded = {l.strip() for l in args.exclude.read_text().splitlines() if l.strip()}
        print(f"Excluding {len(excluded):,} proteins from {args.exclude}")

    protein_ids = get_protein_ids(data_root, select_all=True)
    if excluded:
        before = len(protein_ids)
        protein_ids = [pid for pid in protein_ids if pid not in excluded]
        print(f"  Removed {before - len(protein_ids):,} excluded proteins")

    rows = []
    skipped = 0
    for pid in protein_ids:
        try:
            meta = load_metadata(pid, data_root)
        except FileNotFoundError:
            skipped += 1
            continue
        row = {col: _fmt(meta.get(col)) for col in COLUMNS}
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} proteins → {out_path}")
    if skipped:
        print(f"Skipped {skipped:,} proteins (no metadata JSON)")
    _print_summary(rows)


if __name__ == "__main__":
    main()
