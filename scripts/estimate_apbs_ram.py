"""
scripts/estimate_apbs_ram.py

Estimate peak APBS RAM for each downloaded protein from its CIF bounding box,
then split UniProt IDs into two files for batched pipeline runs.

Algorithm:
    1. Scan data_root for all protein dirs with a .cif file
    2. Read atomic coordinates via gemmi; compute bounding box (Lx, Ly, Lz)
    3. Estimate APBS grid:
           padding = 12.0 Å each side (typical PDB2PQR fine-grid padding)
           spacing = 0.5 Å  (APBS fine-grid spacing)
           dime_n  = next odd integer >= ceil((Ln + 2*padding) / spacing)
           est_bytes = dime_x * dime_y * dime_z * 120
    4. Group by UniProt ID (max RAM across fragments)
    5. Print sorted table, write --small-out and --large-out UniProt ID files

Usage:
    python scripts/estimate_apbs_ram.py
    python scripts/estimate_apbs_ram.py --threshold 5.0 --small-out data/small_ids.txt --large-out data/large_ids.txt
"""

import argparse
import math
import re
from pathlib import Path

import gemmi

from src.utils.config import get_data_root
from src.utils.paths import ProteinPaths

_PADDING_A   = 12.0   # Å padding each side (matches PDB2PQR fine-grid defaults)
_SPACING_A   = 0.5    # Å fine-grid spacing
_BYTES_POINT = 120    # conservative peak RSS per grid point (~15 arrays × 8 bytes)


def _next_odd(n: int) -> int:
    """Return the smallest odd integer >= n."""
    return n if n % 2 == 1 else n + 1


def estimate_ram_from_cif(cif_path: Path) -> float:
    """
    Return estimated peak APBS RAM in GB from a CIF file's atomic bounding box.
    Returns 0.0 if the file cannot be read or has no atoms.
    """
    try:
        struct = gemmi.read_structure(str(cif_path))
    except Exception:
        return 0.0

    xs, ys, zs = [], [], []
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    xs.append(atom.pos.x)
                    ys.append(atom.pos.y)
                    zs.append(atom.pos.z)

    if not xs:
        return 0.0

    lx = max(xs) - min(xs)
    ly = max(ys) - min(ys)
    lz = max(zs) - min(zs)

    dime_x = _next_odd(math.ceil((lx + 2 * _PADDING_A) / _SPACING_A))
    dime_y = _next_odd(math.ceil((ly + 2 * _PADDING_A) / _SPACING_A))
    dime_z = _next_odd(math.ceil((lz + 2 * _PADDING_A) / _SPACING_A))

    return dime_x * dime_y * dime_z * _BYTES_POINT / 1024 ** 3


def _uniprot_from_protein_id(protein_id: str) -> str:
    """Extract UniProt accession from AF-<UNIPROTID>-F1 pattern."""
    m = re.match(r"AF-([A-Z0-9]+)-F\d+", protein_id)
    return m.group(1) if m else protein_id


def main():
    parser = argparse.ArgumentParser(
        description="Estimate APBS RAM from CIF bounding boxes and split ID files."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="GB cutoff between small and large batches (default: 5.0).")
    parser.add_argument("--small-out", type=Path, default=Path("data/small_ids.txt"),
                        help="Output file for UniProt IDs with est. RAM < threshold.")
    parser.add_argument("--large-out", type=Path, default=Path("data/large_ids.txt"),
                        help="Output file for UniProt IDs with est. RAM >= threshold.")
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()

    protein_dirs = sorted([
        d for d in Path(data_root).iterdir()
        if d.is_dir()
    ])

    if not protein_dirs:
        print(f"No protein directories found in {data_root}")
        return

    # Estimate RAM per protein_id
    estimates: dict[str, float] = {}
    missing = 0
    for protein_dir in protein_dirs:
        protein_id = protein_dir.name
        p = ProteinPaths(protein_id, data_root)
        if not p.cif_path.exists():
            missing += 1
            continue
        est_gb = estimate_ram_from_cif(p.cif_path)
        estimates[protein_id] = est_gb

    if missing:
        print(f"  {missing} protein dirs skipped (no .cif file)")

    if not estimates:
        print("No CIF files found — run the download step first.")
        return

    # Group by UniProt ID: use max RAM across fragments
    uniprot_max: dict[str, tuple[float, str]] = {}  # uniprot_id -> (max_gb, protein_id)
    for protein_id, est_gb in estimates.items():
        uid = _uniprot_from_protein_id(protein_id)
        if uid not in uniprot_max or est_gb > uniprot_max[uid][0]:
            uniprot_max[uid] = (est_gb, protein_id)

    # Sort ascending by estimated RAM
    sorted_items = sorted(uniprot_max.items(), key=lambda x: x[1][0])

    # Print table
    small_count = sum(1 for _, (gb, _) in sorted_items if gb < args.threshold)
    large_count = len(sorted_items) - small_count
    print(f"\nAPBS RAM estimates — {len(sorted_items)} UniProt IDs  "
          f"(threshold: {args.threshold} GB  |  small: {small_count}  large: {large_count})\n")
    print(f"  {'UniProt ID':<15} {'Protein ID':<22} {'Est. GB':>8}  Batch")
    print(f"  {'-'*15} {'-'*22} {'-'*8}  -----")
    for uid, (gb, pid) in sorted_items:
        batch = "small" if gb < args.threshold else "LARGE"
        print(f"  {uid:<15} {pid:<22} {gb:>8.2f}  {batch}")
    print()

    # Write output files
    small_ids = [uid for uid, (gb, _) in sorted_items if gb < args.threshold]
    large_ids = [uid for uid, (gb, _) in sorted_items if gb >= args.threshold]

    args.small_out.parent.mkdir(parents=True, exist_ok=True)
    args.large_out.parent.mkdir(parents=True, exist_ok=True)

    args.small_out.write_text("\n".join(small_ids) + "\n")
    args.large_out.write_text("\n".join(large_ids) + "\n")

    print(f"  Wrote {len(small_ids)} IDs → {args.small_out}")
    print(f"  Wrote {len(large_ids)} IDs → {args.large_out}")


if __name__ == "__main__":
    main()
