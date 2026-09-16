"""
scripts/analyze_residue_exposure.py

Surveys the dataset's SES meshes to estimate, per standard amino acid type,
how much solvent-excluded surface it typically presents — i.e. whether that
residue tends to sit on the outside (solvent-exposed) or the inside (buried
core) of a folded protein.

Method
------
Each mesh vertex is assigned to its geometrically nearest heavy atom (the
same nearest-atom assignment used in survey_mesh_atom_overlap.py). A vertex's
share of surface area is approximated as ``ses_area / n_verts`` (MSMS
produces a near-uniformly spaced vertex cloud at a fixed density, so this is
a reasonable per-vertex weight without re-deriving triangle areas from the
mesh faces). Per-atom areas are summed into per-residue-instance areas, then
divided by each residue type's theoretical maximum ASA (Tien et al. 2013,
Gly-X-Gly scale) to get a relative solvent accessibility (RSA) per residue
instance. A residue instance is called "buried" if RSA < 0.25 and "exposed"
otherwise — the standard two-state threshold (Rost & Sander 1994).

Usage:
    python scripts/analyze_residue_exposure.py --all --workers 8
    python scripts/analyze_residue_exposure.py --all --workers 8 \\
        --output /home/student/thesis/outputs/residue_exposure.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import RESIDUE_VOCAB
from src.utils.config import get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths

# Theoretical maximum ASA (Å²), Gly-X-Gly tripeptide scale.
# Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO (2013).
# "Maximum Allowed Solvent Accessibilites of Residues in Proteins." PLOS ONE.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

BURIED_RSA_THRESHOLD = 0.25  # Rost & Sander (1994) two-state cutoff


def _read_pqr_atoms(pqr_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return (xyz (N,3), radii (N,), resnames (N,), resseq (N,)) for heavy atoms only."""
    coords, radii, resnames, resseq = [], [], [], []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                radius = float(fields[9])
            except ValueError:
                continue
            if radius <= 0:
                continue  # PARSE united-atom hydrogens — never on the SES
            coords.append((float(fields[5]), float(fields[6]), float(fields[7])))
            radii.append(radius)
            resnames.append(fields[3])
            resseq.append(int(fields[4]))
    return (
        np.array(coords, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        resnames,
        np.array(resseq, dtype=np.int64),
    )


def _survey_protein(protein_id: str, data_root: str) -> list[tuple[str, float]] | None:
    p = ProteinPaths(protein_id, Path(data_root))
    if not (p.mesh_path.exists() and p.pqr_path.exists()):
        return None

    mesh = np.load(p.mesh_path)
    verts = mesh["verts"]
    n_verts = int(mesh["n_verts"])
    if n_verts == 0 or len(verts) == 0:
        return None
    vertex_area = float(mesh["ses_area"]) / n_verts

    xyz, radii, resnames, resseq = _read_pqr_atoms(p.pqr_path)
    if len(xyz) == 0:
        return None

    tree = cKDTree(xyz)
    _, nearest = tree.query(verts, k=1)
    atom_area = np.bincount(nearest, minlength=len(xyz)) * vertex_area

    residue_area: dict[int, float] = {}
    residue_name: dict[int, str] = {}
    for i in range(len(xyz)):
        rid = int(resseq[i])
        residue_area[rid] = residue_area.get(rid, 0.0) + float(atom_area[i])
        residue_name[rid] = resnames[i]

    return [(residue_name[rid], residue_area[rid]) for rid in residue_area]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey per-residue-type SES surface exposure across the dataset."
    )
    add_filter_args(parser)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default="/home/student/thesis/outputs/residue_exposure.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_root   = Path(args.data_root) if args.data_root else get_data_root()
    protein_ids = get_protein_ids_from_args(args, data_root)
    print(f"Surveying {len(protein_ids):,} proteins  (workers={args.workers})")

    results = run_parallel(
        _survey_protein,
        [(pid, str(data_root)) for pid in protein_ids],
        n_workers=args.workers,
        label="exposure",
    )

    errors = [(pid, r) for pid, r in results if isinstance(r, Exception)]
    rows   = [r for _, r in results if isinstance(r, list)]
    n_skip = len(protein_ids) - len(rows) - len(errors)

    if errors:
        print(f"\n{len(errors)} errors (first 5):")
        for pid, exc in errors[:5]:
            print(f"  {pid}: {exc}")
    print(f"Parsed {len(rows)} proteins, skipped {n_skip} (missing mesh/pqr).")

    areas_by_resname: dict[str, list[float]] = {name: [] for name in RESIDUE_VOCAB}
    for protein_rows in rows:
        for resname, area in protein_rows:
            if resname in areas_by_resname:
                areas_by_resname[resname].append(area)

    out_rows = []
    for resname, areas in areas_by_resname.items():
        if not areas:
            continue
        areas_arr = np.array(areas)
        rsa = areas_arr / MAX_ASA[resname]
        out_rows.append({
            "residue":        resname,
            "n_instances":    len(areas_arr),
            "mean_area_A2":   float(areas_arr.mean()),
            "mean_rsa":       float(rsa.mean()),
            "median_rsa":     float(np.median(rsa)),
            "frac_buried":    float((rsa < BURIED_RSA_THRESHOLD).mean()),
            "frac_exposed":   float((rsa >= BURIED_RSA_THRESHOLD).mean()),
        })

    df = pd.DataFrame(out_rows).sort_values("mean_rsa", ascending=False).reset_index(drop=True)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    pd.set_option("display.width", 120)
    print(f"\n{df.round(4).to_string(index=False)}")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
