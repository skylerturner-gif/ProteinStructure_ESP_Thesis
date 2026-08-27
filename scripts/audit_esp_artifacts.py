"""
scripts/audit_esp_artifacts.py

Flags ESP mesh vertices whose offset ESP-sampling point still falls inside
an atom's PQR (van der Waals) radius — a proxy for APBS grid-interpolation
artifacts near point-charge singularities, as opposed to genuine surface
electrostatics.

Background: 04_sample_esp.py samples the APBS grid at each mesh vertex
offset outward `esp_mapping.normal_offset` Å along its normal
(see notebooks/decisions/01_normal_offset_strategy.ipynb — the offset was
chosen to push samples off the SES boundary into solvent). That mitigation
is only partial: on some meshes the offset point still lands inside a
nearby atom's radius, where trilinear interpolation of the ESP grid samples
the steep near-field of a point charge rather than a physically meaningful
solvent-side potential. This script finds every such vertex across the
dataset so the resulting distribution/tail analysis
(notebooks/initial_protein_data_analysis.ipynb) can account for them.

Usage:
    python scripts/audit_esp_artifacts.py --all
    python scripts/audit_esp_artifacts.py --filter --min-plddt 70
    python scripts/audit_esp_artifacts.py --all --workers 8 \\
        --output outputs/esp_artifact_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths


def _read_pqr_coords_radii(pqr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse atom coordinates and PARSE-forcefield radii from a PQR file.

    PQR ATOM line format (space-delimited, PDB2PQR output — no chain column):
        ATOM serial name resname resseq x y z charge radius
    """
    coords, radii = [], []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            radius = float(fields[9])
            if radius <= 0:
                continue
            coords.append((float(fields[5]), float(fields[6]), float(fields[7])))
            radii.append(radius)
    return np.array(coords, dtype=np.float32), np.array(radii, dtype=np.float32)


def _audit_protein(protein_id: str, data_root: str, normal_offset: float) -> dict | None:
    """
    Worker: for every mesh vertex, check whether its offset ESP-sampling
    point falls inside the PQR radius of its nearest atom.

    Returns a per-protein summary dict, or None if required files are
    missing or the PQR has no parseable atoms.
    """
    p = ProteinPaths(protein_id, Path(data_root))
    if not (p.mesh_path.exists() and p.esp_path.exists() and p.pqr_path.exists()):
        return None

    mesh = np.load(p.mesh_path)
    verts, normals = mesh["verts"], mesh["normals"]

    esp = np.load(p.esp_path)
    esp_verts, query_idx = esp["esp_verts"], esp["query_idx"]

    atom_xyz, atom_radii = _read_pqr_coords_radii(p.pqr_path)
    if len(atom_xyz) == 0:
        return None

    # Same offset used in production sampling (src/surface/esp_mapping.py)
    sample_pts = verts + normal_offset * normals

    tree = cKDTree(atom_xyz)
    dist, nearest_atom = tree.query(sample_pts, k=1)
    violation = dist < atom_radii[nearest_atom]

    n_verts     = len(verts)
    n_violation = int(violation.sum())

    is_query = np.zeros(n_verts, dtype=bool)
    is_query[query_idx] = True

    abs_esp = np.abs(esp_verts)
    penetration = atom_radii[nearest_atom] - dist  # positive where inside the atom

    return {
        "protein_id":            protein_id,
        "n_verts":               n_verts,
        "n_violations":          n_violation,
        "violation_frac":        n_violation / n_verts,
        "n_violations_in_query": int((violation & is_query).sum()),
        "max_abs_esp_overall":   float(abs_esp.max()),
        "max_abs_esp_violation": float(abs_esp[violation].max()) if n_violation else float("nan"),
        "max_abs_esp_clean":     float(abs_esp[~violation].max()) if (~violation).any() else float("nan"),
        "max_penetration_A":     float(penetration[violation].max()) if n_violation else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ESP mesh vertices for APBS interpolation artifacts "
                     "(offset sample point falling inside an atom's PQR radius)."
    )
    add_filter_args(parser)
    parser.add_argument("--data-root", type=str, default=None,
                         help="Override data_root from config.yaml.")
    parser.add_argument("--output", type=str, default="outputs/esp_artifact_audit.csv",
                         help="Path to write the per-protein summary CSV.")
    parser.add_argument("--workers", type=int, default=8,
                         help="Parallel worker processes.")
    args = parser.parse_args()

    data_root     = Path(args.data_root) if args.data_root else get_data_root()
    normal_offset = get_config()["esp_mapping"]["normal_offset"]

    protein_ids = get_protein_ids_from_args(args, data_root)
    print(f"Auditing {len(protein_ids):,} proteins  (normal_offset={normal_offset:.2f} Å, "
          f"workers={args.workers})")

    results = run_parallel(
        _audit_protein,
        [(pid, str(data_root), normal_offset) for pid in protein_ids],
        n_workers=args.workers,
        label="audit",
    )

    rows      = [r for _, r in results if isinstance(r, dict)]
    errors    = [(pid, r) for pid, r in results if isinstance(r, Exception)]
    n_skipped = len(protein_ids) - len(rows) - len(errors)

    if errors:
        print(f"\n{len(errors)} proteins raised errors (showing up to 10):")
        for pid, exc in errors[:10]:
            print(f"  {pid}: {exc}")

    if not rows:
        print("No results to write.")
        return

    rows.sort(key=lambda r: -r["violation_frac"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_verts       = sum(r["n_verts"] for r in rows)
    total_violations  = sum(r["n_violations"] for r in rows)
    total_in_query    = sum(r["n_violations_in_query"] for r in rows)
    n_affected        = sum(1 for r in rows if r["n_violations"] > 0)

    print(f"\nProteins audited                                  : {len(rows):,}  "
          f"(skipped: {n_skipped}, errors: {len(errors)})")
    print(f"Total mesh vertices                               : {total_verts:,}")
    print(f"Vertices inside nearest atom's radius (artifacts) : {total_violations:,} "
          f"({100 * total_violations / total_verts:.4f}%)")
    print(f"Proteins with >=1 artifact vertex                 : {n_affected:,} "
          f"({100 * n_affected / len(rows):.1f}%)")
    print(f"Artifact vertices retained as query nodes         : {total_in_query:,}")
    print(f"\nTop 10 proteins by violation fraction:")
    print(f"  {'protein_id':<20} {'violations':>10} {'frac':>8} {'max|ESP| (viol.)':>18}")
    for r in rows[:10]:
        print(f"  {r['protein_id']:<20} {r['n_violations']:>10} "
              f"{r['violation_frac']:>7.2%} {r['max_abs_esp_violation']:>18.1f}")
    print(f"\nPer-protein CSV written to: {output_path}")


if __name__ == "__main__":
    main()
