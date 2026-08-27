"""
scripts/survey_mesh_singularities.py

Scans cached _mesh.npz files + PQR files to find proteins whose mesh
contains vertices that exceed the GEOM_THRESHOLD penetration depth into a
nearby atom's VdW sphere.  These are the proteins that need the singularity-
correction retrofit before ESP re-sampling.

No MSMS re-run required — reads only the cached mesh and PQR.

Writes a plain-text ID file (one protein per line) suitable for passing
directly to retrofit_mesh_singularities.py and the stage 4/6 pipeline scripts.

Usage:
    python scripts/survey_mesh_singularities.py --all --workers 8
    python scripts/survey_mesh_singularities.py --all \\
        --output data/needs_mesh_fix.txt --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surface.mesh import GEOM_THRESHOLD, atom_coords_radii_from_pqr, geom_validity
from src.utils.config import get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths


def _survey_protein(protein_id: str, data_root: str, threshold: float) -> dict | None:
    p = ProteinPaths(protein_id, Path(data_root))
    if not (p.mesh_path.exists() and p.pqr_path.exists()):
        return None

    mesh = np.load(p.mesh_path)
    verts = mesh["verts"]

    atom_xyz, atom_radii = atom_coords_radii_from_pqr(p.pqr_path)
    if len(atom_xyz) == 0:
        return None

    pen, needs_fix = geom_validity(verts, atom_xyz, atom_radii, threshold)

    return {
        "protein_id":    protein_id,
        "n_verts":       len(verts),
        "n_needs_fix":   int(needs_fix.sum()),
        "max_pen_A":     float(pen.max()),
        "needs_fix":     bool(needs_fix.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey which proteins need mesh singularity correction."
    )
    add_filter_args(parser)
    parser.add_argument("--data-root",  type=str,   default=None)
    parser.add_argument("--output",     type=str,   default="data/needs_mesh_fix.txt",
                        help="Output file for protein IDs needing retrofit.")
    parser.add_argument("--threshold",  type=float, default=GEOM_THRESHOLD,
                        help=f"Penetration depth threshold in Å (default {GEOM_THRESHOLD}).")
    parser.add_argument("--workers",    type=int,   default=8)
    args = parser.parse_args()

    data_root   = Path(args.data_root) if args.data_root else get_data_root()
    protein_ids = get_protein_ids_from_args(args, data_root)
    print(f"Surveying {len(protein_ids):,} proteins  "
          f"(threshold={args.threshold:.2f} Å, workers={args.workers})")

    results = run_parallel(
        _survey_protein,
        [(pid, str(data_root), args.threshold) for pid in protein_ids],
        n_workers=args.workers,
        label="survey",
    )

    rows   = [r for _, r in results if isinstance(r, dict)]
    errors = [(pid, r) for pid, r in results if isinstance(r, Exception)]
    n_skip = len(protein_ids) - len(rows) - len(errors)

    if errors:
        print(f"\n{len(errors)} errors (first 5):")
        for pid, exc in errors[:5]:
            print(f"  {pid}: {exc}")

    needs_fix = [r for r in rows if r["needs_fix"]]
    clean     = [r for r in rows if not r["needs_fix"]]

    print(f"\nProteins surveyed   : {len(rows):,}  (skipped: {n_skip}, errors: {len(errors)})")
    print(f"Need singularity fix: {len(needs_fix):,}  ({100*len(needs_fix)/max(len(rows),1):.1f}%)")
    print(f"Already clean       : {len(clean):,}")

    if needs_fix:
        pens = [r["max_pen_A"] for r in needs_fix]
        counts = [r["n_needs_fix"] for r in needs_fix]
        print(f"\nAmong proteins needing fix:")
        print(f"  Singularity vertices  total={sum(counts):,}  "
              f"mean={np.mean(counts):.1f}  max={max(counts)}")
        print(f"  Max penetration (Å)   mean={np.mean(pens):.3f}  "
              f"median={np.median(pens):.3f}  max={max(pens):.3f}")

        print(f"\nTop 15 by max penetration:")
        top = sorted(needs_fix, key=lambda r: -r["max_pen_A"])[:15]
        print(f"  {'protein_id':<25} {'n_fix':>7} {'max_pen_Å':>10}")
        for r in top:
            print(f"  {r['protein_id']:<25} {r['n_needs_fix']:>7} {r['max_pen_A']:>10.3f}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fix_ids = sorted(r["protein_id"] for r in needs_fix)
        output_path.write_text("\n".join(fix_ids) + "\n")
        print(f"\nID list written to: {output_path}  ({len(fix_ids)} proteins)")


if __name__ == "__main__":
    main()
