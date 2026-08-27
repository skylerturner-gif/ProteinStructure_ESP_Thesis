"""
scripts/check_mesh_singularities.py

Read-only audit of already-downloaded/processed proteins for the mesh
singularity defect (vertices that penetrate too deep into a nearby atom's
VdW sphere — see GEOM_THRESHOLD in src/surface/mesh.py).

This is deliberately standalone and separate from the main pipeline and
from scripts/retrofit_mesh_singularities.py. It exists purely to report
how many already-downloaded structures are affected — it does not write an
ID file, does not modify anything, and does not feed into any fix step.

Meshes generated from here on are corrected automatically at generation
time (src/surface/mesh.py:build_mesh() runs the same check-and-fix inline
before saving), so this script only has anything to find among structures
downloaded/meshed *before* that fix landed. The main pipeline
(01_download_structures.py onward) always skips proteins that already have
a cached mesh, so it will never re-surface or re-fix these on its own —
checking (and, if ever desired, retrofitting) already-downloaded structures
is intentionally kept out of the normal pipeline/sweep workflow, which only
processes incoming structures that haven't been downloaded yet.

If you do want to fix what this script finds, run (separately, as its own
explicit step — not part of any pipeline stage):
    python scripts/survey_mesh_singularities.py --all --workers 8
    python scripts/retrofit_mesh_singularities.py --id-file data/needs_mesh_fix.txt

Usage:
    python scripts/check_mesh_singularities.py --all --workers 8
    python scripts/check_mesh_singularities.py --filter --min-plddt 70
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


def _check_protein(protein_id: str, data_root: str, threshold: float) -> dict | None:
    """Return singularity stats for one protein, or None if not yet meshed."""
    p = ProteinPaths(protein_id, Path(data_root))
    if not (p.mesh_path.exists() and p.pqr_path.exists()):
        return None

    mesh  = np.load(p.mesh_path)
    verts = mesh["verts"]

    atom_xyz, atom_radii = atom_coords_radii_from_pqr(p.pqr_path)
    if len(atom_xyz) == 0:
        return None

    pen, needs_fix = geom_validity(verts, atom_xyz, atom_radii, threshold)

    return {
        "protein_id":  protein_id,
        "n_verts":     len(verts),
        "n_affected":  int(needs_fix.sum()),
        "max_pen_A":   float(pen.max()),
        "affected":    bool(needs_fix.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit: how many already-downloaded structures have "
                    "the mesh singularity defect? Does not modify anything or "
                    "write an ID file — see retrofit_mesh_singularities.py for that."
    )
    add_filter_args(parser)
    parser.add_argument("--data-root", type=str,   default=None)
    parser.add_argument("--threshold", type=float, default=GEOM_THRESHOLD,
                        help=f"Penetration depth threshold in Å (default {GEOM_THRESHOLD}).")
    parser.add_argument("--workers",   type=int,   default=8)
    args = parser.parse_args()

    data_root   = Path(args.data_root) if args.data_root else get_data_root()
    protein_ids = get_protein_ids_from_args(args, data_root)
    print(f"Checking {len(protein_ids):,} already-downloaded proteins  "
          f"(threshold={args.threshold:.2f} Å, workers={args.workers})")

    results = run_parallel(
        _check_protein,
        [(pid, str(data_root), args.threshold) for pid in protein_ids],
        n_workers=args.workers,
        label="check",
    )

    rows   = [r for _, r in results if isinstance(r, dict)]
    errors = [(pid, r) for pid, r in results if isinstance(r, Exception)]
    n_skip = len(protein_ids) - len(rows) - len(errors)

    if errors:
        print(f"\n{len(errors)} errors (first 5):")
        for pid, exc in errors[:5]:
            print(f"  {pid}: {exc}")

    affected = [r for r in rows if r["affected"]]
    clean    = [r for r in rows if not r["affected"]]

    print(f"\nProteins checked : {len(rows):,}  "
          f"(not yet meshed / skipped: {n_skip}, errors: {len(errors)})")
    print(f"Affected         : {len(affected):,}  ({100*len(affected)/max(len(rows),1):.1f}%)")
    print(f"Clean            : {len(clean):,}")

    if affected:
        pens   = [r["max_pen_A"]  for r in affected]
        counts = [r["n_affected"] for r in affected]
        print(f"\nAmong affected proteins:")
        print(f"  Singularity vertices  total={sum(counts):,}  "
              f"mean={np.mean(counts):.1f}  max={max(counts)}")
        print(f"  Max penetration (Å)   mean={np.mean(pens):.3f}  "
              f"median={np.median(pens):.3f}  max={max(pens):.3f}")

        print(f"\nTop 15 by max penetration:")
        top = sorted(affected, key=lambda r: -r["max_pen_A"])[:15]
        print(f"  {'protein_id':<25} {'n_affected':>10} {'max_pen_Å':>10}")
        for r in top:
            print(f"  {r['protein_id']:<25} {r['n_affected']:>10} {r['max_pen_A']:>10.3f}")

        print(
            f"\nNo files were modified and no ID list was written. To fix these, "
            f"run scripts/survey_mesh_singularities.py and "
            f"scripts/retrofit_mesh_singularities.py explicitly."
        )


if __name__ == "__main__":
    main()
