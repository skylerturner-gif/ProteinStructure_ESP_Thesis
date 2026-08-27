"""
scripts/retrofit_mesh_singularities.py

Applies the mesh singularity correction to existing cached _mesh.npz files
in-place.  Reads the cached mesh + PQR (no MSMS re-run), pushes any vertex
exceeding GEOM_THRESHOLD penetration outward to the atom surface, recomputes
vertex normals only for proteins where something actually moved, and resaves
the mesh.

After this script: re-run stage 4 (04_sample_esp.py) and stage 6
(06_build_graphs.py) for the same ID file.

Usage:
    python scripts/retrofit_mesh_singularities.py --id-file data/needs_mesh_fix.txt
    python scripts/retrofit_mesh_singularities.py --id-file data/needs_mesh_fix.txt \\
        --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surface.mesh import (
    GEOM_THRESHOLD,
    _compute_vertex_normals,
    atom_coords_radii_from_pqr,
    geom_validity,
    resolve_singularities,
    save_npz_mesh,
)
from src.utils.config import get_data_root
from src.utils.helpers import notify
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths


def _retrofit_protein(protein_id: str, data_root: str, threshold: float) -> dict:
    p = ProteinPaths(protein_id, Path(data_root))

    mesh = np.load(p.mesh_path)
    verts   = mesh["verts"].copy()
    normals = mesh["normals"].copy()
    faces   = mesh["faces"]
    ses_area = float(mesh["ses_area"])

    atom_xyz, atom_radii = atom_coords_radii_from_pqr(p.pqr_path)
    pen, needs_fix = geom_validity(verts, atom_xyz, atom_radii, threshold)

    if not needs_fix.any():
        return {"protein_id": protein_id, "n_corrected": 0, "max_pen_before": float(pen.max())}

    max_pen_before = float(pen[needs_fix].max())
    fixed_subset, n_corrected = resolve_singularities(verts[needs_fix], atom_xyz, atom_radii)
    verts[needs_fix] = fixed_subset
    normals = _compute_vertex_normals(verts, faces)

    # Verify: no vertices should remain above threshold after correction
    pen_after, still_bad = geom_validity(verts, atom_xyz, atom_radii, threshold)
    n_unresolved = int(still_bad.sum())

    # Use a dummy logger (save_npz_mesh requires one)
    import logging
    plog = logging.getLogger(f"retrofit.{protein_id}")
    save_npz_mesh(p.mesh_path, verts, normals, faces, ses_area, plog)

    return {
        "protein_id":    protein_id,
        "n_corrected":   n_corrected,
        "max_pen_before": max_pen_before,
        "max_pen_after": float(pen_after.max()),
        "n_unresolved":  n_unresolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrofit mesh singularity correction to existing _mesh.npz files."
    )
    parser.add_argument("--id-file",   type=str, required=True,
                        help="Text file with one protein ID per line.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=GEOM_THRESHOLD)
    parser.add_argument("--workers",   type=int,   default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else get_data_root()
    id_file   = Path(args.id_file)
    if not id_file.exists():
        print(f"ID file not found: {id_file}")
        sys.exit(1)

    protein_ids = [l.strip() for l in id_file.read_text().splitlines() if l.strip()]
    print(f"Retrofitting {len(protein_ids):,} proteins  "
          f"(threshold={args.threshold:.2f} Å, workers={args.workers})")

    results = run_parallel(
        _retrofit_protein,
        [(pid, str(data_root), args.threshold) for pid in protein_ids],
        n_workers=args.workers,
        label="retrofit",
    )

    rows   = [r for _, r in results if isinstance(r, dict)]
    errors = [(pid, r) for pid, r in results if isinstance(r, Exception)]

    moved       = [r for r in rows if r["n_corrected"] > 0]
    unresolved  = [r for r in rows if r.get("n_unresolved", 0) > 0]

    print(f"\nProteins processed  : {len(rows):,}")
    print(f"Proteins corrected  : {len(moved):,}")
    print(f"Total verts moved   : {sum(r['n_corrected'] for r in moved):,}")
    if unresolved:
        print(f"Proteins with unresolved verts (hit 4 Å cap): {len(unresolved):,}")
        for r in unresolved[:10]:
            print(f"  {r['protein_id']}  unresolved={r['n_unresolved']}  "
                  f"max_pen_after={r['max_pen_after']:.3f} Å")
    if errors:
        print(f"\nErrors: {len(errors)}")
        for pid, exc in errors[:10]:
            print(f"  {pid}: {exc}")

    try:
        notify(f"Mesh retrofit complete: {len(moved)}/{len(rows)} proteins corrected, "
               f"{len(errors)} errors")
    except Exception:
        pass


if __name__ == "__main__":
    main()
