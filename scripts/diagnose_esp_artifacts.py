"""
scripts/diagnose_esp_artifacts.py

For proteins flagged by audit_esp_artifacts.py, diagnose WHY each artifact
vertex is violated by characterising its geometry:

  - Hypothesis A (inward normal): the surface normal points toward the
    nearest atom; offset goes into the protein interior.
    Signal: dot(normal, atom_center - vertex) > 0,  same atom before/after offset.

  - Hypothesis B (concavity shortcut): the normal is outward, but the offset
    ray crosses into a DIFFERENT atom on the far wall of a pocket or groove.
    Signal: violated atom != nearest atom to vertex; normal dot product < 0.

  - Degenerate: vertex itself is already inside an atom's radius (SES mesh
    error). Signal: vertex-to-nearest-atom dist < atom radius.

Usage:
    python scripts/diagnose_esp_artifacts.py --all
    python scripts/diagnose_esp_artifacts.py --all --output outputs/esp_artifact_diagnosis.csv
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
    coords, radii = [], []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            coords.append((float(fields[5]), float(fields[6]), float(fields[7])))
            radii.append(float(fields[9]))
    return np.array(coords, dtype=np.float32), np.array(radii, dtype=np.float32)


def _diagnose_protein(protein_id: str, data_root: str, normal_offset: float) -> list[dict]:
    p = ProteinPaths(protein_id, Path(data_root))
    if not (p.mesh_path.exists() and p.esp_path.exists() and p.pqr_path.exists()):
        return []

    mesh = np.load(p.mesh_path)
    verts, normals = mesh["verts"], mesh["normals"]

    atom_xyz, atom_radii = _read_pqr_coords_radii(p.pqr_path)
    if len(atom_xyz) == 0:
        return []

    sample_pts = verts + normal_offset * normals

    tree = cKDTree(atom_xyz)

    # Nearest atom to each SAMPLE POINT (violation check)
    sample_dist, sample_nearest = tree.query(sample_pts, k=1)
    violation_mask = sample_dist < atom_radii[sample_nearest]

    if not violation_mask.any():
        return []

    # Nearest atom to each VERTEX (before offset)
    vert_dist, vert_nearest = tree.query(verts, k=1)

    rows = []
    for vi in np.where(violation_mask)[0]:
        n = normals[vi]
        v = verts[vi]
        s_atom = int(sample_nearest[vi])
        v_atom = int(vert_nearest[vi])
        atom_center = atom_xyz[s_atom]

        # Component of normal toward the violated atom
        to_atom = atom_center - v
        to_atom_norm = to_atom / (np.linalg.norm(to_atom) + 1e-9)
        normal_dot = float(np.dot(n, to_atom_norm))

        vertex_inside = bool(vert_dist[vi] < atom_radii[v_atom])
        same_atom = s_atom == v_atom

        # Classification
        if vertex_inside:
            cause = "vertex_inside_atom"         # SES mesh error
        elif normal_dot > 0 and same_atom:
            cause = "inward_normal"              # Hypothesis A
        elif not same_atom:
            cause = "concavity_shortcut"         # Hypothesis B
        else:
            cause = "other"

        rows.append({
            "protein_id":           protein_id,
            "vertex_idx":           int(vi),
            "cause":                cause,
            "penetration_A":        float(atom_radii[s_atom] - sample_dist[vi]),
            "vert_to_nearest_A":    float(vert_dist[vi]),
            "vert_nearest_radius":  float(atom_radii[v_atom]),
            "vertex_inside_atom":   vertex_inside,
            "same_atom":            same_atom,
            "normal_dot_to_atom":   normal_dot,
            "abs_esp":              float(np.abs(np.load(p.esp_path)["esp_verts"][vi])),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why ESP artifact vertices occur (inward normal vs concavity shortcut)."
    )
    add_filter_args(parser)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/esp_artifact_diagnosis.csv")
    args = parser.parse_args()

    data_root     = Path(args.data_root) if args.data_root else get_data_root()
    normal_offset = get_config()["esp_mapping"]["normal_offset"]

    protein_ids = get_protein_ids_from_args(args, data_root)
    print(f"Diagnosing {len(protein_ids):,} proteins  (normal_offset={normal_offset:.2f} Å)")

    results = run_parallel(
        _diagnose_protein,
        [(pid, str(data_root), normal_offset) for pid in protein_ids],
        n_workers=8,
        label="diagnose",
    )

    all_rows = []
    for _, r in results:
        if isinstance(r, list):
            all_rows.extend(r)
        elif isinstance(r, Exception):
            pass

    if not all_rows:
        print("No artifact vertices found.")
        return

    # Tally causes
    from collections import Counter
    cause_counts = Counter(r["cause"] for r in all_rows)
    n = len(all_rows)

    print(f"\nTotal artifact vertices diagnosed: {n}")
    print(f"\nCause breakdown:")
    for cause, count in cause_counts.most_common():
        print(f"  {cause:<25} {count:>5}  ({100*count/n:.1f}%)")

    # Stats by cause
    for cause in cause_counts:
        subset = [r for r in all_rows if r["cause"] == cause]
        pens   = [r["penetration_A"] for r in subset]
        esps   = [r["abs_esp"] for r in subset]
        dots   = [r["normal_dot_to_atom"] for r in subset]
        print(f"\n  [{cause}]")
        print(f"    penetration  mean={np.mean(pens):.3f} Å   max={np.max(pens):.3f} Å")
        print(f"    |ESP|        mean={np.mean(esps):.2f}       max={np.max(esps):.2f}")
        print(f"    normal·atom  mean={np.mean(dots):.3f}       range=[{np.min(dots):.3f}, {np.max(dots):.3f}]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-vertex CSV written to: {output_path}")


if __name__ == "__main__":
    main()
