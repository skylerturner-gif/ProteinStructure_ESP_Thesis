"""
scripts/survey_mesh_atom_overlap.py

Surveys all mesh vertices (before any normal offset) across the dataset to
characterise how many are geometrically embedded inside atom VdW spheres
(re-entrant SES patches) and what their ESP values look like vs. vertices
that are cleanly outside all atoms.

Three vertex categories per protein:
  clean       — vertex outside all atom VdW spheres
  embedded    — vertex inside at least one atom's VdW sphere (re-entrant patch)
  deep        — embedded AND penetration depth > 1.0 Å (likely pathological)

For each category we report ESP value statistics to distinguish physically
meaningful surface potentials from near-charge-singularity artifacts.

Usage:
    python scripts/survey_mesh_atom_overlap.py --all --workers 8
    python scripts/survey_mesh_atom_overlap.py --all --workers 8 \\
        --output outputs/mesh_atom_overlap_survey.csv
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

from src.utils.config import get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths

DEEP_THRESHOLD_A = 1.0  # penetration > this → "deep" category


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


def _survey_protein(protein_id: str, data_root: str) -> dict | None:
    p = ProteinPaths(protein_id, Path(data_root))
    if not (p.mesh_path.exists() and p.esp_path.exists() and p.pqr_path.exists()):
        return None

    mesh = np.load(p.mesh_path)
    verts = mesh["verts"]

    esp_data  = np.load(p.esp_path)
    esp_verts = esp_data["esp_verts"]
    query_idx = esp_data["query_idx"]

    atom_xyz, atom_radii = _read_pqr_coords_radii(p.pqr_path)
    if len(atom_xyz) == 0:
        return None

    tree = cKDTree(atom_xyz)
    dist, nearest = tree.query(verts, k=1)
    penetration = atom_radii[nearest] - dist   # positive = inside atom

    embedded = penetration > 0
    deep     = penetration > DEEP_THRESHOLD_A
    clean    = ~embedded

    is_query = np.zeros(len(verts), dtype=bool)
    is_query[query_idx] = True

    abs_esp = np.abs(esp_verts)

    def _stats(mask: np.ndarray) -> dict:
        if not mask.any():
            return dict(n=0, mean_abs_esp=float("nan"), p50=float("nan"),
                        p95=float("nan"), p99=float("nan"), max_abs_esp=float("nan"),
                        n_query=0)
        vals = abs_esp[mask]
        return dict(
            n            = int(mask.sum()),
            mean_abs_esp = float(vals.mean()),
            p50          = float(np.percentile(vals, 50)),
            p95          = float(np.percentile(vals, 95)),
            p99          = float(np.percentile(vals, 99)),
            max_abs_esp  = float(vals.max()),
            n_query      = int((mask & is_query).sum()),
        )

    s_clean    = _stats(clean)
    s_embedded = _stats(embedded)
    s_deep     = _stats(deep)

    return {
        "protein_id":               protein_id,
        "n_verts":                  len(verts),
        # clean
        "clean_n":                  s_clean["n"],
        "clean_mean_abs_esp":       s_clean["mean_abs_esp"],
        "clean_p50_abs_esp":        s_clean["p50"],
        "clean_p95_abs_esp":        s_clean["p95"],
        "clean_p99_abs_esp":        s_clean["p99"],
        "clean_max_abs_esp":        s_clean["max_abs_esp"],
        # embedded (re-entrant patch, any penetration)
        "embedded_n":               s_embedded["n"],
        "embedded_frac":            s_embedded["n"] / len(verts),
        "embedded_mean_abs_esp":    s_embedded["mean_abs_esp"],
        "embedded_p50_abs_esp":     s_embedded["p50"],
        "embedded_p95_abs_esp":     s_embedded["p95"],
        "embedded_p99_abs_esp":     s_embedded["p99"],
        "embedded_max_abs_esp":     s_embedded["max_abs_esp"],
        "embedded_n_query":         s_embedded["n_query"],
        # deep (penetration > 1 Å)
        "deep_n":                   s_deep["n"],
        "deep_frac":                s_deep["n"] / len(verts),
        "deep_mean_abs_esp":        s_deep["mean_abs_esp"],
        "deep_max_abs_esp":         s_deep["max_abs_esp"],
        "deep_n_query":             s_deep["n_query"],
        # max penetration seen in this protein
        "max_penetration_A":        float(penetration[embedded].max()) if embedded.any() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey mesh vertex / atom VdW overlap and ESP value distributions."
    )
    add_filter_args(parser)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/mesh_atom_overlap_survey.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_root   = Path(args.data_root) if args.data_root else get_data_root()
    protein_ids = get_protein_ids_from_args(args, data_root)
    print(f"Surveying {len(protein_ids):,} proteins  (workers={args.workers})")

    results = run_parallel(
        _survey_protein,
        [(pid, str(data_root)) for pid in protein_ids],
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

    if not rows:
        print("No results.")
        return

    # ── Global aggregates ──────────────────────────────────────────────────────
    total_v        = sum(r["n_verts"]     for r in rows)
    total_embedded = sum(r["embedded_n"]  for r in rows)
    total_deep     = sum(r["deep_n"]      for r in rows)
    total_clean    = sum(r["clean_n"]     for r in rows)
    n_affected     = sum(1 for r in rows if r["embedded_n"] > 0)
    total_emb_q    = sum(r["embedded_n_query"] for r in rows)
    total_deep_q   = sum(r["deep_n_query"]     for r in rows)

    # Collect all ESP values by category for distribution stats
    # (approximated from per-protein stats — exact only for max/n)
    clean_p99s    = [r["clean_p99_abs_esp"]    for r in rows if r["clean_n"] > 0]
    emb_p99s      = [r["embedded_p99_abs_esp"] for r in rows if r["embedded_n"] > 0]
    clean_maxes   = [r["clean_max_abs_esp"]    for r in rows if r["clean_n"] > 0]
    emb_maxes     = [r["embedded_max_abs_esp"] for r in rows if r["embedded_n"] > 0]

    print(f"\n{'='*65}")
    print(f"Proteins surveyed          : {len(rows):,}  "
          f"(skipped: {n_skip}, errors: {len(errors)})")
    print(f"Total mesh vertices        : {total_v:,}")
    print(f"{'='*65}")
    print(f"\nVertex category breakdown:")
    print(f"  clean (outside all atoms): {total_clean:>10,}  "
          f"({100*total_clean/total_v:.3f}%)")
    print(f"  embedded (inside ≥1 atom): {total_embedded:>10,}  "
          f"({100*total_embedded/total_v:.3f}%)")
    print(f"  deep (penetration >1 Å)  : {total_deep:>10,}  "
          f"({100*total_deep/total_v:.4f}%)")
    print(f"\nProteins with ≥1 embedded vertex : {n_affected:,} "
          f"({100*n_affected/len(rows):.1f}%)")
    print(f"\nEmbedded vertices that are query nodes : {total_emb_q:,}")
    print(f"Deep vertices that are query nodes     : {total_deep_q:,}")

    print(f"\nESP |value| statistics (per-protein p99, then max):")
    print(f"  {'category':<12}  {'median p99':>12}  {'p95 of p99':>12}  "
          f"{'median max':>12}  {'p95 of max':>12}")
    for label, p99s, maxes in [
        ("clean",    clean_p99s, clean_maxes),
        ("embedded", emb_p99s,   emb_maxes),
    ]:
        if p99s:
            print(f"  {label:<12}  "
                  f"{np.median(p99s):>12.2f}  "
                  f"{np.percentile(p99s, 95):>12.2f}  "
                  f"{np.median(maxes):>12.2f}  "
                  f"{np.percentile(maxes, 95):>12.2f}")

    print(f"\nTop 10 proteins by embedded fraction:")
    top = sorted(rows, key=lambda r: -r["embedded_frac"])[:10]
    print(f"  {'protein_id':<22} {'embedded':>9} {'frac':>7} "
          f"{'max_pen_Å':>10} {'emb_max|ESP|':>14}")
    for r in top:
        print(f"  {r['protein_id']:<22} {r['embedded_n']:>9} "
              f"{r['embedded_frac']:>7.3%} "
              f"{r['max_penetration_A']:>10.2f} "
              f"{r['embedded_max_abs_esp']:>14.1f}")

    rows.sort(key=lambda r: -r["embedded_frac"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-protein CSV written to: {output_path}")


if __name__ == "__main__":
    main()
