"""
scripts/compute_seed_agreement.py

Seed-conformation pilot, phase B3 core computation: for each of the 10
pilot proteins, measures how much the 5 AlphaFold seed conformations
agree with each other -- both structurally (CA-RMSD after rigid
alignment) and in their predicted ESP field (cross-seed RBF-interpolated
Pearson r/RMSE) -- and joins that against each seed's actual model error
(scripts/eval_seed_conformations.py's output). This is the data
notebooks/seed_conformation_analysis.ipynb needs to test the central
question: does higher inter-seed agreement (AlphaFold is "confident"
about this protein's structure) predict lower/more consistent model
error across seeds?

Methodology:
  - seed000 of each parent protein is the reference. For each other seed,
    a rigid-body (rotation + translation) alignment is fit on CA atoms
    via the Kabsch algorithm, giving both a structural CA-RMSD and the
    transform needed to bring that seed's whole surface into the
    reference frame.
  - ESP-field agreement can't be read off vertex-to-vertex (each seed has
    its own independently-generated SES mesh, different vertex count and
    topology) -- so each non-reference seed's full ESP field is refit as
    an RBF interpolator (scipy RBFInterpolator, same
    multiquadric-kernel/epsilon convention as
    src/surface/esp_mapping.py::rbf_reconstruct) in the now-aligned
    frame, then evaluated at the reference seed's own vertex positions.
    That gives a same-length, directly-comparable field to correlate
    against the reference seed's actual ESP.

Run in the `pyg_env` conda environment (scipy; no torch needed, but kept
in the same env as the rest of this session's analysis scripts).

Usage:
    conda activate pyg_env
    python -u scripts/compute_seed_agreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surface.esp_mapping import _rbf_epsilon
from src.utils.config import get_data_root
from src.utils.paths import ProteinPaths

MANIFEST_CSV = Path("/home/student/thesis/outputs/seed_conformation_manifest.csv")
EVAL_SUMMARY_CSV = Path("/home/student/thesis/outputs/seed_conformation_eval_summary.csv")
OUT_CSV = Path("/home/student/thesis/outputs/seed_conformation_agreement.csv")


def _read_ca_coords(pqr_path: Path) -> np.ndarray:
    """(N_res, 3) CA atom coordinates, in resid order."""
    coords = []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line.split()[2] == "CA":
                fields = line.split()
                coords.append([float(fields[5]), float(fields[6]), float(fields[7])])
    return np.array(coords, dtype=np.float64)


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rigid-body (rotation + translation) alignment of `mobile` onto `target`
    via the Kabsch algorithm. Returns (R, t, rmsd) such that
    mobile @ R.T + t ~= target, and rmsd is the post-alignment CA-RMSD.
    """
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    H = mobile_c.T @ target_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = target.mean(axis=0) - R @ mobile.mean(axis=0)
    aligned = (R @ mobile.T).T + t
    rmsd = float(np.sqrt(((aligned - target) ** 2).sum(axis=1).mean()))
    return R, t, rmsd


def _esp_field_agreement(
    ref_verts: np.ndarray, ref_esp: np.ndarray,
    other_query_pos: np.ndarray, other_query_esp: np.ndarray,
    R: np.ndarray, t: np.ndarray,
) -> tuple[float, float]:
    """
    Bring `other`'s ESP field into the reference frame via (R, t), fit an
    RBF interpolator on its curvature-sampled query-node subset (same
    sparse-input convention as src/surface/esp_mapping.py::rbf_reconstruct
    -- fitting on the full dense vertex set instead produces near-coplanar
    local neighborhoods and a singular RBF system), evaluate at the
    reference's own vertex positions, and return (pearson_r, rmse) against
    the reference's own ESP -- a same-length, directly comparable pair.
    """
    other_query_aligned = (R @ other_query_pos.T).T + t
    eps = _rbf_epsilon(other_query_aligned)
    rbf = RBFInterpolator(
        other_query_aligned, other_query_esp.astype(np.float64),
        kernel="multiquadric", epsilon=eps, neighbors=50,
    )
    other_at_ref = rbf(ref_verts.astype(np.float64))
    r = float(np.corrcoef(ref_esp, other_at_ref)[0, 1])
    rmse = float(np.sqrt(np.mean((ref_esp - other_at_ref) ** 2)))
    return r, rmse


def main() -> None:
    data_root = get_data_root()
    manifest = pd.read_csv(MANIFEST_CSV)
    eval_summary = pd.read_csv(EVAL_SUMMARY_CSV)

    parent_ids = sorted(manifest["parent_protein_id"].unique())
    print(f"{len(parent_ids)} parent proteins")

    rows = []
    for parent_id in parent_ids:
        sub = manifest[manifest["parent_protein_id"] == parent_id].sort_values("seed_index")
        seed_ids = sub["protein_id"].tolist()
        pilot_group = sub["pilot_group"].iloc[0]
        print(f"\n=== {parent_id} ({pilot_group}) -- {len(seed_ids)} seeds ===")

        ref_id = seed_ids[0]
        ref_paths = ProteinPaths(ref_id, data_root)
        ref_ca = _read_ca_coords(ref_paths.pqr_path)
        ref_esp_npz = np.load(ref_paths.esp_path)
        ref_verts, ref_esp = ref_esp_npz["verts"].astype(np.float64), ref_esp_npz["esp_verts"].astype(np.float64)

        ca_rmsds, field_rs, field_rmses = [], [], []
        for other_id in seed_ids[1:]:
            other_paths = ProteinPaths(other_id, data_root)
            other_ca = _read_ca_coords(other_paths.pqr_path)
            if len(other_ca) != len(ref_ca):
                print(f"  [{other_id}] CA count mismatch ({len(other_ca)} vs {len(ref_ca)}) -- skipping pair")
                continue

            R, t, ca_rmsd = _kabsch(other_ca, ref_ca)
            other_esp_npz = np.load(other_paths.esp_path)
            other_query_idx = other_esp_npz["query_idx"]
            other_query_pos = other_esp_npz["verts"][other_query_idx].astype(np.float64)
            other_query_esp = other_esp_npz["esp_verts"][other_query_idx].astype(np.float64)

            r, rmse = _esp_field_agreement(ref_verts, ref_esp, other_query_pos, other_query_esp, R, t)
            ca_rmsds.append(ca_rmsd)
            field_rs.append(r)
            field_rmses.append(rmse)
            print(f"  [{ref_id} vs {other_id}] CA-RMSD={ca_rmsd:.3f} A  ESP field r={r:.4f}  RMSE={rmse:.3f}")

        model_err = eval_summary[eval_summary["protein_id"].isin(seed_ids)]
        row = {
            "parent_protein_id": parent_id, "pilot_group": pilot_group,
            "mean_ca_rmsd": float(np.mean(ca_rmsds)), "max_ca_rmsd": float(np.max(ca_rmsds)),
            "mean_esp_field_r": float(np.mean(field_rs)), "min_esp_field_r": float(np.min(field_rs)),
            "mean_esp_field_rmse": float(np.mean(field_rmses)),
        }
        for model in ("attention", "distance"):
            m = model_err[model_err["model"] == model]
            row[f"{model}_rmse_mean"] = float(m["rmse"].mean())
            row[f"{model}_rmse_std"] = float(m["rmse"].std())
            row[f"{model}_pearson_r_mean"] = float(m["pearson_r"].mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
