"""
scripts/compute_pdb_af_agreement.py

PDB-vs-AlphaFold comparison notebook, phase 2: for each of the 41 AF<->PDB
pairs in ~/thesis/pdb_comparison_data/, measures structural agreement
(Kabsch-aligned CA-RMSD, independently recomputed as a cross-check against
the pre-existing `ca_rmsd_vs_af` in each pdb/*/metadata.json) and
ESP-field agreement (cross-structure RBF-interpolated Pearson r/RMSE,
identical technique to scripts/compute_seed_agreement.py) between the two
structures of the same protein.

Methodology (extends compute_seed_agreement.py's Kabsch+RBF technique with
one necessary addition -- see below):
  - **Residues must be sequence-aligned before Kabsch, not paired
    positionally.** Unlike the seed-conformation pilot (5 ColabFold
    predictions of the *exact same* sequence, so positional CA order was a
    safe pairing key), real PDB crystal structures commonly have
    missing/unresolved N-/C-terminal residues or disordered internal loops,
    **and their own residue numbering does not follow the AF/UniProt
    numbering** -- confirmed empirically (e.g. AF-P09373-F1 starts
    "1 MET, 2 SER, 3 GLU..." while its paired PDB-P09373-1H16-A starts
    "1 SER, 2 GLU, 3 LEU..." -- the crystal structure's unresolved
    N-terminal Met means everything is off by one, and this offset differs
    per structure). Naive index- or resid-based pairing silently misaligns
    residues (verified: produced CA-RMSDs of 3.7-3.8 A against a
    pre-validated reference of ~0.2-0.3 A for the same pairs). The fix:
    global pairwise sequence alignment (Bio.Align.PairwiseAligner) on the
    one-letter CA sequence from each structure, keeping only positions
    where the alignment reports an exact residue-identity match, then using
    *those* matched indices to pull the corresponding CA coordinates for
    Kabsch. This is standard practice for AF-vs-crystal-structure RMSD (and
    almost certainly what the original, now-missing
    `scripts/fetch_af_pdb_pairs.py` did to produce the `ca_rmsd_vs_af`
    metadata field this script cross-checks against).
  - AF is the reference frame (the structure the models were trained on).
    The PDB structure's *aligned* CA atoms are Kabsch-fit onto the AF
    structure's matching CA atoms, giving both a CA-RMSD and the (R, t)
    transform needed to bring the PDB structure's whole surface into the
    AF frame.
  - ESP fields can't be compared vertex-to-vertex (independently-generated
    SES meshes, different topology) -- the PDB structure's ESP is refit as
    an RBF interpolator (scipy RBFInterpolator, multiquadric kernel, same
    epsilon convention as src/surface/esp_mapping.py::rbf_reconstruct) on
    its own curvature-sampled query-node subset in the aligned frame (fitting
    on the full dense vertex set produces near-coplanar local neighborhoods
    and a singular RBF system -- discovered and fixed in
    compute_seed_agreement.py), then evaluated at the AF mesh's own vertex
    positions for a direct, same-length comparison against the AF
    structure's actual ESP.

Run in the `pyg_env` conda environment (scipy, biopython; no torch needed).

Usage:
    conda activate pyg_env
    python -u scripts/compute_pdb_af_agreement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.interpolate import RBFInterpolator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.structure.seed_conformations import THREE_TO_ONE
from src.surface.esp_mapping import _rbf_epsilon
from src.utils.paths import ProteinPaths

PDB_COMPARISON_ROOT = Path("/home/student/thesis/pdb_comparison_data")
AF_DATA_ROOT = PDB_COMPARISON_ROOT / "af"
PDB_DATA_ROOT = PDB_COMPARISON_ROOT / "pdb"

OUT_CSV = Path("/home/student/thesis/outputs/pdb_af_comparison.csv")


def _read_ca_seq_coords(pqr_path: Path) -> tuple[str, np.ndarray]:
    """One-letter CA sequence + (N_res, 3) CA coordinates, both in file
    (positional) order -- positional order is reliable within one
    structure; only cross-structure resid numbering is unreliable (see
    module docstring)."""
    letters, coords = [], []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line.split()[2] == "CA":
                fields = line.split()
                resname = fields[3]
                letters.append(THREE_TO_ONE.get(resname, "X"))
                coords.append([float(fields[5]), float(fields[6]), float(fields[7])])
    return "".join(letters), np.array(coords, dtype=np.float64)


_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.match_score = 2
_ALIGNER.mismatch_score = -1
_ALIGNER.open_gap_score = -10
_ALIGNER.extend_gap_score = -0.5


def _align_ca_coords(
    af_seq: str, af_coords: np.ndarray, pdb_seq: str, pdb_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Global sequence alignment (Bio.Align.PairwiseAligner) between the two
    structures' CA sequences, keeping only exact-identity-matched
    positions. Returns (af_matched_coords, pdb_matched_coords, n_matched)
    -- same-length, index-paired coordinate arrays ready for Kabsch.
    """
    alignment = _ALIGNER.align(af_seq, pdb_seq)[0]
    af_idx, pdb_idx = alignment.indices  # -1 where the other sequence has a gap
    keep = (af_idx >= 0) & (pdb_idx >= 0) & (np.array(list(af_seq))[np.clip(af_idx, 0, None)]
                                              == np.array(list(pdb_seq))[np.clip(pdb_idx, 0, None)])
    return af_coords[af_idx[keep]], pdb_coords[pdb_idx[keep]], int(keep.sum())


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
    """Bring `other`'s ESP field into the reference frame via (R, t), fit an
    RBF interpolator on its sparse query-node subset, evaluate at the
    reference's own vertex positions, and return (pearson_r, rmse) against
    the reference's own ESP."""
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
    pdb_ids = sorted(p.name for p in PDB_DATA_ROOT.iterdir() if p.is_dir())
    print(f"{len(pdb_ids)} PDB entries (each paired to one AF structure via metadata)")

    rows = []
    for pdb_id in pdb_ids:
        pdb_paths = ProteinPaths(pdb_id, PDB_DATA_ROOT)
        pdb_meta = json.loads(pdb_paths.metadata_path.read_text())
        af_id = pdb_meta["af_protein_id"]
        af_paths = ProteinPaths(af_id, AF_DATA_ROOT)
        af_meta = json.loads(af_paths.metadata_path.read_text())

        print(f"\n=== {af_id} vs {pdb_id} (uniprot={pdb_meta['accession']}) ===")

        af_seq, af_ca = _read_ca_seq_coords(af_paths.pqr_path)
        pdb_seq, pdb_ca = _read_ca_seq_coords(pdb_paths.pqr_path)

        if not af_paths.esp_path.exists() or not pdb_paths.esp_path.exists():
            print(f"  ESP file missing (af={af_paths.esp_path.exists()}, pdb={pdb_paths.esp_path.exists()}) -- skipping pair")
            continue

        af_ca_aligned, pdb_ca_aligned, n_matched = _align_ca_coords(af_seq, af_ca, pdb_seq, pdb_ca)
        coverage = n_matched / len(af_seq)
        if n_matched < 20 or coverage < 0.5:
            print(f"  Alignment too poor ({n_matched}/{len(af_seq)} residues matched, "
                  f"{coverage:.1%} coverage) -- skipping pair")
            continue

        R, t, ca_rmsd = _kabsch(pdb_ca_aligned, af_ca_aligned)

        af_esp_npz = np.load(af_paths.esp_path)
        af_verts, af_esp = af_esp_npz["verts"].astype(np.float64), af_esp_npz["esp_verts"].astype(np.float64)

        pdb_esp_npz = np.load(pdb_paths.esp_path)
        pdb_query_idx = pdb_esp_npz["query_idx"]
        pdb_query_pos = pdb_esp_npz["verts"][pdb_query_idx].astype(np.float64)
        pdb_query_esp = pdb_esp_npz["esp_verts"][pdb_query_idx].astype(np.float64)

        esp_field_r, esp_field_rmse = _esp_field_agreement(af_verts, af_esp, pdb_query_pos, pdb_query_esp, R, t)

        print(f"  aligned {n_matched}/{len(af_seq)} residues ({coverage:.1%})  "
              f"CA-RMSD recomputed={ca_rmsd:.3f} A  (metadata: {pdb_meta['ca_rmsd_vs_af']:.3f} A)  "
              f"ESP field r={esp_field_r:.4f}  RMSE={esp_field_rmse:.3f}")

        rows.append({
            "uniprot_id": pdb_meta["accession"],
            "af_protein_id": af_id,
            "pdb_protein_id": pdb_id,
            "pdb_id": pdb_meta["pdb_id"],
            "chain": pdb_meta["chain"],
            "sequence_length": pdb_meta["sequence_length"],
            "resolution_A": pdb_meta["resolution_A"],
            "n_aligned_residues": n_matched,
            "alignment_coverage": coverage,
            "ca_rmsd_recomputed": ca_rmsd,
            "ca_rmsd_metadata": pdb_meta["ca_rmsd_vs_af"],
            "esp_field_r": esp_field_r,
            "esp_field_rmse": esp_field_rmse,
            "ses_area_af": af_meta["ses_area"],
            "ses_area_pdb": pdb_meta["ses_area"],
            "n_vertices_af": af_meta["n_vertices"],
            "n_vertices_pdb": pdb_meta["n_vertices"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")

    delta = (df["ca_rmsd_recomputed"] - df["ca_rmsd_metadata"]).abs()
    print(f"\nCross-check: |recomputed - metadata| CA-RMSD delta: mean={delta.mean():.4f}  max={delta.max():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
