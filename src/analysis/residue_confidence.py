"""
src/analysis/residue_confidence.py

Joins AlphaFold's own per-residue confidence signals (pLDDT, PAE) onto
query-node mesh vertices, via nearest-atom-then-resid lookup. No model
inference involved — pure geometry + metadata join, used to test whether
model prediction error correlates with AlphaFold's structural confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils.io import load_metadata
from src.utils.paths import ProteinPaths

__all__ = [
    "read_pqr_atom_coords_resid",
    "load_pae_matrix",
    "nearest_atom_resid",
    "query_confidence_table",
]


def read_pqr_atom_coords_resid(pqr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse per-atom coordinates and residue sequence numbers from a PQR file.

    PQR ATOM line format (space-delimited):
        ATOM serial name resname chain resseq x y z charge radius

    Returns:
        coords: (n_atoms, 3) float32 array
        resid:  (n_atoms,) int array, AlphaFold's 1-indexed residue number
    """
    coords: list[list[float]] = []
    resid:  list[int] = []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                resid.append(int(fields[4]))
                coords.append([float(fields[5]), float(fields[6]), float(fields[7])])
    return np.array(coords, dtype=np.float32), np.array(resid, dtype=np.int64)


def load_pae_matrix(pae_path: Path) -> np.ndarray | None:
    """
    Load an AlphaFold PAE JSON's predicted_aligned_error matrix.

    Returns (n_res, n_res) float32 array, or None if the file is missing
    or malformed (PAE download can fail per-protein — callers must treat
    this as optional, not assumed present).
    """
    if not pae_path.exists():
        return None
    try:
        with open(pae_path) as f:
            data = json.load(f)
        entry = data[0] if isinstance(data, list) else data
        pae = entry.get("predicted_aligned_error")
        if pae is None:
            return None
        return np.array(pae, dtype=np.float32)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def nearest_atom_resid(
    query_pos: np.ndarray, atom_coords: np.ndarray, atom_resid: np.ndarray
) -> np.ndarray:
    """For each query position, the resid of its nearest atom (cKDTree)."""
    tree = cKDTree(atom_coords)
    _, idx = tree.query(query_pos, k=1)
    return atom_resid[idx]


def query_confidence_table(protein_id: str, data_root: Path) -> pd.DataFrame | None:
    """
    Build a per-query-vertex confidence table for one protein: resid,
    pLDDT, and mean PAE (row-mean of that residue's PAE to all other
    residues — the natural single-residue summary of a pairwise matrix).

    Returns None if the PQR or esp.npz is missing. Query vertices whose
    nearest-atom resid falls outside the pLDDT array's range (would only
    happen for multi-fragment proteins with non-contiguous numbering) are
    dropped and counted in the returned `n_out_of_range` attrs-style note
    via the DataFrame's `.attrs` dict, not silently misaligned.
    """
    paths = ProteinPaths(protein_id, data_root)
    if not paths.pqr_path.exists() or not paths.esp_path.exists():
        return None

    esp_npz = np.load(paths.esp_path)
    verts, query_idx = esp_npz["verts"], esp_npz["query_idx"]
    query_pos = verts[query_idx]

    atom_coords, atom_resid = read_pqr_atom_coords_resid(paths.pqr_path)
    if len(atom_coords) == 0:
        return None

    metadata = load_metadata(protein_id, data_root)
    plddt = np.array(metadata.get("plddt_per_residue") or [], dtype=np.float32)
    if plddt.size == 0:
        return None

    pae_matrix = load_pae_matrix(paths.pae_path)
    pae_row_mean = pae_matrix.mean(axis=1) if pae_matrix is not None else None

    resid = nearest_atom_resid(query_pos, atom_coords, atom_resid)
    res_index = resid - 1  # AlphaFold resid is 1-indexed

    in_range = (res_index >= 0) & (res_index < len(plddt))
    n_out_of_range = int((~in_range).sum())

    rows = {
        "query_idx":  query_idx[in_range],
        "resid":      resid[in_range],
        "plddt":      plddt[res_index[in_range]],
    }
    if pae_row_mean is not None:
        pae_in_range = in_range & (res_index < len(pae_row_mean))
        pae_col = np.full(in_range.sum(), np.nan, dtype=np.float32)
        # align pae_in_range (over full query set) to the already-filtered in_range rows
        sub_mask = pae_in_range[in_range]
        pae_col[sub_mask] = pae_row_mean[res_index[in_range][sub_mask]]
        rows["pae_mean"] = pae_col
    else:
        rows["pae_mean"] = np.full(in_range.sum(), np.nan, dtype=np.float32)

    df = pd.DataFrame(rows)
    df.attrs["n_out_of_range"] = n_out_of_range
    df.attrs["n_query"] = len(query_idx)
    df.attrs["has_pae"] = pae_row_mean is not None
    return df
