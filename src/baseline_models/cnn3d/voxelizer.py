"""Convert a PQR file into a 5-channel element-type 3D voxel grid.

Channels (element occupancy COUNT only — no charge information of any kind):
  0: H  (hydrogen)
  1: C  (carbon)
  2: N  (nitrogen)
  3: O  (oxygen)
  4: S  (sulfur)

Matches graph_builder.py's ELEMENT_VOCAB exactly ({"H":0,"C":1,"N":2,"O":3,"S":4})
so the CNN sees the same atom-identity coverage as the GNN's atom nodes, which
do include hydrogen (PQR files from PDB2PQR carry explicit H). Only rare
elements (P, Fe, etc.) are still dropped — negligible occurrence, absent from
the GNN's ELEMENT_VOCAB too (falls into its "unknown" bucket there).

Each channel is a plain per-voxel atom COUNT (`+= 1.0`), never weighted or
scaled by the PQR partial charge column — charge is parsed nowhere in this
module. This keeps the CNN on the same epistemological footing as the GNN:
structure and atom identity, but no partial charges (which are the direct
physical cause of the ESP signal being predicted).

Resolution: 1 Å/voxel (configurable).
Bounding box: heavy-atom extent + padding on each side.

Returns:
  grid:       (5, D, H, W) float32 tensor
  origin:     (3,) float32 — xyz of voxel (0,0,0) corner in Å
  voxel_size: float — Å per voxel
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Channel index per element; elements not in this map are skipped.
# Order matches src/data/graph_builder.py's ELEMENT_VOCAB.
_ELEMENT_CHANNEL = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
N_CHANNELS = 5


def pqr_to_voxel(
    pqr_path: Path,
    voxel_size: float = 1.0,
    padding: float = 5.0,
) -> tuple[Tensor, np.ndarray, float]:
    """Voxelize a PQR file into a 4-channel element-occupancy grid.

    Args:
        pqr_path:   path to PQR file from PDB2PQR
        voxel_size: Å per voxel (default 1.0)
        padding:    Å of empty space around the heavy-atom bounding box

    Returns:
        grid:       (4, D, H, W) float32 torch Tensor
        origin:     (3,) float32 numpy array — xyz of voxel grid corner in Å
        voxel_size: the voxel_size used (for later query-position mapping)
    """
    positions, channels = _parse_pqr_elements(pqr_path)

    lo = positions.min(axis=0) - padding
    hi = positions.max(axis=0) + padding
    dims = np.ceil((hi - lo) / voxel_size).astype(np.int32) + 1  # (3,) — D, H, W

    grid = np.zeros((N_CHANNELS, *dims), dtype=np.float32)

    idx = np.floor((positions - lo) / voxel_size).astype(np.int32)  # (N, 3)
    idx = np.clip(idx, 0, dims - 1)

    for pos_idx, ch in zip(idx, channels):
        d, h, w = pos_idx
        grid[ch, d, h, w] += 1.0

    return torch.from_numpy(grid), lo.astype(np.float32), float(voxel_size)


def query_xyz_to_grid_coords(
    query_pos: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    grid_shape: tuple[int, int, int],
) -> Tensor:
    """Convert query positions (Å) to normalised grid sample coordinates.

    Returns:
        coords: (1, Q, 1, 1, 3) float32 in [-1, 1]^3 — suitable for
                F.grid_sample with align_corners=True on a (1, C, D, H, W) grid.

    Note: grid_sample uses (W, H, D) ordering (x, y, z), which matches
    our (D, H, W) = (z, y, x) axis ordering in pqr_to_voxel.
    """
    D, H, W = grid_shape

    frac = (query_pos - origin) / voxel_size  # (Q, 3) — fractional voxel coords

    # Normalise to [-1, 1] with align_corners=True: -1 = voxel edge 0, +1 = edge dim-1
    norm_d = 2.0 * frac[:, 0] / (D - 1) - 1.0  # (Q,)
    norm_h = 2.0 * frac[:, 1] / (H - 1) - 1.0
    norm_w = 2.0 * frac[:, 2] / (W - 1) - 1.0

    # grid_sample coord order: (x=W, y=H, z=D)
    coords = np.stack([norm_w, norm_h, norm_d], axis=-1).astype(np.float32)  # (Q, 3)
    coords = torch.from_numpy(coords[None, :, None, None, :])  # (1, Q, 1, 1, 3)
    return coords


def _parse_pqr_elements(pqr_path: Path) -> tuple[np.ndarray, list[int]]:
    """Read PQR → atom positions and element channel indices.

    Skips only rare elements not in _ELEMENT_CHANNEL (P, metals, etc.); H is
    included. Element inferred from first character of atom name (PDB2PQR
    convention).
    """
    positions, channels = [], []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            atom_name = parts[2]
            element = atom_name[0].upper()
            ch = _ELEMENT_CHANNEL.get(element)
            if ch is None:
                continue  # skip H and rare elements
            x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
            positions.append([x, y, z])
            channels.append(ch)
    return np.array(positions, dtype=np.float32), channels
