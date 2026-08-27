"""Dataset for 3D CNN baseline.

Each sample is a full-protein voxel grid with query positions for interpolation.
Voxelization is done on-the-fly in __getitem__ to avoid storing large grids on disk.

Memory profile per protein (1Å/voxel, 5Å padding):
  - Median protein (~340 residues): ~5 MB grid
  - P90 protein (~780 residues):   ~15 MB grid
  - Max protein (1000 residues):   ~30 MB grid
Batch size 1 is strongly recommended.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.baseline_models.cnn3d.voxelizer import pqr_to_voxel, query_xyz_to_grid_coords
from src.utils.paths import ProteinPaths


class Vox3DDataset(Dataset):
    """Returns voxel grids + query coords + ESP targets for 3D CNN training."""

    def __init__(
        self,
        protein_ids: list[str],
        data_root: Path,
        voxel_size: float = 1.0,
        padding: float = 5.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.voxel_size = voxel_size
        self.padding = padding
        self._valid = self._filter_available(protein_ids)

    def _filter_available(self, protein_ids: list[str]) -> list[str]:
        valid = []
        for pid in protein_ids:
            p = ProteinPaths(pid, self.data_root)
            if p.pqr_path.exists() and p.esp_path.exists():
                valid.append(pid)
        return valid

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        pid = self._valid[idx]
        p = ProteinPaths(pid, self.data_root)

        grid, origin, voxel_size = pqr_to_voxel(
            p.pqr_path, voxel_size=self.voxel_size, padding=self.padding
        )  # (5, D, H, W)

        esp_data = np.load(p.esp_path)
        query_idx = esp_data["query_idx"]
        query_pos = esp_data["verts"][query_idx].astype(np.float32)  # (Q, 3)
        query_esp = esp_data["esp_verts"][query_idx].astype(np.float32)  # (Q,)

        grid_shape = (grid.shape[1], grid.shape[2], grid.shape[3])
        query_coords = query_xyz_to_grid_coords(query_pos, origin, voxel_size, grid_shape)

        return {
            "protein_id": pid,
            "grid": grid.unsqueeze(0),     # (1, 5, D, H, W)
            "query_coords": query_coords,  # (1, Q, 1, 1, 3)
            "query_esp": torch.from_numpy(query_esp),   # (Q,)
        }
