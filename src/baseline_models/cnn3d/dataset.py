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

from src.baseline_models.cnn3d.voxelizer import (
    estimate_grid_voxels,
    pqr_to_voxel,
    query_xyz_to_grid_coords,
)
from src.utils.paths import ProteinPaths

# Default cap on D*H*W for a single grid. The encoder/decoder run full-res
# 32-96-channel Conv3d blocks (see model.py) whose activation memory scales
# ~linearly with voxel count and batch size is fixed at 1 -- there is no
# batching lever to fall back on for an oversized protein the way the main
# GNN pipeline's DynamicBatchSampler has. A full-dataset scan (2026-09-22,
# 7613 train+val proteins) found p99=5.82M voxels but a long tail up to
# 17.75M (elongated/multi-domain proteins) -- one of those OOM'd a 40GB A100
# on a single forward+backward at batch=1. 6,000,000 sits just above that
# p99 and excludes roughly the top ~1% of proteins by bounding-box volume.
DEFAULT_MAX_VOXELS = 6_000_000


class Vox3DDataset(Dataset):
    """Returns voxel grids + query coords + ESP targets for 3D CNN training."""

    def __init__(
        self,
        protein_ids: list[str],
        data_root: Path,
        voxel_size: float = 1.0,
        padding: float = 5.0,
        max_voxels: int | None = DEFAULT_MAX_VOXELS,
    ) -> None:
        self.data_root = Path(data_root)
        self.voxel_size = voxel_size
        self.padding = padding
        self.max_voxels = max_voxels
        self._valid = self._filter_available(protein_ids)

    def _filter_available(self, protein_ids: list[str]) -> list[str]:
        valid = []
        n_missing = n_oversized = 0
        for pid in protein_ids:
            p = ProteinPaths(pid, self.data_root)
            if not (p.pqr_path.exists() and p.esp_path.exists()):
                n_missing += 1
                continue
            if self.max_voxels is not None:
                n_vox = estimate_grid_voxels(p.pqr_path, self.voxel_size, self.padding)
                if n_vox > self.max_voxels:
                    n_oversized += 1
                    continue
            valid.append(pid)
        if n_oversized:
            print(f"Vox3DDataset: skipped {n_oversized} protein(s) over "
                  f"max_voxels={self.max_voxels:,} (grid too large for a "
                  f"batch-size-1 dense forward/backward), {n_missing} missing "
                  f"files, {len(valid)} usable of {len(protein_ids)} requested.")
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
