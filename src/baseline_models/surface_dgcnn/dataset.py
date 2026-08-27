"""Dataset for Surface DGCNN baseline.

Each sample is a protein surface point cloud with pre-computed static kNN graph:
  - point_cloud:  (S, 33)     — xyz + normals + nearest-atom element/residue
                                one-hot, randomly downsampled to max_points
  - edge_index:   (2, S*k)    — kNN graph on xyz coordinates, cached to disk
  - query_pos:    (Q, 3)      — query node positions
  - query_esp:    (Q,)        — APBS ground-truth ESP at query nodes

The geometry-only variant (xyz+normals, no chemistry) was dropped — it scored
Pearson r=0.085 on the test set (surface shape alone carries almost no ESP
signal), see notebooks/baseline_comparison.ipynb. Chemistry features
(element+residue identity, no partial charges) are always on now.

kNN graph is built on 3D coordinates (not feature embeddings), which makes it
fast (scipy cKDTree, ~0.1s for 8192 points) and cacheable. Building it once per
protein eliminates the dynamic kNN overhead from every forward pass.

Cache files:
  <mesh_dir>/<pid>_surf{max_points}_k{k}.npy  — (S*k, 2) int32 edge pairs
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor
from torch.utils.data import Dataset

from MDAnalysis.topology.guessers import guess_types
import MDAnalysis as mda

from src.data.graph_builder import (
    ELEMENT_VOCAB, RESIDUE_VOCAB, N_ELEMENT_TYPES, N_RESIDUE_TYPES,
)
from src.utils.paths import ProteinPaths


class SurfaceDataset(Dataset):
    """Point-cloud surface dataset for DGCNN ESP prediction."""

    def __init__(
        self,
        protein_ids: list[str],
        data_root: Path,
        max_points: Optional[int] = 8192,
        k: int = 20,
    ) -> None:
        """
        Args:
            protein_ids:       list of protein IDs in this split
            data_root:         path to external_protein_data directory
            max_points:        random subsample target; None keeps all vertices
            k:                 neighbours per node in the static kNN graph
        """
        self.protein_ids = protein_ids
        self.data_root = Path(data_root)
        self.max_points = max_points
        self.k = k
        self._valid: list[str] = self._filter_available()

    @property
    def in_features(self) -> int:
        """Feature dimension of each surface point (for model construction)."""
        return 6 + N_ELEMENT_TYPES + N_RESIDUE_TYPES  # xyz+normals + element + residue = 33

    def _filter_available(self) -> list[str]:
        valid = []
        for pid in self.protein_ids:
            p = ProteinPaths(pid, self.data_root)
            mesh_path = self.data_root / pid / "mesh" / f"{pid}_mesh.npz"
            if p.esp_path.exists() and mesh_path.exists():
                valid.append(pid)
        return valid

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        pid = self._valid[idx]
        p = ProteinPaths(pid, self.data_root)
        mesh_dir = self.data_root / pid / "mesh"
        mesh_path = mesh_dir / f"{pid}_mesh.npz"

        mesh = np.load(mesh_path)
        verts: np.ndarray = mesh["verts"]      # (N_v, 3)
        normals: np.ndarray = mesh["normals"]  # (N_v, 3)

        if self.max_points is not None and len(verts) > self.max_points:
            keep = _random_subsample(verts, self.max_points, seed=_pid_seed(pid))
            verts = verts[keep]
            normals = normals[keep]

        edge_index = self._load_or_compute_knn(pid, verts, mesh_dir)

        esp_data = np.load(p.esp_path)
        query_idx = esp_data["query_idx"]
        query_pos = esp_data["verts"][query_idx].astype(np.float32)  # (Q, 3)
        query_esp = esp_data["esp_verts"][query_idx].astype(np.float32)  # (Q,)

        # (Q, n_query_k) — index of k nearest surface points per query node
        query_surf = self._load_or_compute_query_surf(
            pid, verts, query_pos, mesh_dir
        )

        elem_idx, res_idx = self._load_or_compute_chem(pid, verts, mesh_dir)
        elem_oh = np.eye(N_ELEMENT_TYPES, dtype=np.float32)[elem_idx]  # (S, 6)
        res_oh  = np.eye(N_RESIDUE_TYPES, dtype=np.float32)[res_idx]   # (S, 21)
        point_cloud = np.concatenate(
            [verts, normals, elem_oh, res_oh], axis=1
        ).astype(np.float32)  # (S, 33)

        return {
            "protein_id": pid,
            "point_cloud": torch.from_numpy(point_cloud),              # (S, 33)
            "edge_index": torch.from_numpy(edge_index).long(),         # (2, S*k)
            "query_surf": torch.from_numpy(query_surf).long(),         # (Q, n_query_k)
            "query_esp": torch.from_numpy(query_esp),                   # (Q,)
        }

    def _load_or_compute_knn(
        self, pid: str, verts: np.ndarray, mesh_dir: Path
    ) -> np.ndarray:
        """Return (2, S*k) edge array, caching to disk on first call."""
        S = len(verts)
        cache = mesh_dir / f"{pid}_surf{S}_k{self.k}.npy"
        if cache.exists():
            return np.load(cache)
        edge_index = _build_knn_graph(verts, self.k)
        np.save(cache, edge_index)
        return edge_index

    def _load_or_compute_query_surf(
        self, pid: str, verts: np.ndarray, query_pos: np.ndarray, mesh_dir: Path
    ) -> np.ndarray:
        """Return (Q, n_query_k) int32 index of nearest surface points per query.

        Cached per protein+surface-size combination.
        """
        n_qk = 8  # fixed; model uses min(n_query_k, S) anyway
        S = len(verts)
        cache = mesh_dir / f"{pid}_surf{S}_qsurf{n_qk}.npy"
        if cache.exists():
            return np.load(cache)
        tree = cKDTree(verts)
        _, nbr = tree.query(query_pos, k=n_qk, workers=-1)  # (Q, n_qk)
        nbr = nbr.astype(np.int32)
        np.save(cache, nbr)
        return nbr


    def _load_or_compute_chem(
        self, pid: str, verts: np.ndarray, mesh_dir: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (elem_idx, res_idx) arrays of shape (S,) for each surface vertex.

        Each vertex is assigned the element and residue of its nearest PQR atom.
        Uses the same ELEMENT_VOCAB / RESIDUE_VOCAB as the GNN — no partial charges.
        Cached to disk so PQR parsing only runs once per protein.
        """
        S = len(verts)
        cache = mesh_dir / f"{pid}_surf{S}_chem.npy"
        if cache.exists():
            data = np.load(cache)
            return data[:, 0], data[:, 1]

        p = ProteinPaths(pid, self.data_root)
        u = mda.Universe(str(p.pqr_path))
        elements = list(guess_types(u.atoms.names))
        atom_xyz = u.atoms.positions.astype(np.float32)
        atom_resnames = list(u.atoms.resnames)

        elem_idx = np.array(
            [ELEMENT_VOCAB.get(e, N_ELEMENT_TYPES - 1) for e in elements], dtype=np.int16
        )
        res_idx = np.array(
            [RESIDUE_VOCAB.get(r, N_RESIDUE_TYPES - 1) for r in atom_resnames], dtype=np.int16
        )

        tree = cKDTree(atom_xyz)
        _, nearest = tree.query(verts, k=1, workers=-1)
        nearest = np.asarray(nearest).ravel()

        chem = np.stack([elem_idx[nearest], res_idx[nearest]], axis=1)
        np.save(cache, chem)
        return chem[:, 0], chem[:, 1]


def _build_knn_graph(points: np.ndarray, k: int) -> np.ndarray:
    """Build a kNN graph on 3D coordinates via scipy cKDTree.

    Returns edge_index (2, N*k) in COO format. Self-loops excluded.
    Fast: O(N log N) tree build + O(N k log N) queries. ~0.1s for N=8192.
    """
    tree = cKDTree(points)
    # Query k+1 because the nearest neighbour of each point is itself
    _, nbr_indices = tree.query(points, k=k + 1, workers=-1)  # (N, k+1)
    nbr_indices = nbr_indices[:, 1:]  # (N, k) — drop self-loop

    N = len(points)
    src = np.repeat(np.arange(N, dtype=np.int32), k)  # (N*k,)
    dst = nbr_indices.flatten().astype(np.int32)       # (N*k,)
    return np.stack([src, dst], axis=0)                # (2, N*k)


def _random_subsample(points: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Random subset of n_samples indices. O(N), deterministic per protein."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(points), size=n_samples, replace=False)


def _pid_seed(pid: str) -> int:
    """Stable integer seed from protein ID (invariant across Python versions)."""
    return int.from_bytes(pid.encode()[:8], byteorder="little") & 0x7FFFFFFF


def collate_surface(batch: list[dict]) -> dict[str, list | Tensor]:
    """Custom collate: keeps variable-length tensors as lists."""
    out: dict = {k: [] for k in batch[0].keys()}
    for item in batch:
        for k, v in item.items():
            out[k].append(v)
    return out
