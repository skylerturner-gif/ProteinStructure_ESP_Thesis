"""Vacuum Coulomb ESP predictor — physics lower bound for APBS ESP prediction.

Computes V(q) = Σ_i q_i · BJERRUM_VACUUM / r_iq  [kT/e]
using PARSE partial charges from PQR files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

# e² / (4πε₀ · kBT) at 298 K — converts charge/Å to kT/e
BJERRUM_VACUUM = 560.0

# Clamp distance floor to avoid singularities at atom centers
R_MIN = 0.5

# Query points evaluated per chunk to bound peak RAM
CHUNK_SIZE = 512


def parse_pqr(pqr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a PQR file and return atom positions and PARSE partial charges.

    Returns:
        atom_pos:     (N, 3) float32 — xyz in Å
        atom_charges: (N,)   float32 — elementary charges
    """
    positions, charges = [], []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
            charge = float(parts[-2])
            positions.append([x, y, z])
            charges.append(charge)
    return np.array(positions, dtype=np.float32), np.array(charges, dtype=np.float32)


def coulomb_potential(
    query_pos: np.ndarray,
    atom_pos: np.ndarray,
    atom_charges: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Evaluate vacuum Coulomb potential at query positions.

    Args:
        query_pos:    (Q, 3) float32 — query xyz in Å
        atom_pos:     (N, 3) float32 — atom xyz in Å
        atom_charges: (N,)   float32 — PARSE partial charges
        chunk_size:   number of query points per chunk

    Returns:
        V: (Q,) float32 — predicted ESP in kT/e
    """
    n_q = len(query_pos)
    v = np.zeros(n_q, dtype=np.float64)
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        diff = query_pos[start:end, np.newaxis, :] - atom_pos[np.newaxis, :, :]  # (chunk, N, 3)
        r = np.linalg.norm(diff, axis=-1)  # (chunk, N)
        np.clip(r, R_MIN, None, out=r)
        v[start:end] = (atom_charges[np.newaxis, :] / r).sum(axis=-1) * BJERRUM_VACUUM
    return v.astype(np.float32)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    """Return (pearson_r, rmse, mae) for paired 1-D arrays."""
    r, _ = pearsonr(pred.astype(float), true.astype(float))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    return float(r), rmse, mae


class CoulombESP:
    """Stateless predictor: load PQR once, call predict() for any query positions."""

    def __init__(self, pqr_path: Path) -> None:
        self.atom_pos, self.atom_charges = parse_pqr(pqr_path)

    def predict(self, query_pos: np.ndarray) -> np.ndarray:
        """Return Coulomb ESP (kT/e) at query_pos (Q, 3)."""
        return coulomb_potential(query_pos, self.atom_pos, self.atom_charges)

    def evaluate(self, query_pos: np.ndarray, true_esp: np.ndarray) -> tuple[float, float, float]:
        """Return (pearson_r, rmse, mae) vs ground-truth ESP."""
        pred = self.predict(query_pos)
        return compute_metrics(pred, true_esp)
