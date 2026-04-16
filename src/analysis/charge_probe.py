"""
src/analysis/charge_probe.py

Partial-charge probe: train a small frozen-backbone MLP to predict per-atom
partial charges (from PQR files) using atom embeddings from a trained model.
Tests whether the model's internal atom representations encode chemistry.

The backbone is always frozen — only ChargeProbe.mlp parameters are trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.egnn import _mlp
from src.utils.paths import ProteinPaths

__all__ = [
    "ChargeProbe",
    "read_pqr_charges",
    "read_pqr_atoms",
    "extract_atom_embeddings",
    "train_probe",
    "evaluate_probe",
]


class ChargeProbe(nn.Module):
    """
    Three-layer MLP predicting per-atom partial charges from atom embeddings.
    Only this module's parameters are trained; backbone weights stay frozen.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = _mlp([hidden_dim, hidden_dim // 2, 1])

    def forward(self, h_atom: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_atom).squeeze(-1)


def read_pqr_charges(pqr_path: Path) -> np.ndarray:
    """
    Parse per-atom partial charges from a PQR file.

    PQR ATOM line format (space-delimited):
        ATOM serial name resname chain resseq x y z charge radius

    Returns:
        (n_atoms,) float32 array of partial charges in units of e.
    """
    charges: list[float] = []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                charges.append(float(fields[8]))
    return np.array(charges, dtype=np.float32)


def read_pqr_atoms(pqr_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Parse per-atom charges, atom names, and residue names from a PQR file.

    PQR ATOM line format (space-delimited):
        ATOM serial name resname chain resseq x y z charge radius

    Returns:
        charges:   (n_atoms,) float32 array of partial charges in units of e
        atom_names: list of atom name strings (e.g. "CA", "OG", "NZ")
        res_names:  list of residue name strings (e.g. "ALA", "SER", "PHE")
    """
    charges:    list[float] = []
    atom_names: list[str]   = []
    res_names:  list[str]   = []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                atom_names.append(fields[2])
                res_names.append(fields[3])
                charges.append(float(fields[8]))
    return np.array(charges, dtype=np.float32), atom_names, res_names


def extract_atom_embeddings(
    model: nn.Module,
    data,
    layer: str = "after_mp",
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extract atom-level embeddings from a frozen model.

    Args:
        model:  trained model (AttentionESPN or DistanceESPN), frozen
        data:   HeteroData for one protein (loaded via _load_graph)
        layer:  "after_encoder" — raw element/residue/bond embeddings
                "after_mp"      — after Stage 1 bond+radial message passing
        device: run inference on this device

    Returns:
        (n_atoms, hidden_dim) float tensor on CPU
    """
    if device is not None:
        data = data.to(device)

    with torch.no_grad():
        h = model.atom_encoder(data)
        if layer == "after_mp":
            h = model.atom_mp(h, data)
    return h.cpu()


def train_probe(
    probe: ChargeProbe,
    model: nn.Module,
    protein_ids: list[str],
    data_root: Path,
    layer: str = "after_mp",
    device: torch.device | None = None,
    epochs: int = 30,
    lr: float = 1e-3,
) -> ChargeProbe:
    """
    Train the probe MLP on partial-charge prediction with backbone frozen.

    Proteins where the PQR atom count does not match the graph are skipped.
    Returns the trained probe moved to CPU.
    """
    from src.analysis.embedding_analysis import _load_graph  # avoid circular import

    if device is None:
        device = torch.device("cpu")

    probe     = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        n_proteins = 0

        for pid in protein_ids:
            paths = ProteinPaths(pid, data_root)
            if not paths.pqr_path.exists():
                continue

            charges = read_pqr_charges(paths.pqr_path)
            data    = _load_graph(pid, data_root)
            h_atom  = extract_atom_embeddings(model, data, layer=layer, device=device)

            if h_atom.shape[0] != len(charges):
                continue  # atom count mismatch between graph and PQR — skip

            h_atom = h_atom.to(device)
            y      = torch.tensor(charges, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss = loss_fn(probe(h_atom), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_proteins += 1

        print(f"  Epoch {epoch:3d}/{epochs}  train MSE: {total_loss / max(n_proteins, 1):.4f}")

    return probe.cpu()


def evaluate_probe(
    probe: ChargeProbe,
    model: nn.Module,
    protein_ids: list[str],
    data_root: Path,
    layer: str = "after_mp",
    device: torch.device | None = None,
) -> dict:
    """
    Evaluate the probe on a set of proteins.

    Returns:
        dict with "global" (rmse, mae, mean_r2, n_proteins, n_atoms) and
        "per_protein" ({protein_id: {rmse, mae, r2, n_atoms}}).
    """
    from src.analysis.embedding_analysis import _load_graph

    if device is None:
        device = torch.device("cpu")

    probe = probe.to(device)
    probe.eval()

    per_protein:  dict[str, dict] = {}
    total_sq_err  = 0.0
    total_abs_err = 0.0
    total_n       = 0

    with torch.no_grad():
        for pid in protein_ids:
            paths = ProteinPaths(pid, data_root)
            if not paths.pqr_path.exists():
                continue

            charges = read_pqr_charges(paths.pqr_path)
            data    = _load_graph(pid, data_root)
            h_atom  = extract_atom_embeddings(model, data, layer=layer, device=device)

            if h_atom.shape[0] != len(charges):
                continue

            h_atom = h_atom.to(device)
            y      = torch.tensor(charges, dtype=torch.float32, device=device)
            pred   = probe(h_atom)

            sq_err  = ((pred - y) ** 2).sum().item()
            abs_err = (pred - y).abs().sum().item()
            n       = len(charges)

            corr = float(np.corrcoef(charges, pred.cpu().numpy())[0, 1]) if n > 1 else 0.0

            per_protein[pid] = {
                "rmse":    float((sq_err / n) ** 0.5),
                "mae":     float(abs_err / n),
                "r2":      corr ** 2,
                "n_atoms": n,
            }
            total_sq_err  += sq_err
            total_abs_err += abs_err
            total_n       += n

    mean_r2 = float(np.mean([v["r2"] for v in per_protein.values()])) if per_protein else 0.0

    return {
        "global": {
            "rmse":       float((total_sq_err / max(total_n, 1)) ** 0.5),
            "mae":        float(total_abs_err / max(total_n, 1)),
            "mean_r2":    mean_r2,
            "n_proteins": len(per_protein),
            "n_atoms":    total_n,
        },
        "per_protein": per_protein,
    }
