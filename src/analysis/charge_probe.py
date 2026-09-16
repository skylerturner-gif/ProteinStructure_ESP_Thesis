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
    "load_parse_charge_reference",
]

# PDB2PQR's PARSE force-field dictionary, in the protein_esp conda env (this
# codebase's PQR-generation environment — see CLAUDE.md's "PDB2PQR (PARSE
# force field, pH 7.0)" pipeline stage).
DEFAULT_PARSE_DAT_PATH = Path(
    "/home/student/miniconda3/envs/protein_esp/lib/python3.10/site-packages/pdb2pqr/dat/PARSE.DAT"
)


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


def _load_protein_embeddings(
    protein_id: str,
    model: nn.Module,
    data_root: Path,
    layer: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load one protein's frozen-backbone atom embeddings + PQR charges.

    Shared by train_probe and evaluate_probe so both extract embeddings the
    same way and skip atom-count mismatches identically.

    Returns (h_atom, charges), both CPU tensors, or None if the PQR is
    missing or its atom count doesn't match the cached graph's atom count.
    """
    from src.analysis.embedding_analysis import _load_graph  # avoid circular import

    paths = ProteinPaths(protein_id, data_root)
    if not paths.pqr_path.exists():
        return None

    charges = read_pqr_charges(paths.pqr_path)
    data    = _load_graph(protein_id, data_root)
    h_atom  = extract_atom_embeddings(model, data, layer=layer, device=device)

    if h_atom.shape[0] != len(charges):
        return None

    return h_atom, torch.tensor(charges, dtype=torch.float32)


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
    Embeddings + charges are extracted from the frozen backbone ONCE up
    front (not once per epoch — the backbone never changes, so re-loading
    every protein's graph and rerunning the forward pass every epoch is
    pure waste) and cached on CPU for the remaining epochs; each epoch just
    moves the small cached tensors to `device` as needed.

    Returns the trained probe moved to CPU.
    """
    if device is None:
        device = torch.device("cpu")

    probe = probe.to(device)

    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for pid in protein_ids:
        result = _load_protein_embeddings(pid, model, data_root, layer, device)
        if result is not None:
            cached.append(result)
    print(f"  Precomputed embeddings for {len(cached)}/{len(protein_ids)} proteins "
          f"(skipped: missing PQR or atom-count mismatch)")

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0

        for h_atom, y in cached:
            h_atom, y = h_atom.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(probe(h_atom), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch:3d}/{epochs}  train MSE: {total_loss / max(len(cached), 1):.4f}")

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
            result = _load_protein_embeddings(pid, model, data_root, layer, device)
            if result is None:
                continue
            h_atom, y = result
            h_atom, y = h_atom.to(device), y.to(device)
            pred = probe(h_atom)

            sq_err  = ((pred - y) ** 2).sum().item()
            abs_err = (pred - y).abs().sum().item()
            n       = len(y)

            corr = float(np.corrcoef(y.cpu().numpy(), pred.cpu().numpy())[0, 1]) if n > 1 else 0.0

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


def load_parse_charge_reference(
    parse_dat_path: Path = DEFAULT_PARSE_DAT_PATH,
) -> dict[tuple[str, str], float]:
    """
    Parse PDB2PQR's PARSE force-field dictionary into a
    {(resname, atomname): charge} lookup, protein residues only (skips the
    5-field RNA rows).

    For documentation/labeling use only — NOT a classification key. PDB2PQR
    normalizes protonation-variant residue names (e.g. HID/HIE/HIP each get
    PARSE-appropriate charges internally but are always written out as
    "HIS" in the PQR), so a name-keyed lookup can't reliably recover which
    tautomer's charge a given *observed* atom actually has. Classify atoms
    by their own observed charge instead; use this table only to describe
    what a given charge value typically corresponds to.
    """
    reference: dict[tuple[str, str], float] = {}
    with open(parse_dat_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 4:
                continue  # skip RNA rows (5 fields) and anything malformed
            resname, atomname, charge, _radius = fields
            reference[(resname, atomname)] = float(charge)
    return reference
