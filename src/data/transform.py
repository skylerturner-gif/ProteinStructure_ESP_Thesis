"""
src/data/transform.py

Graph-level transforms applied to HeteroData objects before or during training.

Transforms are callable objects that accept a HeteroData and return a
(possibly modified) HeteroData.  They can be composed and passed as the
`transform` argument to ProteinGraphDataset.

Available transforms
--------------------
  NormalizeESP(mean, std)    — standardize query.y  (mean=0, std=1 after)
  RandomRotation()           — random SO(3) rotation applied to all pos fields
  Compose(transforms)        — chain multiple transforms

Helper functions
----------------
  compute_esp_stats(dataset) — scan a dataset and return (mean, std) of query.y
  compose(*transforms)       — shorthand for Compose([...])
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import HeteroData


# ── NormalizeESP ──────────────────────────────────────────────────────────────

class NormalizeESP:
    """
    Standardize query-point ESP targets: y ← (y − mean) / std.

    Fit once on the training split using compute_esp_stats(), then apply
    to all splits.  Store mean and std so they can be inverted at inference.

    Args:
        mean: scalar float — mean ESP value across the training set
        std:  scalar float — std  ESP value across the training set

    Example:
        mean, std = compute_esp_stats(train_dataset)
        norm = NormalizeESP(mean, std)
        train_ds = ProteinGraphDataset(..., transform=norm)
        val_ds   = ProteinGraphDataset(..., transform=norm)
    """

    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std  = float(std)

    def __call__(self, data: HeteroData) -> HeteroData:
        if hasattr(data["query"], "y"):
            data["query"].y = (data["query"].y - self.mean) / self.std
        return data

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Undo normalization — useful when converting predictions back to kT/e."""
        return y * self.std + self.mean

    def __repr__(self) -> str:
        return f"NormalizeESP(mean={self.mean:.4f}, std={self.std:.4f})"


# ── Helper: fit ESP stats from a dataset ─────────────────────────────────────

def compute_esp_stats(
    dataset,
    max_proteins: int | None = None,
) -> tuple[float, float]:
    """
    Compute the mean and standard deviation of query-point ESP values across
    a dataset.  Call on the training split only to avoid data leakage.

    Args:
        dataset:      ProteinGraphDataset (or any iterable of HeteroData)
        max_proteins: optional cap on number of proteins to scan (for speed)

    Returns:
        (mean, std) as Python floats

    Raises:
        ValueError: if no query.y found in any graph
    """
    values: list[torch.Tensor] = []
    limit = len(dataset) if max_proteins is None else min(max_proteins, len(dataset))

    for i in range(limit):
        data = dataset[i]
        if hasattr(data["query"], "y"):
            values.append(data["query"].y.float())

    if not values:
        raise ValueError(
            "No query.y found in any graph — make sure ESP files exist "
            "and the dataset variant matches available .npz outputs."
        )

    all_vals = torch.cat(values)
    return float(all_vals.mean()), float(all_vals.std())
