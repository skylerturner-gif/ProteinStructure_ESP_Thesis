"""
src/data/dataset.py

Dataset with disk caching and stratified train/val/test splitting for protein
ESP graphs.

Graphs are expected to be pre-built by scripts/07_build_graphs.py and cached
as .pt files under <data_root>/<protein_id>/graph/.  The dataset loads them
lazily — no graph construction happens at training time unless rebuild=True.

Public API
----------
  ProteinGraphDataset(protein_ids, data_root, *, variant, sample_frac, rebuild, transform)
  split_dataset(dataset, train, val, seed, pinned_test) -> (train_ds, val_ds, test_ds)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.graph_builder import build_graph
from src.utils.io import load_metadata
from src.utils.paths import ProteinPaths


# Proteins always assigned to the test set regardless of stratification.
# These are the three reference proteins used throughout development.
PINNED_TEST_IDS: tuple[str, ...] = (
        "AF-P01082-F1",
        "AF-Q6P2D8-3-F1",
        "AF-Q2ES46-F1",
        "AF-Q6P5X5-F1",
        "AF-Q16613-F1",
        "AF-P28237-F1",
        "AF-B1KRT2-F1",
        "AF-Q6P2D8-5-F1",
        "AF-Q6P2D8-3-F1",
        "AF-B1WC58-F1",
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProteinGraphDataset(Dataset):
    """
    Lazy-loading dataset of pre-built protein HeteroData graphs.

    Each graph is loaded from:
        <data_root>/<protein_id>/graph/<protein_id>_graph.pt

    If a cached file is missing and rebuild=True, the graph is built on the
    fly and saved before returning.  During normal training, graphs should be
    pre-built with scripts/07_build_graphs.py and rebuild=False (default).

    Args:
        protein_ids:  iterable of protein ID strings (e.g. ["AF-Q16613-F1"])
        data_root:    root of the external data directory
        sample_frac:  fraction of mesh vertices used as query nodes
                      (only relevant when rebuilding)
        rebuild:      if True, rebuild and overwrite cached graphs on access
        transform:    optional callable applied to each HeteroData before returning
    """

    def __init__(
        self,
        protein_ids,
        data_root: Path,
        *,
        sample_frac: float = 0.05,
        rebuild: bool = False,
        transform=None,
    ) -> None:
        self.protein_ids = list(protein_ids)
        self.data_root   = Path(data_root)
        self.sample_frac = sample_frac
        self.rebuild     = rebuild
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int):
        protein_id = self.protein_ids[idx]
        p          = ProteinPaths(protein_id, self.data_root)
        graph_path = p.graph_path()

        if not graph_path.exists() and not self.rebuild:
            raise FileNotFoundError(
                f"No cached graph for '{protein_id}'. "
                f"Run scripts/07_build_graphs.py first, or pass rebuild=True."
            )

        if self.rebuild or not graph_path.exists():
            p.ensure_dirs()
            data = build_graph(
                protein_id, self.data_root,
                sample_frac=self.sample_frac,
            )
            torch.save(data, graph_path)
        else:
            data = torch.load(graph_path, weights_only=False)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __repr__(self) -> str:
        return (
            f"ProteinGraphDataset(n={len(self)}, sample_frac={self.sample_frac})"
        )


# ── Stratified train / val / test split ──────────────────────────────────────

def _load_strat_features(
    protein_ids: list[str],
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load sequence_length and net_charge from metadata for stratification.
    Missing values fall back to 0 / 0.0 so the protein is still included.
    """
    seq_lens    = np.zeros(len(protein_ids), dtype=float)
    net_charges = np.zeros(len(protein_ids), dtype=float)

    for i, pid in enumerate(protein_ids):
        try:
            meta            = load_metadata(pid, data_root)
            seq_lens[i]    = float(meta.get("sequence_length", 0))
            net_charges[i] = float(meta.get("net_charge", 0.0))
        except FileNotFoundError:
            pass  # leave defaults (0, 0.0) — protein lands in lowest bin

    return seq_lens, net_charges


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a quantile bin [0, n_bins)."""
    if len(np.unique(values)) <= 1:
        return np.zeros(len(values), dtype=int)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(values, edges)


def _stratified_split(
    ids: list[str],
    seq_lens: np.ndarray,
    net_charges: np.ndarray,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
    n_bins: int = 4,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split ids into train / val / test with stratification on
    binned sequence_length × net_charge.

    Proteins within each stratum are shuffled and distributed proportionally.
    Strata too small for a three-way split are assigned entirely to train.
    """
    seq_bins    = _quantile_bins(seq_lens,    n_bins)
    charge_bins = _quantile_bins(net_charges, n_bins)
    strata      = seq_bins * n_bins + charge_bins

    train_ids, val_ids, test_ids = [], [], []

    for stratum in np.unique(strata):
        idxs         = np.where(strata == stratum)[0]
        stratum_ids  = [ids[i] for i in idxs]
        rng.shuffle(stratum_ids)

        n       = len(stratum_ids)
        n_train = max(1, int(n * train_frac))
        n_val   = max(0, int(n * val_frac))

        # Don't exceed available proteins in the stratum
        n_train = min(n_train, n)
        n_val   = min(n_val,   n - n_train)

        train_ids.extend(stratum_ids[:n_train])
        val_ids.extend(stratum_ids[n_train : n_train + n_val])
        test_ids.extend(stratum_ids[n_train + n_val :])

    return train_ids, val_ids, test_ids


def split_dataset(
    dataset: ProteinGraphDataset,
    train: float = 0.8,
    val: float = 0.1,
    seed: int = 42,
    pinned_test: tuple[str, ...] = PINNED_TEST_IDS,
) -> tuple[ProteinGraphDataset, ProteinGraphDataset, ProteinGraphDataset]:
    """
    Stratified 80/10/10 split of a ProteinGraphDataset.

    Stratification is over binned sequence_length × net_charge read from
    per-protein metadata.  Proteins in pinned_test are always assigned to the
    test split, regardless of stratification.

    The test fraction is implicitly 1 - train - val.  Assign transforms after
    splitting:

        train_ds, val_ds, test_ds = split_dataset(full_ds)
        norm = NormalizeESP(mean, std)
        train_ds.transform = norm
        val_ds.transform   = norm

    Args:
        dataset:      source ProteinGraphDataset
        train:        training fraction (default 0.8)
        val:          validation fraction (default 0.1)
        seed:         RNG seed for reproducibility (default 42)
        pinned_test:  protein IDs always placed in the test split

    Returns:
        (train_ds, val_ds, test_ds) — each a new ProteinGraphDataset
    """
    rng = np.random.default_rng(seed)

    pinned   = set(pinned_test) & set(dataset.protein_ids)
    pool_ids = [pid for pid in dataset.protein_ids if pid not in pinned]

    seq_lens, net_charges = _load_strat_features(pool_ids, dataset.data_root)

    train_ids, val_ids, test_ids = _stratified_split(
        pool_ids, seq_lens, net_charges,
        train_frac=train, val_frac=val, rng=rng,
    )

    # Pinned proteins always go to test
    test_ids.extend(sorted(pinned))

    shared = dict(
        data_root   = dataset.data_root,
        sample_frac = dataset.sample_frac,
        rebuild     = False,
    )

    return (
        ProteinGraphDataset(train_ids, **shared),
        ProteinGraphDataset(val_ids,   **shared),
        ProteinGraphDataset(test_ids,  **shared),
    )
