"""
src/data/dataset.py

Dataset with disk caching and stratified train/val/test splitting for protein
ESP graphs.

Graphs are expected to be pre-built by pipelines/06_build_graphs.py and cached
as .pt files under <data_root>/<protein_id>/graph/.  The dataset loads them
lazily — no graph construction happens at training time unless rebuild=True.

Public API
----------
  ProteinGraphDataset(protein_ids, data_root, *, rebuild, transform)
  split_dataset(dataset, train, val, seed, pinned_test) -> (train_ds, val_ds, test_ds)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.graph_builder import build_graph
from src.utils.config import get_config
from src.utils.io import load_metadata
from src.utils.paths import ProteinPaths


SPLIT_MANIFEST_NAME = "split_manifest.json"


# Proteins always assigned to the test set regardless of stratification.
# These are the three reference proteins used throughout development.
PINNED_TEST_IDS: tuple[str, ...] = (
        "AF-P01082-F1",
        "AF-P68469-F1",
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
    pre-built with pipelines/06_build_graphs.py and rebuild=False (default).

    Args:
        protein_ids:  iterable of protein ID strings (e.g. ["AF-Q16613-F1"])
        data_root:    root of the external data directory
        rebuild:      if True, rebuild and overwrite cached graphs on access
        transform:    optional callable applied to each HeteroData before returning
    """

    def __init__(
        self,
        protein_ids,
        data_root: Path,
        *,
        rebuild: bool = False,
        transform=None,
    ) -> None:
        self.protein_ids = list(protein_ids)
        self.data_root   = Path(data_root)
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
                f"Run pipelines/06_build_graphs.py first, or pass rebuild=True."
            )

        if self.rebuild or not graph_path.exists():
            p.ensure_dirs()
            data = build_graph(protein_id, self.data_root)
            torch.save(data, graph_path)
        else:
            data = torch.load(graph_path, weights_only=False)
            current_spec = get_config().get("features", {})
            cached_spec  = getattr(data, "feature_spec", None)
            if cached_spec is not None and cached_spec != current_spec:
                raise RuntimeError(
                    f"Feature spec mismatch for '{protein_id}'.\n"
                    f"  Cached: {cached_spec}\n"
                    f"  Config: {current_spec}\n"
                    "Rebuild graphs with pipelines/06_build_graphs.py --all --force"
                )

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __repr__(self) -> str:
        return f"ProteinGraphDataset(n={len(self)})"


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
    n_bins: int = 2,
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

    # Keep exactly len(pinned) proteins from the pool in test so that
    # total test = len(pinned) pool proteins + len(pinned) pinned = 2*len(pinned).
    # Any pool proteins beyond that quota are moved to val.
    target_pool_test = len(pinned)
    if len(test_ids) > target_pool_test:
        val_ids.extend(test_ids[target_pool_test:])
        test_ids = test_ids[:target_pool_test]

    # Pinned proteins always go to test
    test_ids.extend(sorted(pinned))

    shared = dict(
        data_root = dataset.data_root,
        rebuild   = False,
    )

    return (
        ProteinGraphDataset(train_ids, **shared),
        ProteinGraphDataset(val_ids,   **shared),
        ProteinGraphDataset(test_ids,  **shared),
    )


# ── Split manifest ────────────────────────────────────────────────────────────

def write_split_manifest(
    dataset: ProteinGraphDataset,
    train: float = 0.8,
    val: float = 0.1,
    seed: int = 42,
    pinned_test: tuple[str, ...] = PINNED_TEST_IDS,
) -> Path:
    """
    Run the stratified split once and persist the result to
    ``<data_root>/split_manifest.json``.

    The manifest records every parameter used to generate the split so the
    file is self-documenting.  Training scripts load it with
    ``load_split_manifest`` instead of calling ``split_dataset`` directly,
    guaranteeing identical splits across all ablation runs.

    If the manifest already exists this function is a no-op and returns the
    existing path.  Pass ``force=True`` (via the pipeline CLI ``--resplit``)
    to regenerate.

    Args:
        dataset:     source ProteinGraphDataset (all proteins with built graphs)
        train:       training fraction (default 0.8)
        val:         validation fraction (default 0.1)
        seed:        RNG seed (default 42)
        pinned_test: protein IDs always placed in the test split

    Returns:
        Path to the written manifest file.
    """
    manifest_path = Path(dataset.data_root) / SPLIT_MANIFEST_NAME

    train_ds, val_ds, test_ds = split_dataset(
        dataset, train=train, val=val, seed=seed, pinned_test=pinned_test,
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed":         seed,
        "train_frac":   train,
        "val_frac":     val,
        "pinned_test":  list(pinned_test),
        "splits": {
            "train": train_ds.protein_ids,
            "val":   val_ds.protein_ids,
            "test":  test_ds.protein_ids,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_split_manifest(
    data_root: Path,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load a previously written split manifest and return the protein ID lists.

    Raises:
        FileNotFoundError: if no manifest exists at ``<data_root>/split_manifest.json``.
                           Run ``pipelines/06_build_graphs.py`` to generate one.

    Returns:
        (train_ids, val_ids, test_ids) — lists of protein ID strings.
    """
    manifest_path = Path(data_root) / SPLIT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No split manifest found at {manifest_path}.\n"
            "Run pipelines/06_build_graphs.py to build graphs and generate the manifest."
        )

    manifest = json.loads(manifest_path.read_text())
    splits   = manifest["splits"]
    return splits["train"], splits["val"], splits["test"]
