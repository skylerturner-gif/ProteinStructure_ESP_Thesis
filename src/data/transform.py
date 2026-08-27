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
  compute_esp_stats(dataset)                     — scan a dataset and return (mean, std) of query.y
  compute_esp_stats_parallel(ids, root, workers)  — same, via a worker pool over cached graphs
  load_or_compute_esp_stats(ids, root, ...)       — cache stats to <data_root>/esp_stats.json
  compose(*transforms)                            — shorthand for Compose([...])
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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


# ── Parallel + cached variants (large datasets) ───────────────────────────────
#
# compute_esp_stats() above loads every graph sequentially in the calling
# process — fine for a few hundred proteins, but on a multi-thousand-protein
# dataset it becomes the dominant cost at the start of *every* training run,
# since it runs before the DataLoader (and therefore num_workers) is ever
# touched. The functions below parallelise that one-time scan across a worker
# pool and cache the result to <data_root>/esp_stats.json so every subsequent
# run just reads two floats off disk instead of re-scanning the dataset.

ESP_STATS_FILENAME = "esp_stats.json"


def _esp_stats_worker(protein_id: str, data_root_str: str) -> tuple[float, float, int]:
    """
    Picklable run_parallel worker: return (sum, sumsq, count) of query.y.

    Reads esp/*.npz rather than the cached graph/*.pt. graph_builder.build_graph
    sets query.y = esp_data["esp_verts"][query_idx] verbatim (no further
    transform), so this is numerically identical to loading the full graph —
    but the npz is ~13x smaller and skips deserializing every atom/bond/
    radial/aq/qq tensor just to read one small array. Measured on this
    dataset: torch.load ~2.9s/protein vs np.load ~4ms/protein.
    """
    from src.utils.paths import ProteinPaths

    esp_path = ProteinPaths(protein_id, Path(data_root_str)).esp_path
    esp_data = np.load(esp_path)
    q_idx = esp_data["query_idx"].astype(np.int64)
    y = esp_data["esp_verts"][q_idx].astype(np.float64)
    return float(y.sum()), float((y * y).sum()), int(y.size)


def compute_esp_stats_parallel(
    protein_ids: Sequence[str],
    data_root,
    n_workers: int = 4,
) -> tuple[float, float]:
    """
    Parallel equivalent of compute_esp_stats() — scans cached esp/*.npz files
    across a worker pool (via src.utils.parallel.run_parallel) instead of
    sequentially in the calling process. Combines per-protein sum/sumsq/count
    rather than concatenating raw tensors, so peak memory and IPC payload
    stay small regardless of dataset size.

    Args:
        protein_ids: training-split protein IDs (e.g. from load_split_manifest)
        data_root:   dataset root containing <protein_id>/esp/*.npz
        n_workers:   worker-pool size

    Returns:
        (mean, std) as Python floats — numerically equivalent to compute_esp_stats()
    """
    from src.utils.parallel import run_parallel

    results = run_parallel(
        _esp_stats_worker,
        [(pid, str(data_root)) for pid in protein_ids],
        n_workers=max(1, n_workers),
        label="esp_stats",
    )

    total_sum = total_sumsq = 0.0
    total_n = 0
    for protein_id, outcome in results:
        if isinstance(outcome, Exception):
            raise RuntimeError(f"[{protein_id}] Failed computing ESP stats: {outcome}") from outcome
        s, ss, n = outcome
        total_sum   += s
        total_sumsq += ss
        total_n     += n

    if total_n == 0:
        raise ValueError(
            "No query.y found in any graph — make sure ESP files exist "
            "and the dataset variant matches available .npz outputs."
        )

    mean = total_sum / total_n
    var  = max(total_sumsq / total_n - mean * mean, 0.0)  # clamp: fp roundoff can nudge this slightly negative
    return mean, var ** 0.5


def load_or_compute_esp_stats(
    train_ids: Sequence[str],
    data_root,
    *,
    n_workers: int = 4,
    force: bool = False,
) -> tuple[float, float]:
    """
    Load ESP normalisation stats cached at <data_root>/esp_stats.json, or
    compute (in parallel) and cache them if the file is missing or force=True.

    Stats are fit deterministically from the training split alone, so caching
    once per data_root is correct as long as the split manifest doesn't
    change — pass force=True after a --resplit to regenerate.

    The cache is checked against len(train_ids) and raises rather than
    silently reusing stats fit on a different split — e.g. after a
    06_build_graphs.py --resplit (dataset growth, reseed, etc.) changes the
    training set but esp_stats.json is left over from the old one.

    In DDP, call this from rank 0 only, then have the other ranks call it
    again (any n_workers, e.g. 1) after a barrier — the cache is guaranteed
    to exist by then, so it's just a fast JSON read on every other rank.
    """
    data_root  = Path(data_root)
    stats_path = data_root / ESP_STATS_FILENAME

    if stats_path.exists() and not force:
        cached = json.loads(stats_path.read_text())
        if cached.get("n_proteins") != len(train_ids):
            raise RuntimeError(
                f"{stats_path} is stale: cached for {cached.get('n_proteins')} "
                f"training proteins, but the current split has {len(train_ids)}. "
                "The split manifest was likely regenerated (e.g. --resplit) since "
                "this cache was written. Re-run with force=True to recompute, or "
                f"delete {stats_path}."
            )
        return cached["esp_mean"], cached["esp_std"]

    mean, std = compute_esp_stats_parallel(train_ids, data_root, n_workers=n_workers)

    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_proteins":  len(train_ids),
        "esp_mean":    mean,
        "esp_std":     std,
    }
    tmp_path = stats_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.rename(stats_path)  # atomic on Linux — no reader ever sees a partial write
    return mean, std
