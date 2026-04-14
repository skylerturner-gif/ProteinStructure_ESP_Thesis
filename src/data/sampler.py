"""
src/data/sampler.py

DynamicBatchSampler — packs protein graphs into batches by edge budget.

Each batch holds as many proteins as fit within `max_num_edges` total edges.
Graphs that exceed the budget by themselves are always emitted as singletons —
they are never skipped.

Edge counts are read once from per-protein metadata (the ``num_edges_total``
field written by ``pipelines/06_build_graphs.py``) at construction time; no
graph files are opened during iteration.

Public API
----------
  DynamicBatchSampler(dataset, max_num_edges, *, shuffle, seed, drop_last)
  sampler.set_epoch(epoch)          # call before each training epoch
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from src.data.dataset import ProteinGraphDataset
from src.utils.io import load_metadata


class DynamicBatchSampler(Sampler[list[int]]):
    """
    Greedy edge-budget batch sampler for heterogeneous protein graphs.

    Iterates over dataset indices in (optionally shuffled) order and greedily
    accumulates them into a batch until the next graph would push the running
    edge total above ``max_num_edges``.

    Graphs larger than the budget are always emitted as singleton batches so
    no protein is silently dropped.

    Args:
        dataset:       ``ProteinGraphDataset`` — edge counts are read from
                       each protein's metadata at construction time.
        max_num_edges: maximum total edges per yielded batch.
        shuffle:       shuffle index order each epoch (default True for
                       training, pass False for validation / test).
        seed:          base RNG seed; the current epoch is added on each
                       reset to produce a distinct permutation per epoch.
        drop_last:     discard the final incomplete batch (default False).
    """

    def __init__(
        self,
        dataset: ProteinGraphDataset,
        max_num_edges: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.max_num_edges = max_num_edges
        self.shuffle       = shuffle
        self.seed          = seed
        self.drop_last     = drop_last
        self.rank          = rank
        self.world_size    = world_size
        self._epoch        = 0

        # Read edge counts once — O(n) metadata reads, no graph files opened.
        self._edge_counts: list[int] = _load_edge_counts(dataset)

    # ------------------------------------------------------------------
    # Epoch control (mirrors DistributedSampler.set_epoch convention)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch index so each epoch produces a distinct shuffle order.
        Call this at the start of every training epoch::

            for epoch in range(n_epochs):
                train_sampler.set_epoch(epoch)
                for batch in train_loader:
                    ...
        """
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self._edge_counts)

        if self.shuffle:
            rng     = np.random.default_rng(self.seed + self._epoch)
            indices = rng.permutation(n).tolist()
        else:
            indices = list(range(n))

        # Build the full batch list on every rank identically (same RNG seed).
        # Sharding at the protein level produces unequal batch counts per rank,
        # which causes NCCL all-reduce timeouts in DDP.  Sharding at the batch
        # level (below) guarantees every rank gets exactly the same count.
        all_batches: list[list[int]] = []
        current_batch: list[int]     = []
        running_edges: int           = 0

        for idx in indices:
            n_edges = self._edge_counts[idx]

            if n_edges > self.max_num_edges:
                # Oversized singleton: flush pending batch first, then emit.
                if current_batch:
                    all_batches.append(current_batch)
                    current_batch = []
                    running_edges = 0
                all_batches.append([idx])
                continue

            if running_edges + n_edges > self.max_num_edges and current_batch:
                # Budget would overflow: close current batch, start a new one.
                all_batches.append(current_batch)
                current_batch = [idx]
                running_edges = n_edges
            else:
                current_batch.append(idx)
                running_edges += n_edges

        if current_batch and not self.drop_last:
            all_batches.append(current_batch)

        # Shard at the batch level so every rank yields the same number of
        # batches.  Truncate to the nearest multiple of world_size first so
        # the stride-slice divides evenly.
        if self.world_size > 1:
            keep = (len(all_batches) // self.world_size) * self.world_size
            all_batches = all_batches[:keep]
            all_batches = all_batches[self.rank::self.world_size]

        yield from all_batches

    # ------------------------------------------------------------------
    # Length estimate
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Approximate number of batches.

        Exact when ``shuffle=False``; a lower bound when shuffled (packing
        efficiency varies with random ordering).
        """
        total = int(np.sum(self._edge_counts)) // self.world_size
        return max(1, math.ceil(total / self.max_num_edges))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_edge_counts(dataset: ProteinGraphDataset) -> list[int]:
    """
    Read ``num_edges_total`` from per-protein metadata for every protein in
    *dataset*.  Falls back to 1 if the field or file is missing so the protein
    is still included (it will form a tiny singleton batch).
    """
    counts: list[int] = []
    for pid in dataset.protein_ids:
        try:
            meta = load_metadata(pid, dataset.data_root)
            counts.append(int(meta.get("num_edges_total", 1)))
        except FileNotFoundError:
            counts.append(1)
    return counts
