"""
scripts/build_batch_aligned_shards.py

I/O mitigation for the size-generalization pilot's large-only training set:
DynamicBatchSampler currently shuffles at the individual-protein level, so
every epoch does ~1600 separate file opens on graphs that are already large
(33-41MB average for the >=600aa tier) -- on this machine's storage, cold
reads of these files range from ~3 MB/s to ~350 MB/s seemingly at random,
and DDP's per-batch gradient all-reduce means one rank's slow read stalls
the other rank too. Measured real cost: ~43-53 min/epoch, actually *worse*
than the full 6768-protein champion run's ~33-35 min/epoch steady state,
despite reading less than half as much data.

This script fixes the *access pattern*, not the data: it materializes the
real DynamicBatchSampler batch assignment (same greedy edge-budget packing
used at training time, same seed) once, groups many consecutive batches
into a handful of shard files, and saves each shard as a single sequential
torch.save(). Reading 30ish large sequential files should be far less
exposed to this storage's per-request latency variance than reading ~1600
scattered ones -- to be confirmed empirically by
scripts/benchmark_shard_io.py, not assumed here.

Known, deliberate tradeoff: batch *composition* is now fixed across all
epochs (only batch *order* is still shuffled at train time) -- some loss of
SGD diversity relative to the champions' fully-reshuffled training. This is
the exploratory-pilot version; a shuffle-buffer design that preserves full
per-epoch reshuffling is a possible follow-up if this proves worthwhile but
insufficient.

Usage:
    conda activate pyg_env
    python scripts/build_batch_aligned_shards.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProteinGraphDataset
from src.data.sampler import DynamicBatchSampler
from src.utils.paths import ProteinPaths

DATA_ROOT = Path("/home/student/thesis/full_protein_dataset_size_large600")
SHARD_ROOT = Path("/home/student/thesis/full_protein_dataset_size_large600_shards")
MAX_EDGES_PER_BATCH = 1_500_000
BATCHES_PER_SHARD = 25
SEED = 42


def build_shards_for_split(split_name: str, protein_ids: list[str]) -> None:
    out_dir = SHARD_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ProteinGraphDataset(protein_ids, DATA_ROOT)
    sampler = DynamicBatchSampler(ds, max_num_edges=MAX_EDGES_PER_BATCH, shuffle=True, seed=SEED)
    sampler.set_epoch(0)
    batches = list(iter(sampler))  # list[list[int]] -- dataset indices per batch
    print(f"[{split_name}] {len(protein_ids)} proteins -> {len(batches)} batches "
          f"(mean {sum(len(b) for b in batches) / len(batches):.2f} graphs/batch)")

    n_shards = (len(batches) + BATCHES_PER_SHARD - 1) // BATCHES_PER_SHARD
    t0 = time.perf_counter()
    for shard_idx in range(n_shards):
        shard_batches_idx = batches[shard_idx * BATCHES_PER_SHARD:(shard_idx + 1) * BATCHES_PER_SHARD]
        shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
        if shard_path.exists():
            continue

        shard_batches = []
        shard_batch_ids = []
        for batch_idx_list in shard_batches_idx:
            graphs, pids = [], []
            for idx in batch_idx_list:
                pid = protein_ids[idx]
                p = ProteinPaths(pid, DATA_ROOT)
                data = torch.load(p.graph_path(), weights_only=False)
                graphs.append(data)
                pids.append(pid)
            shard_batches.append(graphs)
            shard_batch_ids.append(pids)

        torch.save({"batches": shard_batches, "batch_protein_ids": shard_batch_ids}, shard_path)
        elapsed = time.perf_counter() - t0
        print(f"  [{split_name}] shard {shard_idx + 1}/{n_shards} written "
              f"({sum(len(b) for b in shard_batches_idx)} graphs) -- {elapsed:.0f}s elapsed")

    print(f"[{split_name}] done -> {out_dir} ({n_shards} shards)")


def main() -> None:
    import json
    manifest = json.loads((DATA_ROOT / "split_manifest.json").read_text())
    train_ids = manifest["splits"]["train"]

    build_shards_for_split("train", train_ids)


if __name__ == "__main__":
    main()
