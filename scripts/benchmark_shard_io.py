"""
scripts/benchmark_shard_io.py

Isolated I/O-only comparison: does reading the large-tier training set via
scripts/build_batch_aligned_shards.py's consolidated shard files actually
read faster than the original per-protein graph.pt files, on this
machine's storage? No GPU, no DDP, no training -- pure wall-clock file-read
time for one full pass over the same 1628-protein training set, measured
both ways, so the shard approach can be judged on real evidence before
touching the trainer.

Usage:
    conda activate pyg_env
    python scripts/benchmark_shard_io.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import ProteinPaths

DATA_ROOT = Path("/home/student/thesis/full_protein_dataset_size_large600")
SHARD_DIR = Path("/home/student/thesis/full_protein_dataset_size_large600_shards/train")
MANIFEST = DATA_ROOT / "split_manifest.json"


def benchmark_shards() -> tuple[float, int]:
    shard_paths = sorted(SHARD_DIR.glob("shard_*.pt"))
    print(f"Reading {len(shard_paths)} shard files...")
    t0 = time.perf_counter()
    n_graphs = 0
    for i, sp in enumerate(shard_paths):
        payload = torch.load(sp, weights_only=False)
        n_graphs += sum(len(b) for b in payload["batches"])
        if (i + 1) % 5 == 0 or i == len(shard_paths) - 1:
            print(f"  shard {i + 1}/{len(shard_paths)}  {time.perf_counter() - t0:.0f}s elapsed")
    elapsed = time.perf_counter() - t0
    return elapsed, n_graphs


def benchmark_original(protein_ids: list[str]) -> tuple[float, int]:
    print(f"Reading {len(protein_ids)} individual graph.pt files (original layout)...")
    t0 = time.perf_counter()
    n_graphs = 0
    for i, pid in enumerate(protein_ids):
        p = ProteinPaths(pid, DATA_ROOT)
        torch.load(p.graph_path(), weights_only=False)
        n_graphs += 1
        if (i + 1) % 200 == 0 or i == len(protein_ids) - 1:
            print(f"  {i + 1}/{len(protein_ids)}  {time.perf_counter() - t0:.0f}s elapsed")
    elapsed = time.perf_counter() - t0
    return elapsed, n_graphs


def main() -> None:
    train_ids = json.loads(MANIFEST.read_text())["splits"]["train"]

    print("=== Shard-based read (new layout) ===")
    shard_time, shard_n = benchmark_shards()
    print(f"Shard read: {shard_n} graphs in {shard_time:.0f}s ({shard_time / 60:.1f} min)\n")

    print("=== Original per-protein read (current layout) ===")
    orig_time, orig_n = benchmark_original(train_ids)
    print(f"Original read: {orig_n} graphs in {orig_time:.0f}s ({orig_time / 60:.1f} min)\n")

    print("=== Summary ===")
    print(f"Shard layout:    {shard_time:.0f}s for {shard_n} graphs ({shard_time / max(shard_n,1):.3f}s/graph)")
    print(f"Original layout: {orig_time:.0f}s for {orig_n} graphs ({orig_time / max(orig_n,1):.3f}s/graph)")
    speedup = orig_time / shard_time if shard_time > 0 else float("nan")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
