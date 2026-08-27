"""
scripts/smoke_test_peak_batch.py

Find the highest-edge training batch (worst-case VRAM load), then run a
no-grad forward pass through all four trained models against that exact batch
and report peak allocated and reserved VRAM per model.

Runs on a single GPU (GPU 0) — no DDP — so you see per-device numbers
without NCCL overhead splitting the picture.

Usage
-----
    conda run -n pyg_env python scripts/smoke_test_peak_batch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.dataset import ProteinGraphDataset, load_split_manifest
from src.data.sampler import DynamicBatchSampler
from src.data.transform import NormalizeESP, compute_esp_stats
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.utils.config import get_config, get_data_root

CKPT_ROOT   = Path("/home/student/thesis/checkpoints")
MAX_EDGES   = 1_600_000
DEVICE      = torch.device("cuda:0")

MODELS = [
    {"name": "attention_qq6",  "bf16": False},
    {"name": "distance_qq6",   "bf16": False},
    {"name": "attention_bf16", "bf16": True},
    {"name": "distance_bf16",  "bf16": True},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _total_edges(batch) -> int:
    total = 0
    for store in batch.edge_stores:
        if hasattr(store, "edge_index"):
            total += store.edge_index.shape[1]
    return total


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc   = ckpt["model_config"]
    fs   = ckpt.get("feature_spec") or {}
    common = dict(
        hidden_dim            = mc["hidden_dim"],
        n_rbf                 = mc["n_rbf"],
        n_bond_radial_rounds  = mc["n_bond_radial_rounds"],
        n_aq_rounds           = mc["n_aq_rounds"],
        n_qq_rounds           = mc["n_qq_rounds"],
        agg                   = mc.get("agg", "multi"),
        use_element_embedding = mc.get("use_element_embedding", True),
        use_residue_embedding = mc.get("use_residue_embedding", True),
        use_bond_edges        = mc.get("use_bond_edges", True),
        use_radial_edges      = mc.get("use_radial_edges", True),
        has_curvature         = fs.get("query_curvature", False),
        has_normal            = fs.get("query_normal", False),
    )
    if ckpt["model_name"] == "attention":
        model = AttentionESPN(**common, n_heads=mc.get("n_heads", 4))
    else:
        model = DistanceESPN(**common)
    model.load_state_dict(ckpt["model_state"])
    return model.to(DEVICE).eval()


def _measure(model: torch.nn.Module, batch, bf16: bool) -> dict[str, float]:
    batch = batch.to(DEVICE)
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            _ = model(batch)
    torch.cuda.synchronize(DEVICE)
    return {
        "allocated_gb": torch.cuda.max_memory_allocated(DEVICE) / 1024**3,
        "reserved_gb":  torch.cuda.max_memory_reserved(DEVICE)  / 1024**3,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data_root = get_data_root()

    print("Loading train split...")
    train_ids, _, _ = load_split_manifest(data_root)
    train_ds = ProteinGraphDataset(train_ids, data_root, rebuild=False)
    esp_mean, esp_std = compute_esp_stats(train_ds)
    train_ds.transform = NormalizeESP(esp_mean, esp_std)

    # ── Find peak-edge batch ──────────────────────────────────────────────────
    sampler = DynamicBatchSampler(
        train_ds, MAX_EDGES, shuffle=False, drop_last=False,
    )
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)

    print(f"Scanning {len(sampler)} batches for max-edge batch "
          f"(budget: {MAX_EDGES:,} edges)...")

    peak_edges = 0
    peak_batch = None
    peak_n_proteins = 0

    for i, batch in enumerate(loader):
        n_edges = _total_edges(batch)
        if n_edges > peak_edges:
            peak_edges     = n_edges
            peak_batch     = batch          # CPU — cheap to hold
            peak_n_proteins = int(batch["query"].batch.max().item()) + 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sampler)}  peak so far: {peak_edges:,}", end="\r")

    print(f"\nPeak batch: {peak_edges:,} edges across {peak_n_proteins} protein(s)")
    n_atoms  = peak_batch["atom"].pos.shape[0]
    n_query  = peak_batch["query"].pos.shape[0]
    print(f"  atom nodes: {n_atoms:,}   query nodes: {n_query:,}\n")

    # ── Forward-pass smoke test ───────────────────────────────────────────────
    header = f"{'Model':<22}  {'Precision':<10}  {'Allocated':>12}  {'Reserved':>10}"
    print(header)
    print("-" * len(header))

    for spec in MODELS:
        ckpt = CKPT_ROOT / spec["name"] / "best_model.pt"
        precision = "BF16" if spec["bf16"] else "FP32"
        if not ckpt.exists():
            print(f"{spec['name']:<22}  {precision:<10}  {'(no checkpoint)':>12}")
            continue

        model = _load_model(ckpt)
        torch.cuda.empty_cache()

        m = _measure(model, peak_batch, bf16=spec["bf16"])

        del model
        torch.cuda.empty_cache()

        print(
            f"{spec['name']:<22}  {precision:<10}  "
            f"{m['allocated_gb']:>10.2f}GB  {m['reserved_gb']:>8.2f}GB"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
