"""
scripts/vram_smoke_5ep.py

Run 5 real training epochs for each of the 4 trained models (attention_qq6,
distance_qq6, attention_bf16, distance_bf16) while sampling nvidia-smi every
second. No checkpoints are written — existing best_model.pt files are never
touched.

Runs with full 2-GPU DDP so NCCL buffers, backward passes, and gradient
accumulation all contribute to the VRAM numbers, matching actual training.

Usage
-----
    # Run all 4 models sequentially (recommended)
    conda run -n pyg_env python scripts/vram_smoke_5ep.py --all

    # Summarise results after all runs (also called automatically by --all)
    conda run -n pyg_env python scripts/vram_smoke_5ep.py --summarize

    # Run a single model manually (torchrun required)
    conda run -n pyg_env python -m torch.distributed.run --nproc_per_node=2 \\
        scripts/vram_smoke_5ep.py --name attention_qq6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import ProteinGraphDataset, load_split_manifest
from src.data.sampler import DynamicBatchSampler
from src.data.transform import NormalizeESP, compute_esp_stats
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.training.loss import ESPLoss
from src.utils.config import get_data_root

CKPT_ROOT = Path("/home/student/thesis/checkpoints")
LOG_DIR   = _ROOT / "outputs" / "vram_smoke"

SMOKE_MODELS = [
    {"name": "attention_qq6",  "model": "attention", "bf16": False},
    {"name": "distance_qq6",   "model": "distance",  "bf16": False},
    {"name": "attention_bf16", "model": "attention", "bf16": True},
    {"name": "distance_bf16",  "model": "distance",  "bf16": True},
]

MAX_EDGES = 1_600_000
N_EPOCHS  = 5


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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
    return model.to(device)


# ── nvidia-smi monitor ────────────────────────────────────────────────────────

def _start_nvsmi(log_path: Path) -> subprocess.Popen:
    """Start continuous nvidia-smi sampling (1 s interval) → log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-l", "1",
        ],
        stdout=open(log_path, "w"),
        stderr=subprocess.DEVNULL,
    )


def _parse_nvsmi(log_path: Path, epoch_windows: list[tuple[float, float]]) -> list[dict]:
    """
    Parse nvidia-smi CSV log. Returns per-epoch peak for each GPU.

    epoch_windows: list of (start_unix, end_unix) tuples, one per epoch.
    """
    results = []
    with open(log_path) as f:
        lines = f.readlines()

    for ep_idx, (t_start, t_end) in enumerate(epoch_windows):
        peak: dict[int, int] = {}   # gpu_idx → max MiB seen
        for line in lines:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 5:
                continue
            try:
                # timestamp format: "2026/07/15 19:30:01.123" — local time, no tz
                ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            if not (t_start <= ts <= t_end):
                continue
            gpu_idx  = int(parts[1])
            mem_used = int(parts[2])
            peak[gpu_idx] = max(peak.get(gpu_idx, 0), mem_used)

        results.append({
            "epoch":       ep_idx + 1,
            "peak_mib":    peak,
            "peak_mib_max": max(peak.values()) if peak else 0,
        })
    return results


# ── Worker: one model, N epochs ───────────────────────────────────────────────

def run_worker(name: str, bf16: bool, n_epochs: int) -> None:
    ddp        = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device     = torch.device(f"cuda:{local_rank}")
    is_main    = (rank == 0)

    torch.cuda.set_device(device)
    if ddp:
        torch.distributed.init_process_group(backend="nccl")

    if is_main:
        print(f"\n{'='*60}")
        print(f"  VRAM smoke: {name}  ({n_epochs} epochs)  bf16={bf16}")
        print(f"{'='*60}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_root = get_data_root()
    train_ids, _, _ = load_split_manifest(data_root)
    train_ds = ProteinGraphDataset(train_ids, data_root, rebuild=False)
    esp_mean, esp_std = compute_esp_stats(train_ds)
    train_ds.transform = NormalizeESP(esp_mean, esp_std)

    sampler = DynamicBatchSampler(
        train_ds, MAX_EDGES, shuffle=True, drop_last=True,
        rank=rank, world_size=world_size,
    )
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)

    if is_main:
        print(f"  {len(train_ds)} proteins  |  ~{len(sampler)} batches/rank")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    ckpt_path = CKPT_ROOT / name / "best_model.pt"
    model = _build_model(ckpt_path, device)
    if ddp:
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for buf in model.buffers():
            torch.distributed.broadcast(buf, src=0)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn   = ESPLoss(pearson_weight=0.1)

    # ── nvidia-smi monitor (rank 0 only) ──────────────────────────────────────
    nvsmi_log  = LOG_DIR / f"{name}_nvsmi.csv"
    nvsmi_proc = None
    if is_main:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        nvsmi_proc = _start_nvsmi(nvsmi_log)
        time.sleep(1.0)   # let it write a baseline reading

    epoch_windows: list[tuple[float, float]] = []
    torch_peaks:   list[dict]                = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        model.train()
        if is_main:
            torch.cuda.reset_peak_memory_stats(device)

        t_start = time.time()

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                pred   = model(data)
                target = data["query"].y
                batch  = data["query"].batch
                loss   = loss_fn(pred, target, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ddp:
            torch.distributed.barrier()

        t_end = time.time()

        if is_main:
            alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            resrv_gb = torch.cuda.max_memory_reserved(device)  / 1024**3
            epoch_windows.append((t_start, t_end))
            torch_peaks.append({
                "epoch":        epoch,
                "allocated_gb": round(alloc_gb, 3),
                "reserved_gb":  round(resrv_gb, 3),
                "wall_s":       round(t_end - t_start, 1),
            })
            print(
                f"  Epoch {epoch}/{n_epochs}  "
                f"alloc={alloc_gb:.2f}GB  resrv={resrv_gb:.2f}GB  "
                f"t={t_end-t_start:.0f}s"
            )

    # ── Stop monitor + parse log ──────────────────────────────────────────────
    if is_main:
        if nvsmi_proc:
            nvsmi_proc.terminate()
            nvsmi_proc.wait()

        time.sleep(0.5)   # let file flush
        nvsmi_epochs = _parse_nvsmi(nvsmi_log, epoch_windows)

        # Merge torch + nvsmi per-epoch results
        merged = []
        for tp, ns in zip(torch_peaks, nvsmi_epochs):
            merged.append({**tp, "nvsmi_peak_mib": ns["peak_mib"],
                           "nvsmi_max_mib": ns["peak_mib_max"]})

        result = {"name": name, "bf16": bf16, "epochs": merged}
        out_path = LOG_DIR / f"{name}_results.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n  Results saved → {out_path}\n")

    if ddp:
        torch.distributed.destroy_process_group()


# ── Summarise ─────────────────────────────────────────────────────────────────

def summarise() -> None:
    print(f"\n{'='*78}")
    print("  VRAM Smoke Test — Summary (5-epoch averages, GPU 0 & 1)")
    print(f"{'='*78}")

    hdr = (
        f"{'Model':<22}  {'Prec':<5}  "
        f"{'Alloc avg':>10}  {'Rsvd avg':>9}  "
        f"{'NvSMI GPU0':>11}  {'NvSMI GPU1':>11}  "
        f"{'Epoch t':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for spec in SMOKE_MODELS:
        path = LOG_DIR / f"{spec['name']}_results.json"
        if not path.exists():
            print(f"  {spec['name']:<22}  (no results yet)")
            continue
        d      = json.loads(path.read_text())
        epochs = d["epochs"]
        prec   = "BF16" if d["bf16"] else "FP32"

        avg_alloc = sum(e["allocated_gb"]  for e in epochs) / len(epochs)
        avg_rsvd  = sum(e["reserved_gb"]   for e in epochs) / len(epochs)
        avg_time  = sum(e["wall_s"]        for e in epochs) / len(epochs)
        avg_g0    = sum(e["nvsmi_peak_mib"].get("0", 0) for e in epochs) / len(epochs)
        avg_g1    = sum(e["nvsmi_peak_mib"].get("1", 0) for e in epochs) / len(epochs)

        print(
            f"  {spec['name']:<22}  {prec:<5}  "
            f"{avg_alloc:>8.2f}GB  {avg_rsvd:>7.2f}GB  "
            f"{avg_g0/1024:>9.2f}GB  {avg_g1/1024:>9.2f}GB  "
            f"{avg_time:>6.0f}s"
        )

    print()

    # Per-epoch detail for each model
    for spec in SMOKE_MODELS:
        path = LOG_DIR / f"{spec['name']}_results.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        print(f"\n  {spec['name']}  ({'BF16' if d['bf16'] else 'FP32'})")
        print(f"  {'Ep':<4}  {'Alloc':>8}  {'Rsvd':>8}  {'GPU0 (nvsmi)':>14}  {'GPU1 (nvsmi)':>14}  {'Time':>6}")
        for e in d["epochs"]:
            g0 = e["nvsmi_peak_mib"].get("0", 0) / 1024
            g1 = e["nvsmi_peak_mib"].get("1", 0) / 1024
            print(
                f"  {e['epoch']:<4}  "
                f"{e['allocated_gb']:>6.2f}GB  "
                f"{e['reserved_gb']:>6.2f}GB  "
                f"{g0:>12.2f}GB  "
                f"{g1:>12.2f}GB  "
                f"{e['wall_s']:>5.0f}s"
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",       action="store_true",
                       help="Run all 4 models sequentially via torchrun.")
    group.add_argument("--name",      type=str,
                       help="Run a single model by checkpoint name "
                            "(e.g. attention_qq6). Call via torchrun.")
    group.add_argument("--summarize", action="store_true",
                       help="Print summary table from saved JSON results.")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--bf16",   action="store_true", default=False,
                        help="Enable BF16 autocast (set automatically by --all).")
    args = parser.parse_args()

    if args.summarize:
        summarise()
        return

    if args.all:
        n_gpus = torch.cuda.device_count()
        for spec in SMOKE_MODELS:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={n_gpus}",
                __file__,
                "--name", spec["name"],
                "--epochs", str(args.epochs),
            ]
            if spec["bf16"]:
                cmd.append("--bf16")
            print(f"\n[smoke] Launching: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[smoke] {spec['name']} failed (exit {result.returncode}) — continuing.")
        summarise()
        return

    # Worker path — called via torchrun (LOCAL_RANK will be set)
    run_worker(args.name, bf16=args.bf16, n_epochs=args.epochs)


if __name__ == "__main__":
    main()
