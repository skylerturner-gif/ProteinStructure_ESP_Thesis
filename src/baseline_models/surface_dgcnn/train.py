"""Training script for Surface DGCNN baseline.

Batch size = 1 (one protein per GPU step). Supports 2-GPU DDP: each rank
processes a different protein per step and all-reduces gradients. Variable
point-cloud sizes are fine because no cross-GPU batching is needed.
Re-launches itself under torchrun when 2+ GPUs detected.

BF16 autocast matches the GNN baselines; kNN graph building stays FP32 on
CPU DataLoader workers (cached to disk after first pass).

Usage:
    conda run -n pyg_env python -m src.baseline_models.surface_dgcnn.train \\
        [--epochs 100] [--lr 1e-3] [--hidden-dim 128] [--k 20] \\
        [--out-dir model_eval/surface_dgcnn]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.baseline_models.surface_dgcnn.dataset import SurfaceDataset, collate_surface
from src.baseline_models.surface_dgcnn.model import SurfaceDGCNN
from src.data.dataset import load_split_manifest
from src.training.loss import ESPLoss, pearson_r
from src.utils.config import get_data_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pearson-weight", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--k", type=int, default=20, help="Neighbours in static kNN graph")
    p.add_argument("--n-query-k", type=int, default=8,
                   help="Surface points aggregated per query node")
    p.add_argument("--max-points", type=int, default=8192,
                   help="Random downsample target; 0 = keep all vertices")
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for shadow weights, matching the main GNN pipeline's "
                        "Trainer. best_model.pt stores EMA weights; latest_model.pt stores "
                        "raw weights.")
    p.add_argument("--out-dir", default=None,
                   help="Checkpoint output dir. Defaults to <data_root>/../checkpoints/baseline/surface_dgcnn_chem")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _val_epoch(
    model: nn.Module,
    val_ids: list[str],
    data_root: Path,
    args: argparse.Namespace,
    loss_fn: ESPLoss,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[float, float]:
    """Distributed val: each rank evaluates its own slice, then all-reduces."""
    rank_ids = [pid for i, pid in enumerate(val_ids) if i % world_size == rank]
    max_pts = args.max_points if args.max_points > 0 else None
    val_ds = SurfaceDataset(rank_ids, data_root, max_points=max_pts, k=args.k)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_surface, pin_memory=False)

    model.eval()
    total_loss, total_r, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            for i, pc in enumerate(batch["point_cloud"]):
                pc     = pc.to(device)
                ei     = batch["edge_index"][i].to(device)
                qsurf  = batch["query_surf"][i].to(device)
                target = batch["query_esp"][i].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(pc, ei, qsurf)
                fake_batch = torch.zeros(len(pred), dtype=torch.long, device=device)
                total_loss += loss_fn(pred.float(), target, fake_batch).item()
                total_r    += pearson_r(pred.float(), target).item()
                n          += 1

    t = torch.tensor([total_r, total_loss, float(n)], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_r    = (t[0] / t[2]).item()
    val_loss = (t[1] / t[2]).item()
    return val_loss, val_r


def main() -> None:
    # ── Auto multi-GPU: re-exec under torchrun when 2+ GPUs are available ─────
    if "LOCAL_RANK" not in os.environ:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            cmd = [sys.executable, "-m", "torch.distributed.run",
                   f"--nproc_per_node={n_gpus}",
                   "-m", "src.baseline_models.surface_dgcnn.train"] + sys.argv[1:]
            sys.exit(subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode)

    # ── DDP init ───────────────────────────────────────────────────────────────
    ddp        = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main    = (rank == 0)

    if ddp:
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")

    args = _parse_args()
    torch.manual_seed(args.seed + rank)

    data_root = Path(get_data_root())
    out_dir = (Path(args.out_dir) if args.out_dir
               else data_root.parent / "checkpoints" / "baseline" / "surface_dgcnn_chem")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Device: {device}" + (f"  (DDP world_size={world_size})" if ddp else ""))

    train_ids, val_ids, _ = load_split_manifest(data_root)
    max_pts = args.max_points if args.max_points > 0 else None
    train_ds = SurfaceDataset(train_ids, data_root, max_points=max_pts, k=args.k)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    ) if ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=1,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,
        collate_fn=collate_surface,
        pin_memory=False,
    )

    model = SurfaceDGCNN(
        in_features=train_ds.in_features,
        hidden_dim=args.hidden_dim,
        n_query_k=args.n_query_k,
    ).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    if is_main:
        raw = model.module if ddp else model
        n_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}")
        print(f"Train: {len(train_ds)}  Val: {len(val_ids)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.lr_patience, factor=0.5, min_lr=1e-6,
    )
    loss_fn = ESPLoss(pearson_weight=args.pearson_weight)

    # EMA shadow weights — same mechanism as src/training/trainer.py's Trainer:
    # updated after every optimizer.step(), swapped in for validation (so
    # is_best / best_model.pt reflect EMA quality, not raw-weight quality),
    # and swapped back out before the next training step.
    raw_model = model.module if ddp else model
    ema_state: dict = {k: v.clone().detach().float() for k, v in raw_model.state_dict().items()}

    best_val_r = -float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss, n = 0.0, 0
        for batch in train_loader:
            for i, pc in enumerate(batch["point_cloud"]):
                pc     = pc.to(device)
                ei     = batch["edge_index"][i].to(device)
                qsurf  = batch["query_surf"][i].to(device)
                target = batch["query_esp"][i].to(device)

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(pc, ei, qsurf)
                fake_batch = torch.zeros(len(pred), dtype=torch.long, device=device)
                loss = loss_fn(pred.float(), target, fake_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for k, v in raw_model.state_dict().items():
                        ema_state[k].mul_(args.ema_decay).add_(v.detach().float(), alpha=1.0 - args.ema_decay)
                train_loss += loss.item()
                n += 1

        # All-reduce train loss across ranks
        tl = torch.tensor([train_loss, float(n)], device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
        avg_train_loss = (tl[0] / tl[1]).item()

        # Validate under EMA weights, then restore raw weights for training.
        raw_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        raw_model.load_state_dict({k: v.to(raw_state[k].dtype) for k, v in ema_state.items()})
        val_loss, val_r = _val_epoch(
            model, val_ids, data_root, args, loss_fn, device, rank, world_size,
        )
        raw_model.load_state_dict(raw_state)

        if is_main:
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "train_loss": avg_train_loss,
                   "val_loss": val_loss, "val_r": val_r, "lr": lr_now}
            history.append(row)
            print(f"[{epoch:3d}/{args.epochs}] train={avg_train_loss:.4f}  "
                  f"val={val_loss:.4f}  r={val_r:.4f}  lr={lr_now:.2e}")

            csv_path = out_dir / "metrics.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as _cf:
                _w = csv.DictWriter(_cf, fieldnames=["epoch", "train_loss", "val_loss", "val_r", "lr"])
                if write_header:
                    _w.writeheader()
                _w.writerow(row)
                _cf.flush()

            # latest_model.pt: raw weights (so training can resume from a
            # consistent optimizer/weight state). best_model.pt: EMA weights.
            torch.save(raw_model.state_dict(), out_dir / "latest_model.pt")
            if val_r > best_val_r:
                best_val_r = val_r
                ema_state_out = {k: v.to(raw_model.state_dict()[k].dtype) for k, v in ema_state.items()}
                torch.save(ema_state_out, out_dir / "best_model.pt")
                print(f"    → new best val r={val_r:.4f}")
        else:
            scheduler.step(val_loss)

    if is_main:
        with open(out_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(out_dir / "config.json", "w") as f:
            cfg = vars(args)
            cfg["in_features"] = train_ds.in_features
            json.dump(cfg, f, indent=2)
        print(f"Done. Best val r={best_val_r:.4f}  checkpoints in {out_dir}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
