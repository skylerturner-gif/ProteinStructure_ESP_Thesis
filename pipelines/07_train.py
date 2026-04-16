"""
pipelines/07_train.py

Train DistanceESPN or AttentionESPN on ESP surface graphs.

Usage
-----
    # Train on all proteins, distance model
    python pipelines/07_train.py --model distance --all

    # Train on filtered proteins, attention model, resume from checkpoint
    python pipelines/07_train.py --model attention --filter --min-plddt 70 \\
        --epochs 150 --resume checkpoints/run_01/latest_model.pt

    # Override architecture defaults
    python pipelines/07_train.py --model attention --all \\
        --hidden-dim 256 --n-heads 8 --n-aq-rounds 4

Checkpoints are saved to <checkpoint-dir>/<model>/ and contain model weights,
optimizer/scheduler state, and ESP normalization statistics (esp_mean, esp_std).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import ProteinGraphDataset, load_split_manifest
from src.data.sampler import DynamicBatchSampler
from src.data.transform import NormalizeESP, compute_esp_stats
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.training.loss import ESPLoss
from src.training.trainer import Trainer, evaluate_test as run_evaluate_test
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger


def build_model(args, device: torch.device, feat_cfg: dict, model_cfg: dict):
    common = dict(
        hidden_dim           = args.hidden_dim,
        n_rbf                = args.n_rbf,
        n_bond_radial_rounds = args.n_bond_radial_rounds,
        n_aq_rounds          = args.n_aq_rounds,
        n_qq_rounds          = args.n_qq_rounds,
        multi_agg            = model_cfg.get("multi_agg",      False),
        has_curvature        = feat_cfg.get("query_curvature", False),
        has_normal           = feat_cfg.get("query_normal",    False),
    )
    if args.model == "distance":
        model = DistanceESPN(**common)
    else:
        model = AttentionESPN(**common, n_heads=args.n_heads)
    return model.to(device)


def main() -> None:
    # ── Auto multi-GPU ────────────────────────────────────────────────────────
    # If we are not already inside a torchrun launch (LOCAL_RANK not set) and
    # multiple CUDA devices are available, re-exec this script via torchrun so
    # that DDP is configured automatically.  This makes direct `python 07_train.py`
    # invocations behave the same as going through model_pipeline.py.
    if "LOCAL_RANK" not in os.environ:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={n_gpus}",
            ] + sys.argv
            sys.exit(__import__("subprocess").run(cmd).returncode)

    parser = argparse.ArgumentParser(
        description="Train DistanceESPN or AttentionESPN on protein ESP graphs."
    )

    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data_root from config.yaml.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", choices=["distance", "attention"], required=True,
        help="Model architecture to train.",
    )
    # ── Architecture ──────────────────────────────────────────────────────────
    parser.add_argument("--hidden-dim",        type=int,   default=256)
    parser.add_argument("--n-rbf",             type=int,   default=16)
    parser.add_argument("--n-heads",           type=int,   default=4,
                        help="Attention heads (attention model only).")
    parser.add_argument("--n-bond-radial-rounds", type=int,   default=2)
    parser.add_argument("--n-aq-rounds",       type=int,   default=3)
    parser.add_argument("--n-qq-rounds",       type=int,   default=2)

    # ── Graph construction ────────────────────────────────────────────────────
    parser.add_argument("--rebuild-graphs", action="store_true",
                        help="Ignore cached graphs and rebuild from scratch.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--max-edges-per-batch", type=int,  default=200_000,
                        help="Edge budget per batch for DynamicBatchSampler.")
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--pearson-weight", type=float, default=0.1,
                        help="Weight for the Pearson correlation loss term.")
    parser.add_argument("--protein-weighted", action="store_true", default=False,
                        help="Weight MSE equally per protein (not per node) to reduce "
                             "large-protein dominance in greedy batches.")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Accumulate gradients over N batches before stepping "
                             "(1 = disabled, 4 = accumulate 4 batches).")
    parser.add_argument("--clip-grad",      type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable).")
    parser.add_argument("--lr-scheduler",   type=str,   default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR scheduler: cosine annealing (default) or ReduceLROnPlateau.")
    parser.add_argument("--lr-min",        type=float, default=1e-6,
                        help="Minimum LR for cosine annealing (eta_min).")
    parser.add_argument("--lr-patience",   type=int,   default=15,
                        help="ReduceLROnPlateau patience in epochs (plateau scheduler only).")

    # ── I/O ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory for checkpoints. Defaults to <data_root>/../checkpoints/<model>[_suffix].",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Label appended to the default checkpoint dir name, e.g. "
             "'base' → checkpoints/attention_base. No effect if --checkpoint-dir is set.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (0 = main process).")

    # ── Inject config.yaml values as defaults (CLI still overrides) ──────────
    # Priority: CLI flag > config.yaml > argparse default
    _cfg       = get_config()
    _model_cfg = _cfg.get("model",    {})
    _train_cfg = _cfg.get("training", {})

    _config_defaults = {k: v for k, v in {
        "hidden_dim":           _model_cfg.get("hidden_dim"),
        "n_rbf":                _model_cfg.get("n_rbf"),
        "n_heads":              _model_cfg.get("n_heads"),
        "n_bond_radial_rounds": _model_cfg.get("n_bond_radial_rounds"),
        "n_aq_rounds":          _model_cfg.get("n_aq_rounds"),
        "n_qq_rounds":          _model_cfg.get("n_qq_rounds"),
        "epochs":               _train_cfg.get("epochs"),
        "max_edges_per_batch":  _train_cfg.get("max_edges_per_batch"),
        "lr":                   _train_cfg.get("lr"),
        "weight_decay":         _train_cfg.get("weight_decay"),
        "pearson_weight":       _train_cfg.get("pearson_weight"),
        "grad_accum_steps":     _train_cfg.get("grad_accum_steps"),
        "clip_grad":            _train_cfg.get("clip_grad"),
        "lr_patience":          _train_cfg.get("lr_patience"),
    }.items() if v is not None}
    parser.set_defaults(**_config_defaults)

    args = parser.parse_args()

    # ── DDP initialisation ────────────────────────────────────────────────────
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        local_rank = 0
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Device: {device}" + (f"  (DDP world_size={world_size})" if ddp else ""))

    # ── Config ────────────────────────────────────────────────────────────────
    cfg       = get_config()
    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(cfg["paths"]["log_file"]))

    base_name = f"{args.model}_{args.suffix}" if args.suffix else args.model
    ckpt_dir  = args.checkpoint_dir or (
        Path(data_root).parent / "checkpoints" / base_name
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset split (loaded from manifest written by 06_build_graphs.py) ─────
    train_ids, val_ids, test_ids = load_split_manifest(data_root)
    train_ds = ProteinGraphDataset(train_ids, data_root, rebuild=args.rebuild_graphs)
    val_ds   = ProteinGraphDataset(val_ids,   data_root, rebuild=False)
    test_ds  = ProteinGraphDataset(test_ids,  data_root, rebuild=False)
    if rank == 0:
        print(
            f"Split (from manifest) — train: {len(train_ds)}  "
            f"val: {len(val_ds)}  test: {len(test_ds)}"
        )

    # ── ESP normalisation (fit on training split only) ────────────────────────
    if rank == 0:
        print("Computing ESP normalisation statistics from training split...")
    esp_mean, esp_std = compute_esp_stats(train_ds)
    if rank == 0:
        print(f"  mean={esp_mean:.4f}  std={esp_std:.4f}")

    norm = NormalizeESP(esp_mean, esp_std)
    train_ds.transform = norm
    val_ds.transform   = norm
    test_ds.transform  = norm

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if rank == 0:
        print("Building dynamic batch samplers (reads edge counts from metadata)...")
    train_sampler = DynamicBatchSampler(
        train_ds, args.max_edges_per_batch,
        shuffle=True, drop_last=True,
        rank=rank, world_size=world_size,
    )
    val_sampler = DynamicBatchSampler(
        val_ds, args.max_edges_per_batch,
        shuffle=False, drop_last=False,
        rank=rank, world_size=world_size,
    )
    if rank == 0:
        print(
            f"  ~{len(train_sampler)} train batches/rank  |  "
            f"~{len(val_sampler)} val batches/rank  "
            f"(budget: {args.max_edges_per_batch:,} edges/batch)"
        )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    feat_cfg  = cfg.get("features", {})
    model_cfg = cfg.get("model",    {})
    model = build_model(args, device, feat_cfg, model_cfg)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.model}  |  parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience,
        )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume is not None:
        if rank == 0:
            print(f"Resuming from: {args.resume}")
        raw_model = model.module if ddp else model
        Trainer.load_checkpoint(args.resume, raw_model, optimizer, scheduler)

    # ── Trainer ───────────────────────────────────────────────────────────────
    loss_fn = ESPLoss(
        pearson_weight   = args.pearson_weight,
        protein_weighted = args.protein_weighted,
    )
    trainer = Trainer(
        model             = model,
        optimizer         = optimizer,
        scheduler         = scheduler,
        loss_fn           = loss_fn,
        device            = device,
        checkpoint_dir    = ckpt_dir,
        clip_grad_norm    = args.clip_grad,
        grad_accum_steps  = args.grad_accum_steps,
        rank              = rank,
        extra_state    = {
            "model_name":   args.model,
            "esp_mean":     esp_mean,
            "esp_std":      esp_std,
            "model_config": {
                "hidden_dim":           args.hidden_dim,
                "n_rbf":                args.n_rbf,
                "n_heads":              args.n_heads,
                "n_bond_radial_rounds": args.n_bond_radial_rounds,
                "n_aq_rounds":          args.n_aq_rounds,
                "n_qq_rounds":          args.n_qq_rounds,
                "multi_agg":            model_cfg.get("multi_agg", False),
            },
            "feature_spec": feat_cfg,
        },
    )

    if rank == 0:
        log.info(
            "Training %s on %d proteins for %d epochs",
            args.model, len(train_ids) + len(val_ids) + len(test_ids), args.epochs,
        )

    t_train_start = time.perf_counter()
    trainer.fit(train_loader, val_loader, n_epochs=args.epochs)
    train_wall_seconds = time.perf_counter() - t_train_start

    # ── Test evaluation (rank 0 only) ─────────────────────────────────────────
    if not ddp or rank == 0:
        if len(test_ds) == 0:
            print("\nNo test proteins — skipping test evaluation.")
        else:
            print(f"\nEvaluating on {len(test_ds)} test proteins...")
            raw_model = model.module if ddp else model
            Trainer.load_checkpoint(ckpt_dir / "best_model.pt", raw_model)

            pred_dir = ckpt_dir / "test_predictions"
            results  = run_evaluate_test(
                raw_model, loss_fn, test_ds, device, trainer.extra_state,
                checkpoint_dir  = ckpt_dir,
                predictions_dir = pred_dir,
            )

            # Inject wall-clock training time into the global metrics and re-save.
            results["global"]["train_wall_time_s"] = round(train_wall_seconds, 1)
            metrics_path = ckpt_dir / "test_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(results, f, indent=2)

            g = results["global"]
            h, m = divmod(int(train_wall_seconds), 3600)
            m, s = divmod(m, 60)
            print(
                f"\nTest results  (best checkpoint, {g['n_proteins']} proteins)\n"
                f"  Loss:      {g['loss']:.4f}\n"
                f"  RMSE:      {g['rmse']:.4f} kT/e\n"
                f"  MAE:       {g['mae']:.4f} kT/e\n"
                f"  Pearson r: {g['pearson_r']:.4f}\n"
                f"  Train time: {h:02d}:{m:02d}:{s:02d}\n"
            )

            pp     = results["per_protein"]
            ranked = sorted(pp.items(), key=lambda kv: kv[1]["pearson_r"])
            print(f"{'Protein':<30}  {'Pearson r':>10}  {'RMSE':>10}  {'MAE':>10}")
            print("-" * 66)
            for pid, m in ranked:
                print(
                    f"{pid:<30}  {m['pearson_r']:>10.4f}  "
                    f"{m['rmse']:>10.4f}  {m['mae']:>10.4f}"
                )
            print(f"\nPredictions saved to: {pred_dir}")
            print(f"Per-protein metrics:  {ckpt_dir / 'test_metrics.json'}")
            print(f"Training history:     {ckpt_dir / 'metrics.csv'}")

            log.info(
                "Test complete — loss=%.4f  rmse=%.4f  pearson_r=%.4f",
                g["loss"], g["rmse"], g["pearson_r"],
            )

    # ── DDP teardown ──────────────────────────────────────────────────────────
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
