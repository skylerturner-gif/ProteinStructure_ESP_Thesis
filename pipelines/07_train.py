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
from datetime import timedelta
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import ProteinGraphDataset, load_split_manifest
from src.data.sampler import DynamicBatchSampler
from src.data.transform import NormalizeESP, load_or_compute_esp_stats
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.training.loss import ESPLoss
from src.training.trainer import Trainer, evaluate_test as run_evaluate_test
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger


def build_model(args, device: torch.device, feat_cfg: dict):
    common = dict(
        hidden_dim           = args.hidden_dim,
        n_rbf                = args.n_rbf,
        n_bond_radial_rounds = args.n_bond_radial_rounds,
        n_aq_rounds          = args.n_aq_rounds,
        n_qq_rounds          = args.n_qq_rounds,
        agg                  = args.agg,
        use_element_embedding = args.use_element_embedding,
        use_residue_embedding = args.use_residue_embedding,
        use_bond_edges        = args.use_bond_edges,
        use_radial_edges      = args.use_radial_edges,
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
    parser.add_argument("--agg", choices=["mean", "sum", "max", "multi"], default="mean",
                        help="MessageLayer aggregation. 'multi' concatenates "
                             "mean+sum+max. Attention model applies this to "
                             "bond/radial/qq only — its AQ stage is always "
                             "cross-attention.")

    # ── Feature ablation ──────────────────────────────────────────────────────
    parser.add_argument("--no-element-embedding", dest="use_element_embedding",
                        action="store_false", default=True,
                        help="Replace per-element-type embedding with a single shared "
                             "learned constant — all atoms become chemically identical. "
                             "atom_type is not read from the graph.")
    parser.add_argument("--no-residue-embedding", dest="use_residue_embedding",
                        action="store_false", default=True,
                        help="Ablate the residue-type embedding in AtomEncoder.")
    parser.add_argument("--no-bond-edges", dest="use_bond_edges",
                        action="store_false", default=True,
                        help="Ablate the bond-edge message pass in Stage 1, and the "
                             "bond-count projection in AtomEncoder (gated together — "
                             "otherwise bond chemistry leaks back in via h_src/h_dst "
                             "on radial edges). Leaves atom type as the only "
                             "chemistry signal.")
    parser.add_argument("--no-radial-edges", dest="use_radial_edges",
                        action="store_false", default=True,
                        help="Ablate the radial-edge message pass in Stage 1.")

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
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training if val loss does not improve for this many epochs "
                             "(0 = disabled).")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bfloat16 mixed precision (forward pass only; "
                             "optimizer stays FP32). Requires A100 or newer.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for weight initialisation reproducibility.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay factor for the always-on weight shadow used by "
                             "val/best_model.pt (default 0.999).")

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
        "agg":                  _model_cfg.get("agg"),
        "use_element_embedding": _model_cfg.get("use_element_embedding"),
        "use_residue_embedding": _model_cfg.get("use_residue_embedding"),
        "use_bond_edges":        _model_cfg.get("use_bond_edges"),
        "use_radial_edges":      _model_cfg.get("use_radial_edges"),
        "epochs":               _train_cfg.get("epochs"),
        "max_edges_per_batch":  _train_cfg.get("max_edges_per_batch"),
        "lr":                   _train_cfg.get("lr"),
        "weight_decay":         _train_cfg.get("weight_decay"),
        "pearson_weight":       _train_cfg.get("pearson_weight"),
        "grad_accum_steps":          _train_cfg.get("grad_accum_steps"),
        "clip_grad":                 _train_cfg.get("clip_grad"),
        "lr_patience":               _train_cfg.get("lr_patience"),
        "early_stopping_patience":   _train_cfg.get("early_stopping_patience"),
        "seed":                      _train_cfg.get("seed"),
    }.items() if v is not None}
    parser.set_defaults(**_config_defaults)

    args = parser.parse_args()

    # ── DDP initialisation ────────────────────────────────────────────────────
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        # device_id omitted: torch.cuda.set_device() above is sufficient and
        # device_id triggers eager NCCL init which can deadlock if another CUDA
        # process holds a context on the same GPU.
        #
        # timeout raised from PyTorch's 600s default: rank 0 does the (rank-0-only,
        # possibly first-ever-run) ESP-stats scan before any collective call, so
        # NCCL communicator setup — lazily triggered by the first collective, i.e.
        # our load_or_compute_esp_stats() barrier below — could in principle
        # outlast 10min on a large, not-yet-cached dataset while the other ranks
        # sit waiting. In practice the scan now reads esp/*.npz (not the cached
        # graph/*.pt) and measured at ~35s for full_protein_dataset's 6768-protein
        # training split — well under the default — but left generously high
        # since this whole wait only ever happens once per dataset (esp_stats.json
        # is cached after) and there's no cost to erring huge here.
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=12))
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

    # ── ESP normalisation (fit on training split only, cached to data_root) ───
    # Rank 0 computes (or loads the cache) first; other ranks wait behind a
    # barrier and then just read the now-guaranteed-present cache file, so
    # the expensive first-time scan of the training split never happens more
    # than once total, not once per rank.
    #
    # Full scan of all 6768 training proteins, by explicit choice: this is a
    # one-time cost (cached to esp_stats.json for every future run against
    # this dataset), and correctness on the exact training split beat a
    # sampled approximation for this dataset specifically. Reads esp/*.npz
    # (query_esp = esp_verts[query_idx], the same values graph_builder copies
    # verbatim into query.y) rather than the cached graph/*.pt — ~13x less
    # data per protein since it skips deserialising every atom/bond/radial/
    # aq/qq tensor to get one small array. Measured on this machine at
    # ~5ms/file (vs ~1.27s/file reading the full graph) — ~35s total for this
    # one pass, down from ~2-2.5 hours. Sequential (stats_workers=1): an
    # earlier 8-way parallel attempt over the (much larger) graph files was
    # slower, not faster, on this disk (concurrent reads to scattered files
    # caused seek thrashing rather than adding throughput) — moot now that a
    # full sequential pass finishes in well under a minute.
    stats_workers = 1
    if rank == 0:
        cache_hit = (Path(data_root) / "esp_stats.json").exists()
        print(
            "Loading cached ESP normalisation statistics..." if cache_hit else
            f"Computing ESP normalisation statistics from the full {len(train_ds)}-protein "
            f"training split ({stats_workers} worker, cached for future runs)..."
        )
        esp_mean, esp_std = load_or_compute_esp_stats(train_ids, data_root, n_workers=stats_workers)
        print(f"  mean={esp_mean:.4f}  std={esp_std:.4f}")
    if ddp:
        torch.distributed.barrier()
    if rank != 0:
        esp_mean, esp_std = load_or_compute_esp_stats(train_ids, data_root, n_workers=1)

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

    # persistent_workers=True + pin_memory=True was tried as a speed optimization but
    # reliably corrupted data in transit from DataLoader workers: atom_type went out
    # of AtomEncoder's embedding range (negative, past clamp's upper-only bound),
    # crashing every real run (DDP=2 + num_workers=8) with a device-side assert in
    # self.atom_emb(atom_type) on batch 1 — reproduced identically on both models.
    # Confirmed fixed by turning both off (full run completed clean, num_workers=2);
    # left off rather than re-enabled without a validated root cause for the corruption.
    loader_kwargs = dict(
        num_workers         = args.num_workers,
        persistent_workers  = False,
        pin_memory          = False,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   **loader_kwargs)

    # ── Seed (before model init so weight initialisation is reproducible) ─────
    import random as _random
    import numpy as np
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if rank == 0:
        print(f"RNG seed: {args.seed}")

    # ── Model ─────────────────────────────────────────────────────────────────
    feat_cfg  = cfg.get("features", {})
    model = build_model(args, device, feat_cfg)
    if ddp:
        # PyTorch 2.11 bug: _verify_params_across_processes inside DDP.__init__
        # fails with "rank 0 has 0 params" even when both ranks have identical
        # models (confirmed with a minimal reproducer). Work-around: skip DDP's
        # built-in sync (init_sync=False) and broadcast parameters from rank 0
        # manually — this is exactly what DDP would have done internally.
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for buf in model.buffers():
            torch.distributed.broadcast(buf, src=0)
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[local_rank], init_sync=False,
                                        find_unused_parameters=True)
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
        pearson_weight = args.pearson_weight,
    )
    trainer = Trainer(
        model             = model,
        optimizer         = optimizer,
        scheduler         = scheduler,
        loss_fn           = loss_fn,
        device            = device,
        checkpoint_dir    = ckpt_dir,
        clip_grad_norm           = args.clip_grad,
        grad_accum_steps         = args.grad_accum_steps,
        early_stopping_patience  = args.early_stopping_patience,
        rank                     = rank,
        bf16                     = args.bf16,
        ema_decay                = args.ema_decay,
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
                "agg":                  args.agg,
                "use_element_embedding": args.use_element_embedding,
                "use_residue_embedding": args.use_residue_embedding,
                "use_bond_edges":        args.use_bond_edges,
                "use_radial_edges":      args.use_radial_edges,
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
                bf16            = args.bf16,
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
