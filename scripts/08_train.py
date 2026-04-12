"""
scripts/08_train.py

Train DistanceESPN or AttentionESPN on ESP surface graphs.

Usage
-----
    # Train on all proteins, distance model
    python scripts/08_train.py --model distance --all

    # Train on filtered proteins, attention model, resume from checkpoint
    python scripts/08_train.py --model attention --filter --min-plddt 70 \\
        --epochs 150 --batch-size 4 --resume checkpoints/run_01/latest_model.pt

    # Override architecture defaults
    python scripts/08_train.py --model attention --all \\
        --hidden-dim 256 --n-heads 8 --n-aq-rounds 4

Checkpoints are saved to <checkpoint-dir>/<model>_<variant>/ and contain
model weights, optimizer/scheduler state, and ESP normalization statistics
(esp_mean, esp_std) needed for inference.
"""

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import ProteinGraphDataset, split_dataset
from src.data.sampler import DynamicBatchSampler
from src.data.transform import NormalizeESP, compute_esp_stats
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.training.loss import ESPLoss
from src.training.trainer import Trainer
from src.utils.config import get_config, get_data_root
from src.utils.filter import add_filter_args, get_protein_ids_from_args
from src.utils.helpers import get_pipeline_logger


def build_model(args, device: torch.device):
    common = dict(
        hidden_dim           = args.hidden_dim,
        n_rbf                = args.n_rbf,
        n_bond_radial_rounds = args.n_bond_radial_rounds,
        n_aq_rounds          = args.n_aq_rounds,
        n_qq_rounds          = args.n_qq_rounds,
    )
    if args.model == "distance":
        model = DistanceESPN(**common)
    else:
        model = AttentionESPN(**common, n_heads=args.n_heads)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DistanceESPN or AttentionESPN on protein ESP graphs."
    )

    # ── Protein selection ─────────────────────────────────────────────────────
    add_filter_args(parser)
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data_root from config.yaml.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", choices=["distance", "attention"], required=True,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--variant", choices=["interp", "laplacian"], default="interp",
        help="ESP target variant (default: interp).",
    )

    # ── Architecture ──────────────────────────────────────────────────────────
    parser.add_argument("--hidden-dim",        type=int,   default=128)
    parser.add_argument("--n-rbf",             type=int,   default=16)
    parser.add_argument("--n-heads",           type=int,   default=4,
                        help="Attention heads (attention model only).")
    parser.add_argument("--n-bond-radial-rounds", type=int,   default=2)
    parser.add_argument("--n-aq-rounds",       type=int,   default=3)
    parser.add_argument("--n-qq-rounds",       type=int,   default=2)

    # ── Graph construction ────────────────────────────────────────────────────
    parser.add_argument("--sample-frac", type=float, default=0.05,
                        help="Fraction of mesh vertices used as query nodes.")
    parser.add_argument("--rebuild-graphs", action="store_true",
                        help="Ignore cached graphs and rebuild from scratch.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--max-edges-per-batch", type=int,  default=200_000,
                        help="Edge budget per batch for DynamicBatchSampler.")
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--pearson-weight", type=float, default=0.1,
                        help="Weight for the Pearson correlation loss term.")
    parser.add_argument("--clip-grad",      type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable).")
    parser.add_argument("--lr-patience",    type=int,   default=15,
                        help="ReduceLROnPlateau patience in epochs.")
    parser.add_argument("--train-frac",     type=float, default=0.8)
    parser.add_argument("--val-frac",       type=float, default=0.1)
    parser.add_argument("--split-seed",     type=int,   default=42)

    # ── I/O ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory for checkpoints. Defaults to <data_root>/../checkpoints.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (0 = main process).")

    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg       = get_config()
    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(cfg["paths"]["log_file"]))

    ckpt_dir = args.checkpoint_dir or (
        Path(data_root).parent / "checkpoints" / f"{args.model}_{args.variant}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Protein IDs + dataset split ───────────────────────────────────────────
    protein_ids = get_protein_ids_from_args(args, data_root)
    if not protein_ids:
        print("No proteins selected. Exiting.")
        return
    print(f"Proteins selected: {len(protein_ids)}")

    graph_kwargs = dict(
        variant     = args.variant,
        sample_frac = args.sample_frac,
        rebuild     = args.rebuild_graphs,
    )
    full_ds = ProteinGraphDataset(protein_ids, data_root, **graph_kwargs)
    train_ds, val_ds, test_ds = split_dataset(
        full_ds,
        train     = args.train_frac,
        val       = args.val_frac,
        seed      = args.split_seed,
    )
    print(
        f"Split — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}"
    )

    # ── ESP normalisation (fit on training split only) ────────────────────────
    print("Computing ESP normalisation statistics from training split...")
    esp_mean, esp_std = compute_esp_stats(train_ds)
    print(f"  mean={esp_mean:.4f}  std={esp_std:.4f}")

    norm = NormalizeESP(esp_mean, esp_std)
    train_ds.transform = norm
    val_ds.transform   = norm
    test_ds.transform  = norm

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("Building dynamic batch samplers (reads edge counts from metadata)...")
    train_sampler = DynamicBatchSampler(
        train_ds, args.max_edges_per_batch,
        shuffle=True, drop_last=True,
    )
    val_sampler = DynamicBatchSampler(
        val_ds, args.max_edges_per_batch,
        shuffle=False, drop_last=False,
    )
    print(
        f"  ~{len(train_sampler)} train batches  |  "
        f"~{len(val_sampler)} val batches  "
        f"(budget: {args.max_edges_per_batch:,} edges/batch)"
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        Trainer.load_checkpoint(args.resume, model, optimizer, scheduler)

    # ── Trainer ───────────────────────────────────────────────────────────────
    loss_fn = ESPLoss(pearson_weight=args.pearson_weight)
    trainer = Trainer(
        model          = model,
        optimizer      = optimizer,
        scheduler      = scheduler,
        loss_fn        = loss_fn,
        device         = device,
        checkpoint_dir = ckpt_dir,
        clip_grad_norm = args.clip_grad,
        extra_state    = {
            "model_name":   args.model,
            "variant":      args.variant,
            "esp_mean":     esp_mean,
            "esp_std":      esp_std,
            "model_config": {
                "hidden_dim":           args.hidden_dim,
                "n_rbf":                args.n_rbf,
                "n_heads":              args.n_heads,
                "n_bond_radial_rounds": args.n_bond_radial_rounds,
                "n_aq_rounds":          args.n_aq_rounds,
                "n_qq_rounds":          args.n_qq_rounds,
            },
        },
    )

    log.info(
        "Training %s (variant=%s) on %d proteins for %d epochs",
        args.model, args.variant, len(protein_ids), args.epochs,
    )

    trainer.fit(train_loader, val_loader, n_epochs=args.epochs)

    # ── Test evaluation ───────────────────────────────────────────────────────
    if len(test_ds) == 0:
        print("\nNo test proteins — skipping test evaluation.")
    else:
        print(f"\nEvaluating on {len(test_ds)} test proteins...")
        Trainer.load_checkpoint(ckpt_dir / "best_model.pt", model)

        pred_dir = ckpt_dir / "test_predictions"
        results  = trainer.evaluate_test(test_ds, predictions_dir=pred_dir)

        g = results["global"]
        print(
            f"\nTest results  (best checkpoint, {g['n_proteins']} proteins)\n"
            f"  Loss:      {g['loss']:.4f}\n"
            f"  RMSE:      {g['rmse']:.4f} kT/e\n"
            f"  MAE:       {g['mae']:.4f} kT/e\n"
            f"  Pearson r: {g['pearson_r']:.4f}\n"
        )

        # Per-protein summary: sort by Pearson r ascending so worst are first
        pp = results["per_protein"]
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


if __name__ == "__main__":
    main()
