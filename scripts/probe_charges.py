"""
scripts/probe_charges.py

Train a small frozen-backbone MLP to predict per-atom partial charges from
PQR files using atom embeddings extracted from a trained model.  Tests whether
the model's internal atom representations encode chemistry.

Backbone weights are never modified — only ChargeProbe.mlp is trained.

Usage:
    # Using model + suffix
    python scripts/probe_charges.py --model attention --suffix base

    # Probe the raw encoder output (before message passing)
    python scripts/probe_charges.py --model attention --suffix norm_curv \\
        --layer after_encoder --epochs 50

    # Explicit checkpoint dir
    python scripts/probe_charges.py --checkpoint-dir /path/to/checkpoints/attention_base

Outputs written to <checkpoint_dir>/charge_probe/ (or --save-dir):
    charge_probe_results.json — per-protein and global metrics
    charge_scatter.png        — predicted vs true partial charge scatter (test set)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.charge_probe import (
    ChargeProbe,
    evaluate_probe,
    extract_atom_embeddings,
    read_pqr_charges,
    train_probe,
)
from src.analysis.embedding_analysis import _load_graph, load_model_frozen
from src.data.dataset import load_split_manifest
from src.utils.config import get_data_root
from src.utils.paths import ProteinPaths


def _plot_charge_scatter(
    probe: ChargeProbe,
    model,
    protein_ids: list[str],
    data_root: Path,
    layer: str,
    device: torch.device,
    title: str,
    save_path: Path,
) -> None:
    all_true: list[float] = []
    all_pred: list[float] = []

    probe = probe.to(device)
    probe.eval()
    with torch.no_grad():
        for pid in protein_ids:
            paths = ProteinPaths(pid, data_root)
            if not paths.pqr_path.exists():
                continue
            charges = read_pqr_charges(paths.pqr_path)
            data    = _load_graph(pid, data_root)
            h_atom  = extract_atom_embeddings(model, data, layer=layer, device=device)
            if h_atom.shape[0] != len(charges):
                continue
            pred = probe(h_atom.to(device)).cpu().numpy()
            all_true.extend(charges.tolist())
            all_pred.extend(pred.tolist())

    if not all_true:
        print("  No valid proteins for scatter plot.")
        return

    t = np.array(all_true)
    p = np.array(all_pred)
    r = float(np.corrcoef(t, p)[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=2, alpha=0.3, color="steelblue", rasterized=True)
    lo = min(t.min(), p.min())
    hi = max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True partial charge (e)")
    ax.set_ylabel("Predicted partial charge (e)")
    ax.set_title(f"{title}\n(r = {r:.3f})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial-charge probe: predict PQR charges from frozen atom embeddings."
    )
    parser.add_argument("--model", choices=["distance", "attention"], default=None,
                        help="Model type — used to infer checkpoint dir.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for checkpoint dir, e.g. 'base' → checkpoints/attention_base.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Explicit checkpoint directory.")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml.")
    parser.add_argument(
        "--layer", choices=["after_encoder", "after_mp"], default="after_mp",
        help="Atom embedding layer to probe: 'after_encoder' (raw) or 'after_mp' "
             "(after Stage 1 message passing, default).",
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Probe training epochs (default: 30).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate for probe (default: 1e-3).")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Override output directory (default: <ckpt_dir>/charge_probe/).")

    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or get_data_root()

    # ── Resolve checkpoint dir ────────────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        if args.model is None:
            parser.error("Provide --model or --checkpoint-dir.")
        base     = f"{args.model}_{args.suffix}" if args.suffix else args.model
        ckpt_dir = Path(data_root).parent / "checkpoints" / base

    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return

    out_dir = args.save_dir or ckpt_dir / "charge_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frozen model ─────────────────────────────────────────────────────
    print(f"Loading model from {ckpt_dir} ...")
    model, ckpt = load_model_frozen(ckpt_dir, device)
    hidden_dim  = ckpt["model_config"]["hidden_dim"]

    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total  = sum(1 for p in model.parameters())
    print(f"  Backbone params: {n_total} total, {n_frozen} frozen (sanity: should be equal)")

    train_ids, val_ids, test_ids = load_split_manifest(data_root)

    # ── Train probe ───────────────────────────────────────────────────────────
    probe = ChargeProbe(hidden_dim)
    print(f"\nTraining charge probe (layer={args.layer}) on {len(train_ids)} train proteins "
          f"for {args.epochs} epochs...")
    probe = train_probe(
        probe, model, train_ids, data_root,
        layer=args.layer, device=device, epochs=args.epochs, lr=args.lr,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print(f"\nEvaluating on {len(test_ids)} test proteins...")
    results = evaluate_probe(probe, model, test_ids, data_root, layer=args.layer, device=device)

    g = results["global"]
    print(f"\nCharge probe results  (test set, layer={args.layer})")
    print(f"  RMSE:       {g['rmse']:.4f} e")
    print(f"  MAE:        {g['mae']:.4f} e")
    print(f"  Mean R²:    {g['mean_r2']:.4f}")
    print(f"  Proteins:   {g['n_proteins']}   Atoms: {g['n_atoms']}")

    results["config"] = {
        "ckpt_dir": str(ckpt_dir),
        "layer":    args.layer,
        "epochs":   args.epochs,
        "lr":       args.lr,
    }
    json_path = out_dir / "charge_probe_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {json_path}")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    _plot_charge_scatter(
        probe, model, test_ids, data_root, args.layer, device,
        title=f"Charge probe — {ckpt_dir.name} ({args.layer})",
        save_path=out_dir / "charge_scatter.png",
    )


if __name__ == "__main__":
    main()
