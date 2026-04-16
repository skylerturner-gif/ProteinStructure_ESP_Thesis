"""
scripts/analyze_embeddings.py

Post-training embedding analysis: atom-type embedding similarity heatmap and
(for attention models) per-element attention weight statistics.  Optionally
compares two models' embedding tables side by side.

Usage:
    # Single model — embedding heatmap + attention stats on test split
    python scripts/analyze_embeddings.py --model attention --suffix base

    # Cross-model comparison (embedding tables of both models)
    python scripts/analyze_embeddings.py \\
        --model attention --suffix base \\
        --compare-model distance --compare-suffix base

    # Explicit checkpoint paths
    python scripts/analyze_embeddings.py \\
        --checkpoint-dir /path/to/checkpoints/attention_norm_curv \\
        --compare-checkpoint-dir /path/to/checkpoints/distance_norm_curv

Outputs written to the resolved checkpoint dir (or --save-dir):
    embedding_analysis.json      — cosine sim matrix + attention stats (JSON)
    element_cosine_sim.png       — 7×7 element similarity heatmap
    attention_by_element.png     — mean α per element per head (attention only)
    cross_model_embedding_sim.png— per-element cross-model comparison (if --compare-* given)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.embedding_analysis import (
    ELEMENT_NAMES,
    collect_attention_stats,
    compare_embedding_tables,
    embedding_cosine_sim,
    load_model_frozen,
)
from src.data.dataset import load_split_manifest
from src.models.attention_espn import AttentionESPN
from src.utils.config import get_data_root


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plot_cosine_heatmap(
    sim: np.ndarray, labels: list[str], title: str, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_attention_by_element(
    stats: dict, title: str, save_path: Path
) -> None:
    present = [(name, stats[name]) for name in ELEMENT_NAMES if stats[name]["count"] > 0]
    if not present:
        print("  No attention stats to plot.")
        return

    names   = [p[0] for p in present]
    means   = np.array([p[1]["mean"] for p in present])   # (n_elem, n_heads)
    n_heads = means.shape[1]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    x = np.arange(len(names))
    w = 0.7 / n_heads
    for h in range(n_heads):
        ax.bar(x + h * w - 0.35 + w / 2, means[:, h], width=w, label=f"Head {h}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean attention weight α")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_cross_model_sim(
    sims: np.ndarray, title: str, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    x    = np.arange(len(ELEMENT_NAMES))
    bars = ax.bar(x, sims, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(ELEMENT_NAMES)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    for bar, val in zip(bars, sims):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atom embedding and attention weight analysis for a trained ESP model."
    )
    parser.add_argument("--model", choices=["distance", "attention"], default=None,
                        help="Model type — used to infer checkpoint dir.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for checkpoint dir, e.g. 'base' → checkpoints/attention_base.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Explicit checkpoint directory.")

    parser.add_argument("--compare-model", choices=["distance", "attention"], default=None,
                        help="Second model for cross-model embedding comparison.")
    parser.add_argument("--compare-suffix", type=str, default=None)
    parser.add_argument("--compare-checkpoint-dir", type=Path, default=None)

    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split used for attention stats collection (default: test).")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Override output directory for plots and JSON.")

    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or get_data_root()

    # ── Resolve primary checkpoint dir ────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        if args.model is None:
            parser.error("Provide --model or --checkpoint-dir.")
        base     = f"{args.model}_{args.suffix}" if args.suffix else args.model
        ckpt_dir = Path(data_root).parent / "checkpoints" / base

    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return

    out_dir = args.save_dir or ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt_dir} ...")
    model, ckpt = load_model_frozen(ckpt_dir, device)

    # ── Embedding cosine similarity heatmap ───────────────────────────────────
    print("Computing atom embedding cosine similarity...")
    sim, labels = embedding_cosine_sim(model)
    _plot_cosine_heatmap(
        sim, labels,
        title=f"Atom embedding cosine similarity ({ckpt_dir.name})",
        save_path=out_dir / "element_cosine_sim.png",
    )
    results: dict = {"embedding_cosine_sim": sim.tolist(), "element_names": labels}

    # ── Per-element attention stats (attention model only) ────────────────────
    if isinstance(model, AttentionESPN):
        print(f"Collecting attention stats on '{args.split}' split...")
        train_ids, val_ids, test_ids = load_split_manifest(data_root)
        split_map   = {"train": train_ids, "val": val_ids, "test": test_ids}
        protein_ids = split_map[args.split]

        attn_stats = collect_attention_stats(model, protein_ids, data_root, device)
        results["attention_stats"] = attn_stats

        _plot_attention_by_element(
            attn_stats,
            title=f"Mean attention α by element ({ckpt_dir.name}, {args.split})",
            save_path=out_dir / "attention_by_element.png",
        )
    else:
        print("Skipping attention stats (distance model has no cross-attention).")

    # ── Cross-model embedding comparison ─────────────────────────────────────
    cmp_dir = args.compare_checkpoint_dir
    if cmp_dir is None and args.compare_model:
        base    = f"{args.compare_model}_{args.compare_suffix}" if args.compare_suffix else args.compare_model
        cmp_dir = Path(data_root).parent / "checkpoints" / base

    if cmp_dir is not None:
        if not cmp_dir.exists():
            print(f"Compare checkpoint directory not found: {cmp_dir}")
        else:
            print(f"Loading compare model from {cmp_dir} ...")
            model_b, _ = load_model_frozen(cmp_dir, device)
            sims        = compare_embedding_tables(model, model_b)
            results["cross_model_embedding_sim"] = {
                "model_a":    ckpt_dir.name,
                "model_b":    cmp_dir.name,
                "similarity": sims.tolist(),
            }
            _plot_cross_model_sim(
                sims,
                title=f"Embedding similarity: {ckpt_dir.name} vs {cmp_dir.name}",
                save_path=out_dir / "cross_model_embedding_sim.png",
            )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "embedding_analysis.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {json_path.name}")


if __name__ == "__main__":
    main()
