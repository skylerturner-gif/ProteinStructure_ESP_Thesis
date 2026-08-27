"""Evaluate trained Surface DGCNN checkpoint on the canonical test split.

Outputs:
    <out_dir>/surface_dgcnn_results.csv   — per-protein metrics
    <out_dir>/surface_dgcnn_summary.txt   — aggregate statistics

Usage:
    conda run -n pyg_env python -m src.baseline_models.surface_dgcnn.evaluate \\
        [--checkpoint model_eval/surface_dgcnn/best_model.pt] \\
        [--config model_eval/surface_dgcnn/config.json] \\
        [--split test] \\
        [--out-dir model_eval/surface_dgcnn]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.baseline_models.surface_dgcnn.dataset import SurfaceDataset, collate_surface
from src.baseline_models.surface_dgcnn.model import SurfaceDGCNN
from src.data.dataset import load_split_manifest
from src.training.loss import pearson_r as pearson_r_torch
from src.utils.config import get_data_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint. Defaults to <data_root>/../checkpoints/baseline/surface_dgcnn_chem/best_model.pt")
    parser.add_argument("--config", default=None,
                        help="Path to config.json. Defaults to same dir as checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for results. Defaults to <data_root>/../model_eval/baseline/surface_dgcnn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(get_data_root())

    ckpt_dir  = data_root.parent / "checkpoints" / "baseline" / "surface_dgcnn_chem"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "best_model.pt"
    cfg_path  = Path(args.config) if args.config else ckpt_dir / "config.json"
    out_dir   = Path(args.out_dir) if args.out_dir else data_root.parent / "model_eval" / "baseline" / "surface_dgcnn_chem"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    train_ids, val_ids, test_ids = load_split_manifest(data_root)
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    ids = split_map[args.split]

    max_pts = cfg.get("max_points", 8192) or None
    ds = SurfaceDataset(ids, data_root, max_points=max_pts)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_surface, pin_memory=False)

    model = SurfaceDGCNN(
        in_features=cfg.get("in_features", ds.in_features),
        hidden_dim=cfg.get("hidden_dim", 128),
        n_query_k=cfg.get("n_query_k", 8),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            pid = batch["protein_id"][0]
            pc = batch["point_cloud"][0].to(device)
            ei = batch["edge_index"][0].to(device)
            qsurf = batch["query_surf"][0].to(device)
            target = batch["query_esp"][0].to(device)

            pred = model(pc, ei, qsurf)
            r    = pearson_r_torch(pred, target).item()
            rmse = math.sqrt(((pred - target) ** 2).mean().item())
            mae  = (pred - target).abs().mean().item()

            rows.append({
                "protein_id":   pid,
                "split":        args.split,
                "pearson_r":    round(r, 6),
                "rmse":         round(rmse, 6),
                "mae":          round(mae, 6),
                "n_query_nodes": len(target),
            })

    csv_path = out_dir / "surface_dgcnn_results.csv"
    fieldnames = ["protein_id", "split", "pearson_r", "rmse", "mae", "n_query_nodes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {csv_path}")

    rs    = [r["pearson_r"] for r in rows]
    rmses = [r["rmse"]      for r in rows]
    maes  = [r["mae"]       for r in rows]
    summary_path = out_dir / "surface_dgcnn_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"=== {args.split.upper()} ({len(rows)} proteins) ===\n")
        f.write(f"  Pearson r   mean={statistics.mean(rs):.4f}  "
                f"median={statistics.median(rs):.4f}  "
                f"std={statistics.stdev(rs):.4f}\n")
        f.write(f"  RMSE        mean={statistics.mean(rmses):.4f}  "
                f"median={statistics.median(rmses):.4f}  "
                f"std={statistics.stdev(rmses):.4f}\n")
        f.write(f"  MAE         mean={statistics.mean(maes):.4f}  "
                f"median={statistics.median(maes):.4f}  "
                f"std={statistics.stdev(maes):.4f}\n")
    print(f"Summary → {summary_path}")

    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "global": {
                "pearson_r":  statistics.mean(rs),
                "rmse":       statistics.mean(rmses),
                "mae":        statistics.mean(maes),
                "n_proteins": len(rows),
            },
            "per_protein": {
                r["protein_id"]: {
                    "pearson_r":    r["pearson_r"],
                    "rmse":         r["rmse"],
                    "mae":          r["mae"],
                    "n_query_nodes": r["n_query_nodes"],
                }
                for r in rows
            },
        }, f, indent=2)
    print(f"test_metrics.json → {metrics_path}")


if __name__ == "__main__":
    main()
