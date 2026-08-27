"""Evaluate Coulomb baseline on the canonical train/val/test split.

Outputs:
    <out_dir>/coulomb_results.csv   — per-protein metrics
    <out_dir>/coulomb_summary.txt   — aggregate statistics

Usage:
    python -m src.baseline_models.coulomb.evaluate \\
        [--out-dir model_eval/coulomb] \\
        [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.baseline_models.coulomb.predictor import CoulombESP, compute_metrics
from src.data.dataset import load_split_manifest
from src.utils.config import get_data_root
from src.utils.paths import ProteinPaths


def _worker(args: tuple[str, Path, str]) -> dict | None:
    pid, data_root, split = args
    p = ProteinPaths(pid, data_root)
    if not p.pqr_path.exists() or not p.esp_path.exists():
        return None
    try:
        esp_data = np.load(p.esp_path)
        query_idx = esp_data["query_idx"]
        query_pos = esp_data["verts"][query_idx]
        true_esp = esp_data["esp_verts"][query_idx]

        predictor = CoulombESP(p.pqr_path)
        r, rmse, mae = predictor.evaluate(query_pos, true_esp)
        return {
            "protein_id": pid,
            "split": split,
            "pearson_r": round(r, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "n_query_nodes": len(query_idx),
        }
    except Exception as e:
        print(f"[coulomb] SKIP {pid}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Coulomb baseline on split manifest")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"],
                        help="Which split(s) to evaluate (default: test)")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir. Defaults to <data_root>/../model_eval/baseline/coulomb")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes")
    args = parser.parse_args()

    data_root = Path(get_data_root())
    out_dir = (Path(args.out_dir) if args.out_dir
               else data_root.parent / "model_eval" / "baseline" / "coulomb")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = load_split_manifest(data_root)
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}

    if args.split == "all":
        all_items: list[tuple[str, Path, str]] = (
            [(pid, data_root, "train") for pid in train_ids]
            + [(pid, data_root, "val")   for pid in val_ids]
            + [(pid, data_root, "test")  for pid in test_ids]
        )
    else:
        all_items = [(pid, data_root, args.split) for pid in split_map[args.split]]

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in all_items}
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda r: (r["split"], r["protein_id"]))

    csv_path = out_dir / "coulomb_results.csv"
    fieldnames = ["protein_id", "split", "pearson_r", "rmse", "mae", "n_query_nodes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {csv_path}")

    _write_summary(rows, out_dir / "coulomb_summary.txt")
    _write_test_metrics(rows, out_dir / "test_metrics.json")


def _write_test_metrics(rows: list[dict], path: Path) -> None:
    import statistics
    test_rows = [r for r in rows if r["split"] == "test"]
    global_metrics = {
        "pearson_r": statistics.mean(r["pearson_r"] for r in test_rows),
        "rmse":      statistics.mean(r["rmse"]      for r in test_rows),
        "mae":       statistics.mean(r["mae"]        for r in test_rows),
        "n_proteins": len(test_rows),
    }
    per_protein = {
        r["protein_id"]: {
            "pearson_r":    r["pearson_r"],
            "rmse":         r["rmse"],
            "mae":          r["mae"],
            "n_query_nodes": r["n_query_nodes"],
        }
        for r in test_rows
    }
    with open(path, "w") as f:
        json.dump({"global": global_metrics, "per_protein": per_protein}, f, indent=2)
    print(f"test_metrics.json → {path}")


def _write_summary(rows: list[dict], path: Path) -> None:
    import statistics

    splits = ["train", "val", "test", "all"]
    with open(path, "w") as f:
        for split in splits:
            subset = rows if split == "all" else [r for r in rows if r["split"] == split]
            if not subset:
                continue
            rs    = [r["pearson_r"] for r in subset]
            rmses = [r["rmse"]      for r in subset]
            maes  = [r["mae"]       for r in subset]
            f.write(f"=== {split.upper()} ({len(subset)} proteins) ===\n")
            f.write(f"  Pearson r   mean={statistics.mean(rs):.4f}  "
                    f"median={statistics.median(rs):.4f}  "
                    f"std={statistics.stdev(rs):.4f}\n")
            f.write(f"  RMSE        mean={statistics.mean(rmses):.4f}  "
                    f"median={statistics.median(rmses):.4f}  "
                    f"std={statistics.stdev(rmses):.4f}\n")
            f.write(f"  MAE         mean={statistics.mean(maes):.4f}  "
                    f"median={statistics.median(maes):.4f}  "
                    f"std={statistics.stdev(maes):.4f}\n\n")

    print(f"Summary → {path}")


if __name__ == "__main__":
    main()
