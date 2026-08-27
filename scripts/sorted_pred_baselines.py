"""Generate sorted-prediction plots for all baseline models.

Runs inference on the test split for each baseline, saves per-protein
_pred.npz files to <ckpt_dir>/test_predictions/ (same format as GNN
evaluators), then calls plot_sorted_predictions to produce
sorted_predictions.png in model_eval/baseline/<name>/.

Usage:
    conda run -n pyg_env python scripts/sorted_pred_baselines.py [--models all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_plots import plot_sorted_predictions
from src.baseline_models.coulomb.predictor import CoulombESP
from src.baseline_models.cnn3d.dataset import Vox3DDataset
from src.baseline_models.cnn3d.model import Vox3DCNN
from src.baseline_models.surface_dgcnn.dataset import SurfaceDataset, collate_surface
from src.baseline_models.surface_dgcnn.model import SurfaceDGCNN
from src.data.dataset import load_split_manifest
from src.utils.config import get_data_root
from src.utils.paths import ProteinPaths

import json


def _save_preds(pred_dir: Path, pid: str,
                true_esp: np.ndarray, pred_esp: np.ndarray) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{pid}_pred.npz",
                        true_esp=true_esp.astype(np.float32),
                        pred_esp=pred_esp.astype(np.float32))


# ── Coulomb ───────────────────────────────────────────────────────────────────

def run_coulomb(test_ids: list[str], data_root: Path,
                ckpt_dir: Path, eval_dir: Path) -> None:
    print(f"\n=== Coulomb ({len(test_ids)} proteins) ===")
    pred_dir = ckpt_dir / "test_predictions"
    n_done = 0
    for pid in test_ids:
        p = ProteinPaths(pid, data_root)
        if not p.pqr_path.exists() or not p.esp_path.exists():
            print(f"  SKIP {pid} — missing files")
            continue
        esp_data  = np.load(p.esp_path)
        query_idx = esp_data["query_idx"]
        query_pos = esp_data["verts"][query_idx]
        true_esp  = esp_data["esp_verts"][query_idx]
        predictor = CoulombESP(p.pqr_path)
        pred_esp  = predictor.predict(query_pos)
        _save_preds(pred_dir, pid, true_esp, pred_esp)
        n_done += 1
    print(f"  Saved {n_done} _pred.npz files → {pred_dir}")
    plot_sorted_predictions(ckpt_dir, save_dir=eval_dir, model_name="coulomb")


# ── DGCNN (chem) ─────────────────────────────────────────────────────────────

def run_dgcnn(test_ids: list[str], data_root: Path,
              ckpt_dir: Path, eval_dir: Path) -> None:
    print(f"\n=== Surface DGCNN chem ({len(test_ids)} proteins) ===")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = ckpt_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    max_pts = cfg.get("max_points") or None
    ds      = SurfaceDataset(test_ids, data_root, max_points=max_pts)
    loader  = DataLoader(ds, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=collate_surface, pin_memory=False)

    model = SurfaceDGCNN(
        in_features=cfg.get("in_features", ds.in_features),
        hidden_dim=cfg.get("hidden_dim", 128),
        n_query_k=cfg.get("n_query_k", 8),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
    model.eval()

    pred_dir = ckpt_dir / "test_predictions"
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            pid    = batch["protein_id"][0]
            pc     = batch["point_cloud"][0].to(device)
            ei     = batch["edge_index"][0].to(device)
            qsurf  = batch["query_surf"][0].to(device)
            target = batch["query_esp"][0].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(pc, ei, qsurf)
            _save_preds(pred_dir, pid,
                        target.cpu().float().numpy(),
                        pred.cpu().float().numpy())
            n_done += 1
    print(f"  Saved {n_done} _pred.npz files → {pred_dir}")
    plot_sorted_predictions(ckpt_dir, save_dir=eval_dir, model_name="surface_dgcnn_chem")


# ── CNN3D ─────────────────────────────────────────────────────────────────────

def run_cnn3d(test_ids: list[str], data_root: Path,
              ckpt_dir: Path, eval_dir: Path) -> None:
    print(f"\n=== CNN3D ({len(test_ids)} proteins) ===")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = ckpt_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    ds = Vox3DDataset(test_ids, data_root,
                      voxel_size=cfg.get("voxel_size", 1.0),
                      padding=cfg.get("padding", 5.0))

    def _collate_single(batch):
        return batch[0]

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=_collate_single, pin_memory=False)

    model = Vox3DCNN().to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
    model.eval()

    pred_dir = ckpt_dir / "test_predictions"
    n_done = 0
    with torch.no_grad():
        for sample in loader:
            pid    = sample["protein_id"]
            grid   = sample["grid"].to(device)
            coords = sample["query_coords"].to(device)
            target = sample["query_esp"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(grid, coords)
            _save_preds(pred_dir, pid,
                        target.cpu().float().numpy(),
                        pred.cpu().float().numpy())
            n_done += 1
    print(f"  Saved {n_done} _pred.npz files → {pred_dir}")
    plot_sorted_predictions(ckpt_dir, save_dir=eval_dir, model_name="cnn3d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["coulomb", "dgcnn", "cnn3d", "all"],
                        default=["all"])
    args = parser.parse_args()
    run_all = "all" in args.models
    models  = set(args.models)

    data_root  = Path(get_data_root())
    thesis     = data_root.parent
    ckpt_base  = thesis / "checkpoints" / "baseline"
    eval_base  = thesis / "model_eval" / "baseline"

    _, _, test_ids = load_split_manifest(data_root)
    print(f"Test proteins: {len(test_ids)}")

    if run_all or "coulomb" in models:
        run_coulomb(test_ids, data_root,
                    ckpt_dir=ckpt_base / "coulomb",
                    eval_dir=eval_base  / "coulomb")

    if run_all or "dgcnn" in models:
        run_dgcnn(test_ids, data_root,
                  ckpt_dir=ckpt_base / "surface_dgcnn_chem",
                  eval_dir=eval_base  / "surface_dgcnn_chem")

    if run_all or "cnn3d" in models:
        run_cnn3d(test_ids, data_root,
                  ckpt_dir=ckpt_base / "cnn3d",
                  eval_dir=eval_base  / "cnn3d")

    print("\nDone. Plots saved to model_eval/baseline/*/sorted_predictions.png")


if __name__ == "__main__":
    main()
