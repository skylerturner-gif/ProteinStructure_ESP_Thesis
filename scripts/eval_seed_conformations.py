"""
scripts/eval_seed_conformations.py

Seed-conformation pilot, phase B3 prerequisite: builds PyG graphs for the
synthetic seed protein_ids produced by scripts/run_seed_conformations.py
(/home/student/thesis/outputs/seed_conformation_ids.txt), then runs frozen-backbone inference
with both full-dataset champions and caches per-protein RMSE/MAE/Pearson r
+ per-query-node predictions -- exactly what
notebooks/seed_conformation_analysis.ipynb needs, without retraining
anything.

Graph building mirrors pipelines/06_build_graphs.py's own _build_one unit
(same build_graph() call, same atomic save, same metadata fields) --
called directly per synthetic id rather than through that script's CLI,
which only supports --all/--filter selection, not an explicit id list.

Inference reuses src/training/trainer.py's standalone evaluate_test() --
the exact function 07_train.py itself calls for its post-training test
pass, and what every other analysis script this session
(compute_vertex_error_stats.py, run_charge_probe.py,
compute_confidence_error_stats.py) has built on. Single GPU, no DDP (see
evaluate_test's own "rank 0 only" usage in 07_train.py) -- appropriate at
this scale (~50 proteins).

Critically, results are written to a SEPARATE directory
(/home/student/thesis/outputs/seed_conformations/eval/<model>/), never to the champion
checkpoint's own test_predictions/ -- that directory holds the real
848-protein test-split predictions relied on by every other notebook this
session; evaluate_test() deletes all *_pred.npz in its predictions_dir
before writing, so pointing it at the real checkpoint dir would silently
destroy that data.

Run in the `pyg_env` conda environment (torch/PyG).

Usage:
    conda activate pyg_env
    python -u scripts/eval_seed_conformations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.embedding_analysis import load_model_frozen
from src.data.dataset import ProteinGraphDataset
from src.data.graph_builder import build_graph
from src.data.transform import NormalizeESP
from src.training.loss import ESPLoss
from src.training.trainer import evaluate_test
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

CKPT_ROOT = Path("/home/student/thesis/checkpoints/full_dataset")
MODELS = {
    "attention": CKPT_ROOT / "attention_aa4_aq2_qq16",
    "distance":  CKPT_ROOT / "distance_aa8_aq2_qq24",
}

IDS_FILE = Path("/home/student/thesis/outputs/seed_conformation_ids.txt")
EVAL_ROOT = Path("/home/student/thesis/outputs/seed_conformations/eval")


def _build_one_graph(protein_id: str, data_root: Path, log) -> str:
    """Build and cache the graph for one synthetic protein. Mirrors
    pipelines/06_build_graphs.py's _build_one. Returns "ok", "skip", or "fail"."""
    p = ProteinPaths(protein_id, data_root)
    graph_path = p.graph_path()
    if graph_path.exists():
        return "skip"

    missing = [f for f in [p.pqr_path, p.mesh_path, p.esp_path] if not f.exists()]
    if missing:
        log.error("[%s] Missing input for graph build: %s", protein_id, [f.name for f in missing])
        return "fail"

    p.ensure_dirs()
    try:
        with timer() as t:
            data = build_graph(protein_id, data_root)
        tmp_path = graph_path.with_suffix(".pt.tmp")
        torch.save(data, tmp_path)
        tmp_path.rename(graph_path)
        update_metadata(protein_id, data_root=data_root, data={
            "num_atom_nodes":       int(data["atom"].num_nodes),
            "num_query_nodes":      int(data["query"].num_nodes),
            "num_nodes_total":      int(data.num_nodes),
            "num_bond_edges":       int(data["atom", "bond",   "atom"].num_edges),
            "num_radial_edges":     int(data["atom", "radial", "atom"].num_edges),
            "num_aq_edges":         int(data["atom", "aq",     "query"].num_edges),
            "num_qq_edges":         int(data["query", "qq",    "query"].num_edges),
            "num_edges_total":      int(data.num_edges),
            "time_graph_build_sec": t.rounded,
        })
        return "ok"
    except Exception as e:
        log.error("[%s] Graph build failed: %s", protein_id, e)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graphs + run frozen-backbone inference for the "
                     "seed-conformation pilot's synthetic protein IDs."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--ids-file", type=Path, default=IDS_FILE)
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    protein_ids = [l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()]
    print(f"{len(protein_ids)} synthetic protein IDs from {args.ids_file}")

    print("\nBuilding graphs...")
    n_ok = n_skip = n_fail = 0
    built_ids = []
    for i, pid in enumerate(protein_ids, 1):
        status = _build_one_graph(pid, data_root, log)
        if status in ("ok", "skip"):
            built_ids.append(pid)
            n_ok += status == "ok"
            n_skip += status == "skip"
        else:
            n_fail += 1
        if i % 10 == 0 or i == len(protein_ids):
            print(f"\r  {i}/{len(protein_ids)}  (built={n_ok} cached={n_skip} failed={n_fail})", end="", flush=True)
    print()
    if n_fail:
        print(f"  WARNING: {n_fail} graph builds failed -- excluded from evaluation.")

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, ckpt_dir in MODELS.items():
        print(f"\n=== {name} ({ckpt_dir.name}) ===")
        model, ckpt = load_model_frozen(ckpt_dir, device)
        norm = NormalizeESP(ckpt["esp_mean"], ckpt["esp_std"])
        ds = ProteinGraphDataset(built_ids, data_root, transform=norm)
        loss_fn = ESPLoss()

        out_dir = EVAL_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = out_dir / "test_predictions"

        results = evaluate_test(
            model, loss_fn, ds, device, ckpt,
            checkpoint_dir=out_dir, predictions_dir=pred_dir,
        )
        g = results["global"]
        print(f"  RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  Pearson r (mean)={g.get('pearson_r', float('nan')):.4f}  "
              f"proteins={g['n_proteins']}")
        all_results[name] = results

    summary_rows = []
    for name, results in all_results.items():
        for pid, v in results["per_protein"].items():
            summary_rows.append({"protein_id": pid, "model": name, **v})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("/home/student/thesis/outputs/seed_conformation_eval_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {len(summary_df)} rows -> {summary_path}")

    log.info("eval_seed_conformations complete")
    print("\nDone.")


if __name__ == "__main__":
    main()
