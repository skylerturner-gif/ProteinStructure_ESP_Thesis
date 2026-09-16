"""
scripts/eval_pdb_comparison.py

PDB-vs-AlphaFold comparison notebook, phase 1: builds PyG graphs for the
41 AF structures + 41 paired PDB crystal structures already sitting in
~/thesis/pdb_comparison_data/ (af/, pdb/ -- documented in
data/datasets/DATA_ORIGINS.md's "PDB Comparison Dataset" section; fully
processed through PDB2PQR/APBS/MSMS/ESP-sampling already, but with no
cached graphs yet), then runs frozen-backbone inference with both
full-dataset champions on each side separately. Answers: do the models do
significantly worse on real experimental (PDB) structures than on the
AlphaFold structures they were trained on?

Mirrors scripts/eval_seed_conformations.py's structure exactly (same
build_graph + load_model_frozen + ProteinGraphDataset + NormalizeESP +
ESPLoss + evaluate_test composition) -- only the input protein-id lists and
data roots differ (two fixed 41-protein sets here, vs. 50 synthetic seed
structures there).

Results are written to a separate output directory, never to the real
champion checkpoints' own test_predictions/ (evaluate_test deletes existing
*_pred.npz in predictions_dir before writing).

Run in the `pyg_env` conda environment (torch/PyG).

Usage:
    conda activate pyg_env
    python -u scripts/eval_pdb_comparison.py
    python -u scripts/eval_pdb_comparison.py --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
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
from src.utils.helpers import get_pipeline_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

CKPT_ROOT = Path("/home/student/thesis/checkpoints/full_dataset")
MODELS = {
    "attention": CKPT_ROOT / "attention_aa4_aq2_qq16",
    "distance":  CKPT_ROOT / "distance_aa8_aq2_qq24",
}

PDB_COMPARISON_ROOT = Path("/home/student/thesis/pdb_comparison_data")
AF_DATA_ROOT = PDB_COMPARISON_ROOT / "af"
PDB_DATA_ROOT = PDB_COMPARISON_ROOT / "pdb"

EVAL_ROOT = Path("/home/student/thesis/outputs/pdb_comparison/eval")
SUMMARY_CSV = Path("/home/student/thesis/outputs/pdb_comparison_eval_summary.csv")
LOG_FILE = Path("/home/student/thesis/outputs/pdb_comparison_eval.log")


def _build_one_graph(protein_id: str, data_root: Path, log) -> str:
    """Build and cache the graph for one protein. Mirrors
    pipelines/06_build_graphs.py's _build_one / eval_seed_conformations.py's
    _build_one_graph. Returns "ok", "skip", or "fail"."""
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


def _build_graphs(protein_ids: list[str], data_root: Path, label: str, log) -> list[str]:
    print(f"\nBuilding graphs ({label}, {len(protein_ids)} proteins)...")
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
    return built_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graphs + run frozen-backbone inference on the "
                     "paired AF/PDB comparison dataset."
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first N proteins per side.")
    args = parser.parse_args()

    log = get_pipeline_logger(LOG_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    af_ids = sorted(p.name for p in AF_DATA_ROOT.iterdir() if p.is_dir())
    pdb_ids = sorted(p.name for p in PDB_DATA_ROOT.iterdir() if p.is_dir())
    if args.limit is not None:
        af_ids, pdb_ids = af_ids[: args.limit], pdb_ids[: args.limit]
    print(f"AF side: {len(af_ids)} proteins   PDB side: {len(pdb_ids)} proteins")

    built_af = _build_graphs(af_ids, AF_DATA_ROOT, "af", log)
    built_pdb = _build_graphs(pdb_ids, PDB_DATA_ROOT, "pdb", log)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for name, ckpt_dir in MODELS.items():
        print(f"\n=== {name} ({ckpt_dir.name}) ===")
        model, ckpt = load_model_frozen(ckpt_dir, device)
        norm = NormalizeESP(ckpt["esp_mean"], ckpt["esp_std"])
        loss_fn = ESPLoss()

        for source, ids, data_root in [("af", built_af, AF_DATA_ROOT), ("pdb", built_pdb, PDB_DATA_ROOT)]:
            ds = ProteinGraphDataset(ids, data_root, transform=norm)
            out_dir = EVAL_ROOT / name / source
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_dir = out_dir / "test_predictions"

            results = evaluate_test(
                model, loss_fn, ds, device, ckpt,
                checkpoint_dir=out_dir, predictions_dir=pred_dir,
            )
            g = results["global"]
            print(f"  [{source}] RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  "
                  f"Pearson r (mean)={g.get('pearson_r', float('nan')):.4f}  proteins={g['n_proteins']}")

            for pid, v in results["per_protein"].items():
                summary_rows.append({"protein_id": pid, "source": source, "model": name, **v})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {len(summary_df)} rows -> {SUMMARY_CSV}")

    log.info("eval_pdb_comparison complete")
    print("\nDone.")


if __name__ == "__main__":
    main()
