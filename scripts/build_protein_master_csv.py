"""
scripts/build_protein_master_csv.py

Aggregate per-protein structural/chemical/geometric metadata (already
computed and persisted by notebooks/initial_protein_data_analysis.ipynb)
across the full dataset into one master CSV, joined with train/val/test
split assignment and per-protein test metrics from the two full-dataset
champion checkpoints (attention_aa4_aq2_qq16, distance_aa8_aq2_qq24).

AlphaFold-confidence fields (pLDDT, per-residue confidence) are
deliberately excluded — out of scope for this CSV, reserved for a future
AlphaFold-uncertainty notebook.

Refuses to run while any protein's graph-build-stage metadata is still
contaminated by the resample_query_density.py / rebuild_graphs_for_ids.py
symlink bug (see src.utils.io.is_graph_stats_contaminated) — run
scripts/repair_graph_stats.py first. Pass --allow-contaminated-graph-stats
to proceed anyway; affected rows get NaN in the graph-stat columns instead
of wrong numbers.

Run in the `pyg_env` conda environment (only hard dependency is pandas,
but kept single-env with the rest of this project's modeling-stage tools).

Usage:
    conda activate pyg_env
    python scripts/build_protein_master_csv.py
    python scripts/build_protein_master_csv.py --output /home/student/thesis/outputs/protein_master_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_split_manifest
from src.utils.config import get_data_root
from src.utils.io import is_graph_stats_contaminated, load_metadata

CKPT_ROOT       = Path("/home/student/thesis/checkpoints/full_dataset")
ATTENTION_CKPT  = CKPT_ROOT / "attention_aa4_aq2_qq16"
DISTANCE_CKPT   = CKPT_ROOT / "distance_aa8_aq2_qq24"

# Fields pulled straight from metadata.json, unchanged.
STRUCTURAL_FIELDS = [
    "sequence_length", "n_heavy_atoms", "net_charge", "ses_area",
    "n_vertices", "n_query_nodes",
    "interp_pearson_r", "interp_rmse",
    "esp_min", "esp_max", "esp_mean", "esp_std",
    "dipole_x", "dipole_y", "dipole_z", "dipole_magnitude",
    "radius_of_gyration", "asphericity", "acylindricity",
    "hydrophobic_fraction", "charged_fraction", "polar_fraction",
    "volume_A3", "surface_to_volume", "volume_per_atom",
    "area_per_atom", "normalized_area",
]
GRAPH_STAT_FIELDS = [
    "num_atom_nodes", "num_nodes_total",
    "num_bond_edges", "num_radial_edges", "num_aq_edges", "num_qq_edges",
    "num_edges_total",
]
IDENTITY_FIELDS = ["uniprot_id", "protein_name", "organism"]

# Per-protein fields pulled from each checkpoint's test_metrics.json / test_spatial_metrics.json,
# in the order they appear as <prefix>_<field> columns in the output CSV.
MODEL_METRIC_FIELDS = [
    "rmse", "mae", "pearson_r", "morans_i", "esp_error_spearman", "inference_time_s",
]
_TEST_METRICS_FIELDS    = ["rmse", "mae", "pearson_r", "inference_time_s"]
_SPATIAL_METRICS_FIELDS = ["morans_i", "esp_error_spearman"]


def _load_checkpoint_per_protein(ckpt_dir: Path) -> dict[str, dict]:
    """Merge a checkpoint's test_metrics.json + test_spatial_metrics.json per-protein dicts."""
    metrics_path = ckpt_dir / "test_metrics.json"
    spatial_path = ckpt_dir / "test_spatial_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not spatial_path.exists():
        raise FileNotFoundError(f"Missing {spatial_path}")

    metrics = json.loads(metrics_path.read_text())["per_protein"]
    spatial = json.loads(spatial_path.read_text())["per_protein"]

    merged: dict[str, dict] = {}
    for pid, m in metrics.items():
        row = {f: m.get(f) for f in _TEST_METRICS_FIELDS}
        s = spatial.get(pid, {})
        row.update({f: s.get(f) for f in _SPATIAL_METRICS_FIELDS})
        merged[pid] = row
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the master per-protein CSV: structural properties + "
                     "split assignment + champion-model test metrics."
    )
    parser.add_argument("--data-root", type=Path, default=None,
                         help="Override data_root from config.yaml.")
    parser.add_argument("--attention-checkpoint-dir", type=Path, default=ATTENTION_CKPT)
    parser.add_argument("--distance-checkpoint-dir", type=Path, default=DISTANCE_CKPT)
    parser.add_argument("--output", type=Path,
                         default=Path("/home/student/thesis/outputs/protein_master_dataset.csv"))
    parser.add_argument(
        "--allow-contaminated-graph-stats", action="store_true",
        help="Proceed even if some proteins' graph-stat metadata is still "
             "contaminated (see src.utils.io.is_graph_stats_contaminated). "
             "Affected rows get NaN in the graph-stat columns instead of "
             "wrong numbers. Default: refuse and point at "
             "scripts/repair_graph_stats.py.",
    )
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()

    print(f"Loading split manifest from {data_root} ...")
    train_ids, val_ids, test_ids = load_split_manifest(data_root)
    split_of = {pid: "train" for pid in train_ids}
    split_of.update({pid: "val" for pid in val_ids})
    split_of.update({pid: "test" for pid in test_ids})

    print(f"Loading champion checkpoint metrics...")
    attention_per_protein = _load_checkpoint_per_protein(args.attention_checkpoint_dir)
    distance_per_protein  = _load_checkpoint_per_protein(args.distance_checkpoint_dir)

    print(f"Aggregating metadata for all proteins under {data_root} ...")
    protein_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())

    rows = []
    contaminated_ids = []

    for protein_dir in protein_dirs:
        pid = protein_dir.name
        meta_path = protein_dir / f"{pid}_metadata.json"
        if not meta_path.exists():
            continue
        meta = load_metadata(pid, data_root)

        graph_stats_ok = not is_graph_stats_contaminated(meta)
        if not graph_stats_ok:
            contaminated_ids.append(pid)

        row = {"protein_id": pid, "split": split_of.get(pid)}
        for f in IDENTITY_FIELDS:
            row[f] = meta.get(f)
        for f in STRUCTURAL_FIELDS:
            row[f] = meta.get(f)
        for f in GRAPH_STAT_FIELDS:
            row[f] = meta.get(f) if graph_stats_ok else float("nan")
        row["pipeline_complete"] = meta.get("pipeline_complete")

        attn = attention_per_protein.get(pid)
        for f in MODEL_METRIC_FIELDS:
            row[f"attention_{f}"] = attn.get(f) if attn else float("nan")

        dist = distance_per_protein.get(pid)
        for f in MODEL_METRIC_FIELDS:
            row[f"distance_{f}"] = dist.get(f) if dist else float("nan")

        rows.append(row)

    if contaminated_ids and not args.allow_contaminated_graph_stats:
        print()
        print(f"REFUSING TO PROCEED: {len(contaminated_ids):,} proteins still have "
              f"contaminated graph-stat metadata.")
        print("Run this first:")
        print("    python scripts/repair_graph_stats.py --workers 8")
        print("(or pass --allow-contaminated-graph-stats to proceed with NaN'd "
              "graph-stat columns for those rows)")
        sys.exit(1)
    elif contaminated_ids:
        print(f"WARNING: proceeding with {len(contaminated_ids):,} proteins still "
              f"contaminated — their graph-stat columns are NaN in the output.")

    df = pd.DataFrame(rows)

    column_order = (
        ["protein_id"] + IDENTITY_FIELDS + ["split"]
        + STRUCTURAL_FIELDS + GRAPH_STAT_FIELDS + ["pipeline_complete"]
    )
    for prefix in ("attention", "distance"):
        column_order += [f"{prefix}_{f}" for f in MODEL_METRIC_FIELDS]
    df = df[column_order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_test_with_metrics = df["attention_pearson_r"].notna().sum()
    print()
    print(f"Wrote {len(df):,} proteins x {len(df.columns)} columns -> {args.output}")
    print(f"  split counts: {df['split'].value_counts().to_dict()}")
    print(f"  test-split rows with attention metrics: {n_test_with_metrics:,} "
          f"(expected {len(test_ids):,})")


if __name__ == "__main__":
    main()
