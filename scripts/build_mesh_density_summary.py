"""
scripts/build_mesh_density_summary.py

Mesh query-density ablation ("Phase 3" roadmap item), summary step: both
champions were trained at the standard 5% curvature-sampled query density.
This script does not run any model -- it just collects the per-protein
`test_metrics.json` already written by:
  - the standard training/eval pipeline (5% baseline, in each checkpoint dir)
  - scripts/eval_mesh_density.py, run separately for {attention, distance} x
    {10%, 25%} query density (frozen checkpoint, eval-only, no retraining)
into one long-format CSV for notebook analysis.

Usage:
    conda activate pyg_env
    python scripts/build_mesh_density_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CKPT_ROOT = Path("/home/student/thesis/checkpoints/full_dataset")
DENSITY_EVAL_ROOT = Path("/home/student/thesis/model_eval/mesh_density_eval")

MODELS = ["attention_aa4_aq2_qq16", "distance_aa8_aq2_qq24"]

# (density_label, metrics_path_fn)
SOURCES = {
    5: lambda model: CKPT_ROOT / model / "test_metrics.json",
    10: lambda model: DENSITY_EVAL_ROOT / model / "density_10" / "test_metrics.json",
    25: lambda model: DENSITY_EVAL_ROOT / model / "density_25" / "test_metrics.json",
}

OUT_CSV = Path("/home/student/thesis/outputs/mesh_density_eval_summary.csv")


def main() -> None:
    rows = []
    for model in MODELS:
        model_short = "attention" if model.startswith("attention") else "distance"
        for density, path_fn in SOURCES.items():
            path = path_fn(model)
            data = json.loads(path.read_text())
            g = data["global"]
            print(f"{model_short} @ {density}%: r={g['pearson_r']:.4f} rmse={g['rmse']:.4f} "
                  f"mae={g['mae']:.4f} n={g['n_proteins']}  ({path})")
            for protein_id, m in data["per_protein"].items():
                rows.append({
                    "protein_id": protein_id,
                    "model": model_short,
                    "density": density,
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "pearson_r": m["pearson_r"],
                    "n_query_nodes": m["n_query_nodes"],
                    "inference_time_s": m["inference_time_s"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows ({df['protein_id'].nunique()} proteins x "
          f"{df['model'].nunique()} models x {df['density'].nunique()} densities) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
