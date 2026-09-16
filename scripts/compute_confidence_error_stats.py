"""
scripts/compute_confidence_error_stats.py

Joins AlphaFold uncertainty (pLDDT, PAE) and local ESP-field volatility
onto per-query-vertex model error, for both full-dataset champions, and
caches compact artifacts under /home/student/thesis/outputs/ so
notebooks/alphafold_uncertainty_analysis.ipynb only ever loads results.

No GPU/torch needed: reuses each champion's cached test-split predictions
at <ckpt_dir>/test_predictions/<protein_id>_pred.npz (see
compute_vertex_error_stats.py), combined with esp/<id>_esp.npz, the PQR
(atom coords + resid), metadata.json's plddt_per_residue, and the PAE json
(optional per protein). Pure numpy/pandas/scipy.

Outputs, under /home/student/thesis/outputs/:
    confidence_error_dataset.csv.gz    -- per-query-vertex flat table
    confidence_error_buckets.csv       -- pooled mean/median |error| by
                                           pLDDT confidence band and by
                                           local-ESP-volatility decile
    confidence_error_protein_summary.csv -- per-protein x model Spearman
                                           correlations + coverage flags

Run in the `pyg_env` conda environment (no torch import needed, but kept
single-env with the rest of this project's analysis tooling).

Usage:
    conda activate pyg_env
    python -u scripts/compute_confidence_error_stats.py --limit 20   # smoke test
    python -u scripts/compute_confidence_error_stats.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.esp_intensity import local_esp_stats
from src.analysis.residue_confidence import query_confidence_table
from src.surface.esp_mapping import reconstruct_full_mesh
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger
from src.utils.paths import ProteinPaths

CKPT_ROOT = Path("/home/student/thesis/checkpoints/full_dataset")
MODELS = {
    "attention": CKPT_ROOT / "attention_aa4_aq2_qq16",
    "distance":  CKPT_ROOT / "distance_aa8_aq2_qq24",
}

PLDDT_BANDS = (
    ("very_low", 0, 50),
    ("low", 50, 70),
    ("confident", 70, 90),
    ("very_high", 90, 101),  # AF confidence is capped at 100; 101 makes the band inclusive
)


def _plddt_band(plddt: np.ndarray) -> np.ndarray:
    band = np.full(len(plddt), "unknown", dtype=object)
    for name, lo, hi in PLDDT_BANDS:
        band[(plddt >= lo) & (plddt < hi)] = name
    return band


def _one_protein_one_model(
    protein_id: str, data_root: Path, ckpt_dir: Path
) -> dict | None:
    """One protein's per-query-vertex confidence + local-ESP + error rows."""
    pred_path = ckpt_dir / "test_predictions" / f"{protein_id}_pred.npz"
    if not pred_path.exists():
        return None

    conf_df = query_confidence_table(protein_id, data_root)
    if conf_df is None or len(conf_df) == 0:
        return None

    paths = ProteinPaths(protein_id, data_root)
    esp_npz = np.load(paths.esp_path)
    verts, esp_verts, query_idx = esp_npz["verts"], esp_npz["esp_verts"], esp_npz["query_idx"]

    pred_npz = np.load(pred_path)
    pred_esp = pred_npz["pred_esp"]
    if len(pred_esp) != len(query_idx):
        return None

    local = local_esp_stats(verts, esp_verts, query_idx, k=16)

    true_esp_at_query = esp_verts[query_idx]
    abs_error_at_query = np.abs(pred_esp - true_esp_at_query)

    # conf_df is indexed by position within the ORIGINAL query_idx array
    # (rows dropped for out-of-range resid) -> map back via conf_df["query_idx"]
    pos_in_query = {qi: i for i, qi in enumerate(query_idx)}
    sel = np.array([pos_in_query[qi] for qi in conf_df["query_idx"]], dtype=np.int64)

    return {
        "protein_id":      protein_id,
        "resid":           conf_df["resid"].to_numpy(),
        "plddt":           conf_df["plddt"].to_numpy(dtype=np.float32),
        "pae_mean":        conf_df["pae_mean"].to_numpy(dtype=np.float32),
        "local_esp_std":   local["local_esp_std"][sel],
        "local_esp_grad":  local["local_esp_grad"][sel],
        "true_esp":        true_esp_at_query[sel],
        "abs_error":       abs_error_at_query[sel],
        "n_out_of_range":  conf_df.attrs["n_out_of_range"],
        "n_query":         conf_df.attrs["n_query"],
        "has_pae":         conf_df.attrs["has_pae"],
    }


def _quantile_bucket_table(values: np.ndarray, abs_error: np.ndarray, n_bins: int, label: str) -> pd.DataFrame:
    valid = np.isfinite(values) & np.isfinite(abs_error)
    values, abs_error = values[valid], abs_error[valid]
    if len(values) == 0:
        return pd.DataFrame()
    try:
        bins = pd.qcut(values, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    df = pd.DataFrame({"bin": bins, "abs_error": abs_error, "value": values})
    out = df.groupby("bin", observed=True).agg(
        n_vertices=("abs_error", "size"),
        mean_value=("value", "mean"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
    ).reset_index()
    out.insert(0, "predictor", label)
    out["bin"] = out["bin"].astype(str)
    return out


def _plddt_bucket_table(plddt: np.ndarray, abs_error: np.ndarray) -> pd.DataFrame:
    band = _plddt_band(plddt)
    df = pd.DataFrame({"bin": band, "abs_error": abs_error})
    out = df.groupby("bin", observed=True).agg(
        n_vertices=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
    ).reset_index()
    order = {name: i for i, (name, _, _) in enumerate(PLDDT_BANDS)}
    out["_order"] = out["bin"].map(order).fillna(99)
    out = out.sort_values("_order").drop(columns="_order")
    out.insert(0, "predictor", "plddt_band")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join AlphaFold pLDDT/PAE + local ESP volatility onto "
                     "per-query-vertex model error for both champions."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/student/thesis/outputs"))
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first N proteins.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = args.output_dir / "confidence_error_dataset.csv.gz"
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists — skipping (pass --force to recompute).")
        return

    data_root = args.data_root or get_data_root()
    log = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    protein_summary_rows = []

    for model, ckpt_dir in MODELS.items():
        pred_dir = ckpt_dir / "test_predictions"
        protein_ids = sorted(p.name.replace("_pred.npz", "") for p in pred_dir.glob("*_pred.npz"))
        if args.limit is not None:
            protein_ids = protein_ids[: args.limit]
        print(f"\n=== {model} ({ckpt_dir.name}) — {len(protein_ids)} proteins ===")

        n_done, n_skipped, n_pae_missing, n_out_of_range_total = 0, 0, 0, 0
        for i, pid in enumerate(protein_ids, 1):
            result = _one_protein_one_model(pid, data_root, ckpt_dir)
            if result is None:
                n_skipped += 1
            else:
                n = len(result["resid"])
                for j in range(n):
                    all_rows.append((
                        pid, model, int(result["resid"][j]), float(result["plddt"][j]),
                        float(result["pae_mean"][j]), float(result["local_esp_std"][j]),
                        float(result["local_esp_grad"][j]), float(result["true_esp"][j]),
                        float(result["abs_error"][j]),
                    ))
                n_out_of_range_total += result["n_out_of_range"]
                if not result["has_pae"]:
                    n_pae_missing += 1

                plddt, abs_err = result["plddt"], result["abs_error"]
                pae, local_std = result["pae_mean"], result["local_esp_std"]
                r_plddt = spearmanr(plddt, abs_err).correlation if len(plddt) > 2 else np.nan
                r_pae = (spearmanr(pae, abs_err).correlation
                         if result["has_pae"] and np.isfinite(pae).sum() > 2 else np.nan)
                r_local = spearmanr(local_std, abs_err).correlation if len(local_std) > 2 else np.nan
                protein_summary_rows.append({
                    "protein_id": pid, "model": model,
                    "n_query": result["n_query"], "n_out_of_range": result["n_out_of_range"],
                    "has_pae": result["has_pae"], "mean_plddt": float(plddt.mean()),
                    "spearman_plddt_error": r_plddt, "spearman_pae_error": r_pae,
                    "spearman_local_esp_std_error": r_local,
                })
                n_done += 1
            if i % 50 == 0 or i == len(protein_ids):
                print(f"\r  {i}/{len(protein_ids)}  (ok={n_done} skipped={n_skipped})", end="", flush=True)
        print()
        log.info(
            "[%s] done=%d skipped=%d pae_missing=%d out_of_range_query_nodes=%d",
            model, n_done, n_skipped, n_pae_missing, n_out_of_range_total,
        )
        print(f"  {model}: {n_done} proteins ok, {n_skipped} skipped, "
              f"{n_pae_missing} missing PAE, {n_out_of_range_total} out-of-range query nodes dropped")

    dataset_df = pd.DataFrame(
        all_rows,
        columns=["protein_id", "model", "resid", "plddt", "pae_mean",
                 "local_esp_std", "local_esp_grad", "true_esp", "abs_error"],
    )
    dataset_df.to_csv(out_path, index=False, compression="gzip")
    print(f"\nWrote {len(dataset_df):,} rows -> {out_path}")

    bucket_dfs = []
    for model in MODELS:
        sub = dataset_df[dataset_df["model"] == model]
        b1 = _plddt_bucket_table(sub["plddt"].to_numpy(), sub["abs_error"].to_numpy())
        b2 = _quantile_bucket_table(sub["local_esp_std"].to_numpy(), sub["abs_error"].to_numpy(), 10, "local_esp_std_decile")
        b3 = _quantile_bucket_table(sub["pae_mean"].to_numpy(), sub["abs_error"].to_numpy(), 10, "pae_mean_decile")
        for b in (b1, b2, b3):
            b.insert(0, "model", model)
        bucket_dfs.extend([b1, b2, b3])
    buckets_df = pd.concat(bucket_dfs, ignore_index=True)
    buckets_path = args.output_dir / "confidence_error_buckets.csv"
    buckets_df.to_csv(buckets_path, index=False)
    print(f"Wrote {len(buckets_df):,} rows -> {buckets_path}")

    summary_df = pd.DataFrame(protein_summary_rows)
    summary_path = args.output_dir / "confidence_error_protein_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {len(summary_df):,} rows -> {summary_path}")

    log.info("compute_confidence_error_stats complete")
    print("\nDone.")


if __name__ == "__main__":
    main()
