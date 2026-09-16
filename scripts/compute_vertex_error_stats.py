"""
scripts/compute_vertex_error_stats.py

Per-vertex error analysis for the two full-dataset champion checkpoints
(attention_aa4_aq2_qq16, distance_aa8_aq2_qq24): splits full-mesh error into
query-node (direct model prediction) vs. interpolated (RBF-reconstructed)
populations, buckets pooled error by true-ESP value (reusing the
range_delta_breakdown bucket convention via the new pooled_bucket_table),
and — for a small set of illustrative high-error proteins only — computes
full-mesh Moran's I and saves per-vertex arrays for spatial visualization.

Query-node error is read directly from test_predictions/*.npz (exact,
free); only the interpolated (non-query) vertices need a fresh
reconstruct_full_mesh call. Verified empirically: reconstruct_full_mesh
reproduces pred_esp exactly at the query positions themselves (diff=0.0),
so reading pred_esp directly for the query population (rather than slicing
the reconstruction) is for clarity and a free cross-check, not correctness.

Three phases, run per checkpoint:
  1. (parallel, expensive — ~15-20 min at --workers 8 for 848 proteins x
     2 champions) per-protein query/interpolated/whole-mesh RMSE+MAE,
     cached as <ckpt_dir>/test_vertex_error_metrics.json.
  2. (cheap, main process) flatten both checkpoints' caches into
     /home/student/thesis/outputs/vertex_error_dataset.csv, pool vertices and write
     /home/student/thesis/outputs/vertex_error_buckets.csv.
  3. (small-scale, ~20-40 proteins total) full-mesh Moran's I + per-vertex
     example npz files for the highest-error / most hotspot-y / most
     diffuse proteins, under /home/student/thesis/outputs/vertex_examples/.

Run in the `pyg_env` conda environment (no torch needed, but kept
single-env with the rest of this project's modeling-stage tools).

Usage:
    conda activate pyg_env
    python scripts/compute_vertex_error_stats.py --dry-run --limit 10
    python scripts/compute_vertex_error_stats.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_plots import pooled_bucket_table
from src.analysis.model_spatial import _morans_i
from src.surface.esp_mapping import reconstruct_full_mesh
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger
from src.utils.paths import ProteinPaths
from src.utils.parallel import run_parallel

CKPT_ROOT      = Path("/home/student/thesis/checkpoints/full_dataset")
ATTENTION_CKPT = CKPT_ROOT / "attention_aa4_aq2_qq16"
DISTANCE_CKPT  = CKPT_ROOT / "distance_aa8_aq2_qq24"

MODELS = ("attention", "distance")

_SCALAR_FIELDS = [
    "query_rmse", "query_mae", "interpolated_rmse", "interpolated_mae",
    "whole_mesh_rmse", "whole_mesh_mae", "n_query", "n_interpolated", "n_verts",
]


# ── Shared per-protein computation ──────────────────────────────────────────

def _load_protein_arrays(protein_id: str, data_root: str, ckpt_dir: str) -> dict | None:
    """Load a protein's esp.npz + this checkpoint's pred.npz. None if either is missing."""
    p = ProteinPaths(protein_id, Path(data_root))
    if not p.esp_path.exists():
        return None
    pred_path = Path(ckpt_dir) / "test_predictions" / f"{protein_id}_pred.npz"
    if not pred_path.exists():
        return None

    esp_npz  = np.load(p.esp_path)
    pred_npz = np.load(pred_path)
    return {
        "verts":     esp_npz["verts"],
        "esp_verts": esp_npz["esp_verts"],
        "query_idx": esp_npz["query_idx"],
        "query_pos": pred_npz["query_pos"],
        "pred_esp":  pred_npz["pred_esp"],
    }


def _reconstruct_whole_mesh(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid whole-mesh prediction: exact pred_esp at query nodes, RBF elsewhere."""
    verts, query_idx = arrays["verts"], arrays["query_idx"]
    is_query = np.zeros(len(verts), dtype=bool)
    is_query[query_idx] = True

    whole_mesh_pred = np.empty(len(verts), dtype=np.float32)
    whole_mesh_pred[query_idx] = arrays["pred_esp"]
    pred_full = reconstruct_full_mesh(arrays["query_pos"], arrays["pred_esp"], verts, method="multiquadric")
    whole_mesh_pred[~is_query] = pred_full[~is_query]

    return whole_mesh_pred, is_query


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def _mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


# ── Phase 1: bulk per-protein stats (parallel-safe, no logger arg) ─────────

def _phase1_one(protein_id: str, data_root: str, ckpt_dir: str) -> dict | None:
    """Query/interpolated/whole-mesh error stats + pooled bucket arrays for one protein."""
    arrays = _load_protein_arrays(protein_id, data_root, ckpt_dir)
    if arrays is None:
        return None

    try:
        whole_mesh_pred, is_query = _reconstruct_whole_mesh(arrays)
    except Exception as e:
        return {"protein_id": protein_id, "error": str(e)}

    if not np.isfinite(whole_mesh_pred).all():
        return {"protein_id": protein_id, "error": "non-finite reconstruction output"}

    esp_verts = arrays["esp_verts"]
    whole_mesh_error = whole_mesh_pred - esp_verts
    query_error       = whole_mesh_error[is_query]
    interpolated_error = whole_mesh_error[~is_query]

    return {
        "protein_id":         protein_id,
        "query_rmse":         _rmse(query_error),
        "query_mae":          _mae(query_error),
        "interpolated_rmse":  _rmse(interpolated_error) if interpolated_error.size else None,
        "interpolated_mae":   _mae(interpolated_error) if interpolated_error.size else None,
        "whole_mesh_rmse":    _rmse(whole_mesh_error),
        "whole_mesh_mae":     _mae(whole_mesh_error),
        "n_query":            int(is_query.sum()),
        "n_interpolated":     int((~is_query).sum()),
        "n_verts":            int(len(esp_verts)),
        # pooled arrays for the bucket table (float32, dropped before JSON caching)
        "query_true":        esp_verts[is_query].astype(np.float32),
        "query_abs_error":   np.abs(query_error).astype(np.float32),
        "interp_true":       esp_verts[~is_query].astype(np.float32),
        "interp_abs_error":  np.abs(interpolated_error).astype(np.float32),
    }


def _run_phase1(protein_ids: list[str], data_root: Path, ckpt_dir: Path, workers: int, log) -> dict[str, dict]:
    """Run phase 1 for every protein, serial or parallel. Returns {protein_id: result_dict}."""
    results: dict[str, dict] = {}
    if workers == 1:
        for i, pid in enumerate(protein_ids, 1):
            r = _phase1_one(pid, str(data_root), str(ckpt_dir))
            if r is not None:
                results[pid] = r
            if i % 50 == 0 or i == len(protein_ids):
                print(f"\r  phase 1: {i}/{len(protein_ids)}", end="", flush=True)
        print()
    else:
        raw = run_parallel(
            _phase1_one,
            [(pid, str(data_root), str(ckpt_dir)) for pid in protein_ids],
            n_workers=workers,
            label="phase 1",
        )
        for pid, outcome in raw:
            if isinstance(outcome, Exception):
                log.error("[%s] Worker exception: %s", pid, outcome)
                continue
            if outcome is not None:
                results[pid] = outcome

    n_error = sum(1 for r in results.values() if "error" in r)
    if n_error:
        log.warning("%d proteins had reconstruction errors (see log)", n_error)
        for pid, r in results.items():
            if "error" in r:
                log.error("[%s] %s", pid, r["error"])
    return {pid: r for pid, r in results.items() if "error" not in r}


# ── Phase 2: aggregate + cache ──────────────────────────────────────────────

def _write_phase2(model: str, ckpt_dir: Path, results: dict[str, dict], output_dir: Path, dry_run: bool) -> pd.DataFrame:
    """Write test_vertex_error_metrics.json for this checkpoint; return this model's bucket rows."""
    per_protein = {
        pid: {k: r[k] for k in _SCALAR_FIELDS}
        for pid, r in results.items()
    }
    means = {
        f"mean_{k}": float(np.mean([v[k] for v in per_protein.values() if v[k] is not None]))
        for k in ("query_rmse", "interpolated_rmse", "whole_mesh_rmse")
    }
    cache = {"global": {**means, "n_proteins": len(per_protein)}, "per_protein": per_protein}

    cache_path = ckpt_dir / "test_vertex_error_metrics.json"
    if not dry_run:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  [{model}] Cached -> {cache_path}")
    else:
        print(f"  [{model}] (dry run) would cache -> {cache_path}  ({len(per_protein)} proteins)")

    query_true      = np.concatenate([r["query_true"]  for r in results.values()]) if results else np.array([])
    query_abs_err   = np.concatenate([r["query_abs_error"]  for r in results.values()]) if results else np.array([])
    interp_true     = np.concatenate([r["interp_true"] for r in results.values()]) if results else np.array([])
    interp_abs_err  = np.concatenate([r["interp_abs_error"] for r in results.values()]) if results else np.array([])

    bucket_query  = pooled_bucket_table(query_true, query_abs_err)
    bucket_query.insert(0, "population", "query")
    bucket_interp = pooled_bucket_table(interp_true, interp_abs_err)
    bucket_interp.insert(0, "population", "interpolated")

    bucket_df = pd.concat([bucket_query, bucket_interp], ignore_index=True)
    bucket_df.insert(0, "model", model)
    return bucket_df


def _write_vertex_error_dataset(all_per_protein: dict[str, dict[str, dict]], output_dir: Path, dry_run: bool) -> None:
    """Flatten both checkpoints' per-protein scalars into one CSV, <prefix>_<field> naming."""
    protein_ids = sorted(set().union(*(d.keys() for d in all_per_protein.values())))
    rows = []
    for pid in protein_ids:
        row = {"protein_id": pid}
        for model in MODELS:
            entry = all_per_protein[model].get(pid)
            for field in _SCALAR_FIELDS:
                row[f"{model}_{field}"] = entry[field] if entry else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = output_dir / "vertex_error_dataset.csv"
    if not dry_run:
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df):,} rows -> {out_path}")
    else:
        print(f"(dry run) would write {len(df):,} rows -> {out_path}")


# ── Phase 3: illustrative examples ──────────────────────────────────────────

def _run_phase3(
    model: str, ckpt_dir: Path, data_root: Path,
    phase1_results: dict[str, dict], n_candidates: int,
    output_dir: Path, dry_run: bool, log,
) -> list[dict]:
    """Recompute reconstruction + full-mesh Moran's I for the worst-RMSE candidates;
    select hotspot/diffuse/worst-overall examples and save per-vertex npz files."""
    ranked = sorted(phase1_results.items(), key=lambda kv: kv[1]["whole_mesh_rmse"], reverse=True)
    candidates = ranked[:n_candidates]

    enriched = []
    for pid, r in candidates:
        arrays = _load_protein_arrays(pid, str(data_root), str(ckpt_dir))
        if arrays is None:
            continue
        whole_mesh_pred, is_query = _reconstruct_whole_mesh(arrays)
        error = whole_mesh_pred - arrays["esp_verts"]
        mi = _morans_i(np.abs(error).astype(np.float64), arrays["verts"].astype(np.float64), k=8)
        enriched.append({
            "protein_id": pid, "whole_mesh_rmse": r["whole_mesh_rmse"],
            "morans_i_full_mesh": mi, "n_verts": r["n_verts"], "n_query": r["n_query"],
            "verts": arrays["verts"], "esp_verts": arrays["esp_verts"],
            "whole_mesh_pred": whole_mesh_pred, "error": error, "is_query": is_query,
        })
        log.info("[phase3 %s] %s  rmse=%.4f  morans_i=%.4f", model, pid, r["whole_mesh_rmse"], mi)

    if not enriched:
        return []

    by_rmse = sorted(enriched, key=lambda d: d["whole_mesh_rmse"], reverse=True)
    by_mi   = sorted(enriched, key=lambda d: d["morans_i_full_mesh"], reverse=True)

    selected: dict[str, tuple[dict, str]] = {}
    for d in by_rmse[:2]:
        selected.setdefault(d["protein_id"], (d, "worst_whole_mesh_rmse"))
    for d in by_mi[:2]:
        selected.setdefault(d["protein_id"], (d, "hotspot_high_morans_i"))
    for d in by_mi[-2:]:
        selected.setdefault(d["protein_id"], (d, "diffuse_low_morans_i"))

    examples_dir = output_dir / "vertex_examples"
    manifest_rows = []
    for pid, (d, reason) in selected.items():
        npz_name = f"{model}_{pid}_vertex_error.npz"
        manifest_rows.append({
            "protein_id": pid, "model": model, "selection_reason": reason,
            "whole_mesh_rmse": d["whole_mesh_rmse"], "morans_i_full_mesh": d["morans_i_full_mesh"],
            "n_verts": d["n_verts"], "n_query": d["n_query"], "npz_filename": npz_name,
        })
        if not dry_run:
            examples_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                examples_dir / npz_name,
                position=d["verts"].astype(np.float32),
                true_esp=d["esp_verts"].astype(np.float32),
                pred_esp=d["whole_mesh_pred"].astype(np.float32),
                error=d["error"].astype(np.float32),
                is_query=d["is_query"],
            )

    print(f"  [{model}] phase 3: {len(manifest_rows)} example proteins selected"
          f"{' (dry run, npz not written)' if dry_run else f' -> {examples_dir}'}")
    return manifest_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-vertex error analysis: query vs. interpolated, true-ESP "
                     "buckets, and spatial hotspot examples for the two champions."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--attention-checkpoint-dir", type=Path, default=ATTENTION_CKPT)
    parser.add_argument("--distance-checkpoint-dir", type=Path, default=DISTANCE_CKPT)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/student/thesis/outputs"))
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel workers for phase 1 (default 1). Recommend 8 "
                              "for the full run — ~15-20 min for 848 proteins x 2 "
                              "champions (~2 hours serial).")
    parser.add_argument("--id-file", type=Path, default=None,
                         help="Restrict to these protein IDs (smoke test).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N proteins (smoke test).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run the full pipeline but write no cache/CSV/npz files.")
    parser.add_argument("--n-examples-candidates", type=int, default=20,
                         help="Phase-3 candidate pool size per checkpoint (default 20).")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even if /home/student/thesis/outputs/vertex_error_dataset.csv and "
                              "/home/student/thesis/outputs/vertex_error_buckets.csv already exist.")
    args = parser.parse_args()

    if (
        not args.force and not args.dry_run
        and (args.output_dir / "vertex_error_dataset.csv").exists()
        and (args.output_dir / "vertex_error_buckets.csv").exists()
    ):
        print(f"Outputs already exist under {args.output_dir}/ — skipping (pass --force to recompute).")
        return

    data_root = args.data_root or get_data_root()
    log = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    ckpt_dirs = {"attention": args.attention_checkpoint_dir, "distance": args.distance_checkpoint_dir}

    restrict_ids = None
    if args.id_file is not None:
        restrict_ids = {l.strip() for l in args.id_file.read_text().splitlines() if l.strip()}

    all_per_protein: dict[str, dict[str, dict]] = {}
    all_bucket_dfs = []
    all_manifest_rows = []

    for model, ckpt_dir in ckpt_dirs.items():
        pred_dir = ckpt_dir / "test_predictions"
        protein_ids = sorted(p.name.replace("_pred.npz", "") for p in pred_dir.glob("*_pred.npz"))
        if restrict_ids is not None:
            protein_ids = [p for p in protein_ids if p in restrict_ids]
        if args.limit is not None:
            protein_ids = protein_ids[: args.limit]

        print(f"\n=== {model} ({ckpt_dir.name}) — {len(protein_ids)} proteins ===")

        results = _run_phase1(protein_ids, data_root, ckpt_dir, args.workers, log)
        print(f"  phase 1: {len(results)}/{len(protein_ids)} succeeded")

        bucket_df = _write_phase2(model, ckpt_dir, results, args.output_dir, args.dry_run)
        all_bucket_dfs.append(bucket_df)
        all_per_protein[model] = results

        manifest_rows = _run_phase3(
            model, ckpt_dir, data_root, results, args.n_examples_candidates,
            args.output_dir, args.dry_run, log,
        )
        all_manifest_rows.extend(manifest_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_vertex_error_dataset(all_per_protein, args.output_dir, args.dry_run)

    buckets_df = pd.concat(all_bucket_dfs, ignore_index=True)
    buckets_path = args.output_dir / "vertex_error_buckets.csv"
    if not args.dry_run:
        buckets_df.to_csv(buckets_path, index=False)
        print(f"Wrote {len(buckets_df):,} rows -> {buckets_path}")
    else:
        print(f"(dry run) would write {len(buckets_df):,} rows -> {buckets_path}")

    if all_manifest_rows:
        manifest_path = args.output_dir / "vertex_examples" / "example_index.csv"
        if not args.dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_manifest_rows)
            print(f"Wrote {len(all_manifest_rows)} rows -> {manifest_path}")
        else:
            print(f"(dry run) would write {len(all_manifest_rows)} rows -> {manifest_path}")

    log.info("compute_vertex_error_stats complete")


if __name__ == "__main__":
    main()
