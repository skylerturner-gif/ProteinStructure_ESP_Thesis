"""
scripts/slim_graph_edge_attr.py

Convert already-cached graphs from the baked-``edge_attr`` format to the slim
``edge_dist`` format, writing a new dataset root and leaving the original cache
untouched.

Why convert rather than rebuild: a full rebuild re-runs MDAnalysis bond
detection, three kNN passes and the curvature/normal computation per protein.
The RBF expansion it produces is a pure function of the per-edge distance, and
that distance is exactly recoverable from the positions already in the file, so
conversion is a cheap read-transform-write.

What it does per edge store:
  - recompute  dist = ||pos[dst] - pos[src]||  from the stored positions
  - for bond edges, split off column 0 (bond_order) from edge_attr
  - drop edge_attr, keep edge_dist (+ bond_order)

Why recomputing the distance is safe: every distance in graph_builder comes
from a kNN over the same positions it stores, with no offset applied. Verified
against the cache — re-expanding the recomputed distances reproduces the stored
edge_attr to max|Δ| ≈ 1.8e-06 on values in [0, 1], and end-to-end model outputs
agree to max|Δ| ≈ 3e-07, both far below bf16 resolution. Use --verify-only to
re-confirm on your own data before writing anything.

Destination is explicit and mandatory
-------------------------------------
Pass --dest-root to build a parallel dataset root (the safe default: the
original cache is never modified, so a bad conversion costs nothing and both
formats can be trained side by side). --in-place is available but must be asked
for by name.

Sidecars copied into --dest-root so it is trainable on its own:
  <pid>/<pid>_metadata.json   DynamicBatchSampler reads num_edges_total;
                              split_dataset reads sequence_length/net_charge
  split_manifest.json         identical splits as the source root
  esp_stats.json              avoids rescanning esp/*.npz for normalisation

These are COPIED, never symlinked. Symlinking metadata from an isolated dest
root back to a shared source root is what corrupted graph-build stats before
(see src/utils/io.py is_graph_stats_contaminated and
scripts/repair_graph_stats.py) — a later write through the symlink landed on
the shared file. Copies cannot do that.

Note on symlinked source roots: tier roots like
full_protein_dataset_size_large600/ are directories of symlinks into
full_protein_dataset/. Reads follow them transparently. With --dest-root the
output is always a real file, so no symlink is ever written through. With
--in-place the write lands on the shared underlying file and converts it for
every root pointing at it — the resolved target is printed so that is never a
surprise.

Usage
-----
    # 1. check recoverability, write nothing
    python scripts/slim_graph_edge_attr.py --source-root <src> --all --verify-only

    # 2. build a slim copy of a tier's split into a new root
    python scripts/slim_graph_edge_attr.py \\
        --source-root /home/student/thesis/full_protein_dataset_size_large600 \\
        --dest-root   /home/student/thesis/full_protein_dataset_size_large600_slim \\
        --id-file     /home/student/thesis/full_protein_dataset_size_large600/split_manifest.json \\
        --workers 4

    # 3. then train against the new root
    python pipelines/07_train.py --model attention --data-root <dest> ...

Deviates from the standard --all/--filter flags (src/utils/filter.add_filter_args)
because the useful selections here are "every graph under this root" and "exactly
the IDs in this split manifest", not metadata predicates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rbf import RBF_RANGES, rbf_expand
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths

SIDECARS = ("split_manifest.json", "esp_stats.json")


def _infer_n_rbf(edge_type, width: int) -> int:
    """Recover the basis width the cache was built with from an edge_attr width."""
    # bond is [bond_order | rbf], every other relation is bare rbf.
    return width - 1 if edge_type[1] == "bond" else width


def _slim_in_memory(data, tolerance: float) -> tuple[bool, float]:
    """
    Replace every baked edge_attr on *data* with edge_dist (+ bond_order).

    Returns (converted, max_err). Raises ValueError if the recomputed distances
    do not reproduce the cached expansion within *tolerance*.
    """
    converted, max_err = False, 0.0
    for edge_type in data.edge_types:
        store = data[edge_type]
        attr  = store.get("edge_attr", None)
        if attr is None:
            continue  # already slim

        src_type, relation, dst_type = edge_type
        src_idx, dst_idx = store.edge_index[0], store.edge_index[1]
        dist = (data[dst_type].pos[dst_idx] - data[src_type].pos[src_idx]).norm(dim=-1)

        # Verify the recomputed distance reproduces the cached expansion before
        # throwing the cached one away.
        n_rbf        = _infer_n_rbf(edge_type, attr.shape[1])
        d_min, d_max = RBF_RANGES[relation]
        rebuilt      = rbf_expand(dist, n_rbf, d_min, d_max)
        original     = attr[:, 1:] if relation == "bond" else attr
        err          = float((rebuilt - original).abs().max())
        max_err      = max(max_err, err)
        if err > tolerance:
            raise ValueError(
                f"{edge_type} max|Δ|={err:.3e} exceeds tolerance {tolerance:g}"
            )

        # .clone() is load-bearing: a bare edge_attr[:, 0] is a view onto the
        # full edge_attr storage, and torch.save serialises the whole storage
        # behind a view — the file would not shrink at all.
        if relation == "bond":
            store.bond_order = attr[:, 0].clone()
        store.edge_dist = dist.contiguous()
        del store["edge_attr"]
        converted = True

    return converted, max_err


def convert_one(
    protein_id: str,
    source_root_str: str,
    dest_root_str: str | None,
    verify_only: bool,
    tolerance: float,
) -> dict:
    """
    Convert (or just verify) one protein's cached graph.

    status is one of: "converted", "verified", "skipped" (already slim, and
    already present at the destination), "missing", "exceeds_tolerance".
    """
    source_root = Path(source_root_str)
    src_graph   = ProteinPaths(protein_id, source_root).graph_path()

    if not src_graph.exists():
        return {"status": "missing", "protein_id": protein_id}

    size_before = src_graph.resolve().stat().st_size

    if dest_root_str is None:
        dst_graph = src_graph.resolve()     # --in-place
    else:
        dest_root = Path(dest_root_str)
        dst_graph = ProteinPaths(protein_id, dest_root).graph_path()
        if dst_graph.exists():
            return {"status": "skipped", "protein_id": protein_id,
                    "size_before": size_before,
                    "size_after": dst_graph.stat().st_size}

    data = torch.load(src_graph, weights_only=False)
    try:
        converted, max_err = _slim_in_memory(data, tolerance)
    except ValueError as e:
        return {"status": "exceeds_tolerance", "protein_id": protein_id,
                "detail": str(e), "size_before": size_before}

    if verify_only:
        return {"status": "verified", "protein_id": protein_id,
                "max_err": max_err, "size_before": size_before}

    data.edge_attr_slim = True

    # Atomic: write a tmp beside the real destination (same filesystem) then
    # rename over it. Readers never observe a partial graph, and an interrupted
    # run leaves nothing half-written.
    dst_graph.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_graph.with_suffix(".pt.tmp")
    torch.save(data, tmp_path)
    tmp_path.rename(dst_graph)

    # Per-protein metadata sidecar — copied, never symlinked (see module docstring).
    if dest_root_str is not None:
        src_meta = source_root / protein_id / f"{protein_id}_metadata.json"
        dst_meta = Path(dest_root_str) / protein_id / f"{protein_id}_metadata.json"
        if src_meta.exists() and not dst_meta.exists():
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_meta, dst_meta)   # copyfile: follows symlinks, copies bytes

    return {"status": "converted" if converted else "skipped",
            "protein_id": protein_id, "max_err": max_err,
            "size_before": size_before, "size_after": dst_graph.stat().st_size}


def _worker(
    protein_id: str, source_root_str: str, dest_root_str: str | None,
    verify_only: bool, tolerance: float,
) -> tuple[str, dict]:
    """Process-pool-safe wrapper. Returns (protein_id, result)."""
    return protein_id, convert_one(
        protein_id, source_root_str, dest_root_str, verify_only, tolerance,
    )


def _ids_from_file(path: Path) -> list[str]:
    """
    Read protein IDs from either a split_manifest.json (all three splits) or a
    plain newline-delimited ID list.
    """
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        splits   = manifest.get("splits", manifest)
        return [pid for key in ("train", "val", "test") for pid in splits.get(key, [])]
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cached graphs to the slim edge_dist format.",
    )
    parser.add_argument("--source-root", "--data-root", dest="source_root",
                        type=Path, default=None,
                        help="Root to read graphs from. Defaults to config.yaml data_root.")

    dest = parser.add_mutually_exclusive_group()
    dest.add_argument("--dest-root", type=Path, default=None,
                      help="Write converted graphs into this new root, leaving the "
                           "source cache untouched (recommended).")
    dest.add_argument("--in-place", action="store_true",
                      help="Overwrite the source graphs. With a symlinked source "
                           "root this rewrites the shared underlying files.")

    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--all", action="store_true",
                        help="Every protein directory with a cached graph under source-root.")
    select.add_argument("--id-file", type=Path,
                        help="split_manifest.json, or a newline-delimited ID list.")

    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4). Reads on this disk scale "
                             "with concurrency, so >1 is worth it.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only check that distances are recoverable; write nothing.")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Max allowed |Δ| between the cached RBF expansion and the "
                             "one rebuilt from recomputed distances (default 1e-4; "
                             "measured values are ~1.8e-06).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert only the first N selected proteins (for a pilot).")
    args = parser.parse_args()

    if not args.verify_only and args.dest_root is None and not args.in_place:
        parser.error(
            "a destination is required: pass --dest-root PATH (recommended, leaves "
            "the source cache untouched) or --in-place to overwrite the source. "
            "Use --verify-only to check recoverability without writing."
        )

    source_root = args.source_root or get_data_root()
    dest_root   = None if (args.in_place or args.verify_only) else args.dest_root
    log         = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    if args.all:
        protein_ids = sorted(
            d.name for d in Path(source_root).iterdir()
            if d.is_dir() or d.is_symlink()
        )
    else:
        protein_ids = _ids_from_file(args.id_file)
    if args.limit:
        protein_ids = protein_ids[: args.limit]

    if not protein_ids:
        print("No proteins selected. Exiting.")
        return

    mode = ("verify-only" if args.verify_only
            else "in-place" if args.in_place else "copy-to-dest")
    print(f"{mode}: {len(protein_ids)} proteins")
    print(f"  source: {source_root}")
    if args.verify_only:
        print(f"  dest:   (none — verifying only, nothing is written)")
    else:
        print(f"  dest:   {dest_root if dest_root else '(same as source — OVERWRITING)'}")
    sample = ProteinPaths(protein_ids[0], Path(source_root)).graph_path()
    print(f"  first source graph: {sample}")
    if sample.exists() and sample.resolve() != sample:
        print(f"    resolves to:      {sample.resolve()}")
    print(f"  workers={args.workers}  tolerance={args.tolerance:g}\n")
    log.info("slim_graph_edge_attr: mode=%s n=%d src=%s dest=%s",
             mode, len(protein_ids), source_root, dest_root)

    # Root-level sidecars, so the new root is trainable standalone.
    if dest_root is not None:
        dest_root.mkdir(parents=True, exist_ok=True)
        for name in SIDECARS:
            src = Path(source_root) / name
            dst = dest_root / name
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)
                print(f"  copied sidecar: {name}")
        print()

    arg_tuples = [
        (pid, str(source_root), str(dest_root) if dest_root else None,
         args.verify_only, args.tolerance)
        for pid in protein_ids
    ]
    if args.workers == 1:
        results = [_worker(*a) for a in arg_tuples]
    else:
        results = run_parallel(
            _worker, arg_tuples, n_workers=args.workers, label="slim",
        )

    counts: dict[str, int] = {}
    before = after = 0
    worst = 0.0
    failures: list[str] = []

    for protein_id, outcome in results:
        if isinstance(outcome, Exception):
            counts["error"] = counts.get("error", 0) + 1
            failures.append(f"{protein_id}: {outcome}")
            log.error("[%s] %s", protein_id, outcome)
            continue
        _, res = outcome if isinstance(outcome, tuple) else (protein_id, outcome)
        status = res["status"]
        counts[status] = counts.get(status, 0) + 1
        before += res.get("size_before", 0)
        after  += res.get("size_after", res.get("size_before", 0))
        worst   = max(worst, res.get("max_err", 0.0))
        if status in ("exceeds_tolerance", "missing"):
            failures.append(f"{protein_id}: {res.get('detail', status)}")

    print("\n── Summary ──")
    for status, n in sorted(counts.items()):
        print(f"  {status:<20} {n}")
    print(f"  max |Δ| observed     {worst:.3e}  (tolerance {args.tolerance:g})")
    if before:
        print(f"  source bytes         {before / 1e9:.2f} GB")
        if not args.verify_only:
            print(f"  dest bytes           {after / 1e9:.2f} GB"
                  f"   ({before / max(after, 1):.2f}x smaller, "
                  f"{(before - after) / 1e9:.2f} GB saved)")
    if failures:
        print(f"\n  {len(failures)} problem(s); first 10:")
        for line in failures[:10]:
            print(f"    {line}")
    if dest_root is not None and not failures:
        print(f"\nTrain against it with:\n  --data-root {dest_root}")

    log.info("slim_graph_edge_attr done: %s  max_err=%.3e  %.2f GB -> %.2f GB",
             counts, worst, before / 1e9, after / 1e9)


if __name__ == "__main__":
    main()
