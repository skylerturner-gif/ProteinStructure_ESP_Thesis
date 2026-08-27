"""
scripts/build_true_radial_graphs.py

Builds true-radial-kNN graphs (bonded pairs included in the radial edge
set, not excluded) for the phase F `no_bond`/`spatial_only` chemistry-
ablation rungs.

Writes to an ISOLATED output root — never touches the production graph
cache. `true_radial` is not additive like curvature/normal (a graph can't
have both the bond-filtered and true-kNN radial edge sets at once), so
building it in place would silently invalidate every other checkpoint's
graphs, the way the original --force rebuild almost did (see
.claude/EVAL_REPRODUCIBILITY_INVESTIGATION.md). Config.yaml itself is never
touched either — true_radial is forced in-process, per worker, via a
monkeypatched get_config().

Reads PQR/mesh/ESP inputs from the real (production) data_root, which are
untouched by this script, and writes only
<output_root>/<protein_id>/graph/<protein_id>_graph.pt — matching
ProteinPaths' layout exactly, so the result is a drop-in data_root for
`07_train.py --data-root <output_root>` with the existing
ProteinGraphDataset(rebuild=False) path, unmodified.

Protein set is read directly from the CURRENT split_manifest.json
(train + val + test) so the split is guaranteed identical to every other
checkpoint — this script never regenerates or touches the manifest.

Usage:
    python scripts/build_true_radial_graphs.py --workers 6
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_split_manifest
from src.utils.config import get_data_root
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths


def _verify_graph(path: Path) -> bool:
    """Quick ZIP-header check — catches partial writes without deserialising tensors."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return bool(zf.namelist())
    except Exception:
        return False


def _build_one_true_radial(
    protein_id: str,
    read_root_str: str,
    write_root_str: str,
    force: bool,
) -> str:
    """
    Worker: build a true-radial graph, reading PQR/mesh/ESP from read_root
    and writing only the resulting graph .pt to write_root. Runs in its own
    spawned process (see run_parallel) — the get_config() patch below is
    process-local and never touches the real config.yaml or leaks to other
    workers/the parent process.

    Returns "ok", "skip", or "fail".
    """
    from pathlib import Path as _Path

    import torch as _torch

    write_path = _Path(write_root_str) / protein_id / "graph" / f"{protein_id}_graph.pt"
    if write_path.exists() and not force:
        return "skip"

    import src.data.graph_builder as gb_mod
    import src.utils.config as config_mod

    # ProcessPoolExecutor reuses each worker process across many tasks (not
    # one process per protein), so this patch must be idempotent — applying
    # it twice in the same process (2nd+ task on a given worker) must not
    # break, and get_config() may already be a plain lambda from task 1
    # rather than the original lru_cache-decorated function.
    if not getattr(config_mod, "_true_radial_patched", False):
        real_cfg = config_mod.get_config()
        patched_cfg = dict(real_cfg)
        patched_cfg["features"] = {**real_cfg.get("features", {}), "true_radial": True}
        config_mod.get_config = lambda: patched_cfg
        gb_mod.get_config = lambda: patched_cfg
        config_mod._true_radial_patched = True

    from src.data.graph_builder import build_graph

    read_root = _Path(read_root_str)
    p = ProteinPaths(protein_id, read_root)
    missing = [f for f in [p.pqr_path, p.mesh_path, p.esp_path] if not f.exists()]
    if missing:
        return "fail"

    try:
        data = build_graph(protein_id, read_root)
        if not data.feature_spec.get("true_radial"):
            return "fail"  # patch didn't take — don't silently write a bond-filtered graph

        write_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = write_path.with_suffix(".pt.tmp")
        _torch.save(data, tmp_path)
        tmp_path.rename(write_path)  # atomic on Linux
        return "ok"
    except Exception:
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Isolated output root for true-radial graphs. "
             "Default: <data_root>_trueradial (sibling directory).",
    )
    parser.add_argument("--force", action="store_true",
                        help="Rebuild graphs that already exist in the output root.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    read_root  = get_data_root()
    write_root = args.output_root or read_root.parent / f"{read_root.name}_trueradial"
    write_root.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = load_split_manifest(read_root)
    all_ids = train_ids + val_ids + test_ids
    print(f"Building true-radial graphs for {len(all_ids)} proteins "
          f"(train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}) "
          f"— exactly matches current split_manifest.json")
    print(f"Reading inputs from:  {read_root}")
    print(f"Writing graphs to:    {write_root}")
    print(f"Workers: {args.workers}")

    n_ok = n_skip = n_fail = 0
    fail_ids: list[str] = []

    if args.workers == 1:
        for i, pid in enumerate(all_ids, 1):
            status = _build_one_true_radial(pid, str(read_root), str(write_root), args.force)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                fail_ids.append(pid)
            print(f"\r  graphs: {i}/{len(all_ids)}", end="", flush=True)
        print()
    else:
        results = run_parallel(
            _build_one_true_radial,
            [(pid, str(read_root), str(write_root), args.force) for pid in all_ids],
            n_workers=args.workers,
            label="true_radial_graphs",
        )
        for pid, outcome in results:
            if isinstance(outcome, Exception):
                n_fail += 1
                fail_ids.append(pid)
            elif outcome == "ok":
                n_ok += 1
            elif outcome == "skip":
                n_skip += 1
            else:
                n_fail += 1
                fail_ids.append(pid)

    print(f"Done — ok: {n_ok}  skipped: {n_skip}  failed: {n_fail}")
    if fail_ids:
        shown = fail_ids[:20]
        print(f"Failed proteins ({len(fail_ids)}): {shown}{' ...' if len(fail_ids) > 20 else ''}")

    print("Verifying written graphs...")
    bad = []
    for pid in all_ids:
        path = write_root / pid / "graph" / f"{pid}_graph.pt"
        if path.exists() and not _verify_graph(path):
            bad.append(pid)
    if bad:
        print(f"WARNING: {len(bad)} corrupted graph(s) detected: {bad}")
    else:
        n_present = sum((write_root / pid / "graph" / f"{pid}_graph.pt").exists() for pid in all_ids)
        print(f"Verification: all {n_present} graphs present in {write_root} are clean.")


if __name__ == "__main__":
    main()
