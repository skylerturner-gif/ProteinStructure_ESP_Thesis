"""
scripts/time_esp_prediction.py

Times the ML path to ESP-everywhere-on-the-mesh, for direct comparison
against the APBS PDE-solve time already recorded per protein
(``time_apbs_sec``, written by src/electrostatics/run_apbs.py).

Per protein, the timed region covers exactly the two steps that stand in for
the actual computation APBS's own timer covers (not data loading or
preprocessing that either method already did in an earlier pipeline stage):

  1. model forward pass  -> ESP at the query nodes (sparse, ~5% of the mesh)
  2. RBF reconstruction  -> ESP at every mesh vertex, same math and same
     default kernel as src/analysis/model_metrics.compute_full_mesh_metrics
     uses for the accuracy-side full-mesh numbers reported elsewhere
     (src/surface/esp_mapping.reconstruct_full_mesh / rbf_reconstruct)

No predicted values, meshes, or intermediate arrays are written to disk --
this only ever measures wall-clock time. The one thing it writes is
``<model>_esp_time`` (seconds) into each protein's own metadata JSON.

Why a process pool, not threads or a tmp-file relay
----------------------------------------------------
Profiling showed step 2 (RBFInterpolator's evaluation at every mesh vertex,
scipy's ``neighbors``-local mode) is 96-98% of the total time -- the model
forward pass and the KDTree query-to-vertex match are both single-digit
milliseconds by comparison. That evaluation is embarrassingly parallel
across output vertices (each is an independent local fit+solve), but
threading it made things *slower* (scipy's RBFInterpolator does not release
the GIL) -- measured 0.43x-0.7x, not a speedup. Separate processes, each
re-fitting its own cheap local RBFInterpolator (~2-7ms) and evaluating only
its slice of mesh vertices, gave a real, verified-identical-output speedup:
1.93x/3.53x/5.22x at 2/4/6 workers. That's what --interp-workers controls.
A tmp-file relay would add disk I/O for no benefit -- the in-memory,
pickled-over-a-pipe handoff a process pool already does is strictly less
work, and nothing is ever written to disk either way.

The pool is created once (spawn context, so it never inherits this
process's CUDA context) and reused across every protein -- not recreated
per protein -- so pool-startup cost is paid once, not 8,511 times.

Run one instance per model, each pinned to its own GPU. With 8 CPU cores
total and both models running at once, 4 interpolation workers per process
(the default) splits the machine evenly between them:

    CUDA_VISIBLE_DEVICES=0 python -u scripts/time_esp_prediction.py \\
        --model attention --checkpoint-dir checkpoints/full_dataset_slim/attention_aa4_aq2_qq16 &
    CUDA_VISIBLE_DEVICES=1 python -u scripts/time_esp_prediction.py \\
        --model distance  --checkpoint-dir checkpoints/full_dataset_slim/distance_aa8_aq2_qq24 &

Running only one model at a time, raise --interp-workers (e.g. 6-8) for a
faster single-model pass.

Safe to run both against the same data_root at once: update_metadata() takes
a per-protein filelock, and the two runs write different fields
(attention_esp_time / distance_esp_time) to begin with.

Resumable: proteins that already have the target field are skipped unless
--force is passed, so an interrupted run (or extending to more proteins
later) doesn't re-time anything.

Usage:
    python scripts/time_esp_prediction.py --model attention \\
        --checkpoint-dir checkpoints/full_dataset_slim/attention_aa4_aq2_qq16
    python scripts/time_esp_prediction.py ... --limit 20            # smoke test
    python scripts/time_esp_prediction.py ... --force                # re-time everything
    python scripts/time_esp_prediction.py ... --interp-workers 8      # single-model run
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree

from src.analysis.embedding_analysis import load_model_frozen
from src.surface.esp_mapping import _rbf_epsilon, reconstruct_full_mesh
from src.utils.config import get_data_root
from src.utils.filter import get_protein_ids
from src.utils.io import load_metadata, update_metadata
from src.utils.paths import ProteinPaths


def _ids_from_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _eval_chunk(args: tuple) -> np.ndarray:
    """Pool worker: re-fit the (cheap) local RBF model and evaluate it at
    one chunk of mesh vertices. Runs in a separate process -- no GIL
    contention with the other workers, and no CUDA/torch involved at all.
    """
    known_verts, known_vals, eps, kernel, neighbors, chunk = args
    rbf = RBFInterpolator(known_verts, known_vals, kernel=kernel, epsilon=eps, neighbors=neighbors)
    return rbf(chunk)


def _reconstruct_full_mesh_parallel(
    query_pos: np.ndarray,
    query_esp: np.ndarray,
    mesh_verts: np.ndarray,
    method: str,
    pool: ProcessPoolExecutor | None,
    n_workers: int,
    neighbors: int = 50,
) -> None:
    """Same reconstruction as src/surface/esp_mapping.reconstruct_full_mesh,
    parallelized across processes for the RBF-kernel methods. "nearest" has
    no such bottleneck (1-NN via a GIL-releasing KDTree query, already fast)
    so it's left to the original single-process implementation. Return value
    is discarded by the caller -- this function's cost is what's measured,
    not its output.
    """
    if method == "nearest" or pool is None or n_workers <= 1:
        reconstruct_full_mesh(query_pos, query_esp, mesh_verts, method=method)
        return

    tree = KDTree(mesh_verts)
    _, sample_idx = tree.query(query_pos, k=1, workers=-1)
    known_verts = mesh_verts[sample_idx.astype(np.int64)].astype(np.float64)
    eps = _rbf_epsilon(known_verts)

    chunks = np.array_split(mesh_verts.astype(np.float64), n_workers)
    args = [
        (known_verts, query_esp.astype(np.float64), eps, method, neighbors, chunk)
        for chunk in chunks
    ]
    list(pool.map(_eval_chunk, args))


def _time_one(model, device, esp_mean, esp_std, graph_path, mesh_verts,
              reconstruction, pool, n_interp_workers) -> float:
    """Load one protein's cached graph fresh, run forward + full-mesh
    reconstruction, and return elapsed seconds. Always loads a fresh graph
    object -- never reuse one across calls, since materialize_edge_attr()
    mutates the graph in place on first forward() and would silently skip
    the RBF-expansion cost on a second call, making the timing optimistic.
    """
    data = torch.load(graph_path, weights_only=False)
    data = data.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(data)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    pred_raw  = (pred.float() * esp_std + esp_mean).cpu().numpy()
    query_pos = data["query"].pos.cpu().numpy()
    _reconstruct_full_mesh_parallel(
        query_pos, pred_raw, mesh_verts, reconstruction, pool, n_interp_workers,
    )

    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time model inference + full-mesh reconstruction per protein, "
                     "for comparison against APBS's time_apbs_sec."
    )
    parser.add_argument("--model", required=True, choices=["attention", "distance"],
                         help="Names the metadata field written: <model>_esp_time.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None,
                         help="Override data_root from config.yaml.")
    parser.add_argument("--id-file", type=Path, default=None,
                         help="Newline-delimited protein ID list. Default: every protein "
                              "in data_root with a cached graph and mesh.")
    parser.add_argument("--reconstruction", default="multiquadric",
                         choices=["multiquadric", "gaussian", "nearest"],
                         help="RBF kernel for full-mesh reconstruction (default: multiquadric, "
                              "matching src/analysis/model_metrics.compute_full_mesh_metrics).")
    parser.add_argument("--interp-workers", type=int, default=4,
                         help="Process-pool workers for the full-mesh RBF evaluation "
                              "(default 4 -- tuned for two model runs sharing 8 cores; "
                              "raise to 6-8 if running only one model at a time). "
                              "1 disables the pool.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first N proteins.")
    parser.add_argument("--force", action="store_true",
                         help="Re-time proteins that already have <model>_esp_time recorded.")
    args = parser.parse_args()

    data_root  = args.data_root or get_data_root()
    field_name = f"{args.model}_esp_time"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   field: {field_name}   reconstruction: {args.reconstruction}"
          f"   interp_workers: {args.interp_workers}")

    protein_ids = _ids_from_file(args.id_file) if args.id_file else get_protein_ids(data_root, select_all=True)
    if args.limit:
        protein_ids = protein_ids[: args.limit]
    print(f"{len(protein_ids)} candidate proteins")

    model, ckpt = load_model_frozen(args.checkpoint_dir, device)
    esp_mean = float(ckpt["esp_mean"])
    esp_std  = float(ckpt["esp_std"])
    print(f"Loaded {ckpt['model_name']} from {args.checkpoint_dir.name}")

    pool = None
    if args.interp_workers > 1:
        # spawn: never inherits this process's already-initialized CUDA context.
        ctx  = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=args.interp_workers, mp_context=ctx)

    try:
        # Warmup: a throwaway forward+reconstruct pass on its own graph object
        # (never reused below) so the first real protein doesn't absorb CUDA
        # kernel-compilation (or, here, pool spin-up) overhead into its
        # measurement.
        for pid in protein_ids:
            p = ProteinPaths(pid, data_root)
            if p.graph_path().exists() and p.mesh_path.exists():
                mesh_verts = np.load(p.mesh_path)["verts"]
                _time_one(model, device, esp_mean, esp_std, p.graph_path(), mesh_verts,
                          args.reconstruction, pool, args.interp_workers)
                print(f"Warmup pass complete ({pid}).")
                break

        n_done = n_skip = n_missing = 0
        for i, pid in enumerate(protein_ids, 1):
            p = ProteinPaths(pid, data_root)
            graph_path = p.graph_path()
            if not graph_path.exists() or not p.mesh_path.exists():
                n_missing += 1
                continue

            if not args.force:
                meta = load_metadata(pid, data_root=data_root)
                if field_name in meta:
                    n_skip += 1
                    continue

            mesh_verts = np.load(p.mesh_path)["verts"]
            elapsed = _time_one(model, device, esp_mean, esp_std, graph_path, mesh_verts,
                                args.reconstruction, pool, args.interp_workers)

            update_metadata(pid, data_root=data_root, data={field_name: round(elapsed, 5)})
            n_done += 1

            if i % 50 == 0 or i == len(protein_ids):
                print(f"\r  {i}/{len(protein_ids)}  (timed={n_done} skipped={n_skip} missing={n_missing})",
                      end="", flush=True)

        print(f"\nDone — timed: {n_done}  skipped (already had {field_name}): {n_skip}  missing: {n_missing}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
