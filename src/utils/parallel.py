"""
src/utils/parallel.py

Thin helpers for parallelising per-protein pipeline work and launching
training as a subprocess.

Public API
----------
  run_parallel(fn, arg_tuples, n_workers, *, label, use_threads)
  launch_training(data_root, *, model, epochs, ...)

Design notes
------------
ProcessPoolExecutor requires worker functions to be top-level (picklable).
Closures, lambdas, and functions that accept logger objects as arguments
cannot be submitted directly.  Each worker defined in pipeline modules
must therefore:
  - be a module-level function
  - accept only primitive/picklable arguments (str, int, float, bool, Path)
  - create its own logger internally via get_pipeline_logger / get_config
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any, Callable


# ── Generic parallel runner ───────────────────────────────────────────────────

def run_parallel(
    fn: Callable,
    arg_tuples: list[tuple],
    n_workers: int,
    *,
    label: str = "",
    use_threads: bool = False,
) -> list[tuple[Any, Any]]:
    """
    Run ``fn(*args)`` for each args-tuple in arg_tuples using a worker pool.

    Args:
        fn:          Top-level picklable callable.  Must NOT accept a logger
                     argument — workers should create their own loggers.
        arg_tuples:  List of argument tuples to unpack when calling fn.
                     The first element of each tuple is used as the result key.
        n_workers:   Size of the worker pool.
        label:       Optional label printed in the live progress counter.
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor.
                     Pass True for I/O-bound work (downloads, subprocesses);
                     leave False (default) for CPU-bound work (MDAnalysis,
                     graph building, numerical sampling).

    Returns:
        List of ``(key, result)`` tuples in *completion order* (not
        submission order).  ``key`` is the first element of the arg tuple;
        ``result`` is the return value of fn, or an Exception if fn raised.

    Example::

        from src.utils.parallel import run_parallel
        from scripts.my_script import _worker  # top-level function

        results = run_parallel(
            _worker,
            [(pid, str(data_root), sample_frac) for pid in protein_ids],
            n_workers=8,
            label="graphs",
        )
        for key, result in results:
            print(key, result)
    """
    if not arg_tuples:
        return []

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results: list[tuple[Any, Any]] = []
    n = len(arg_tuples)
    done = 0

    with Executor(max_workers=n_workers) as pool:
        futures = {pool.submit(fn, *args): args[0] for args in arg_tuples}
        for future in as_completed(futures):
            key   = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as exc:
                result = exc
            results.append((key, result))
            if label:
                print(f"\r  {label}: {done}/{n}", end="", flush=True)

    if label:
        print()  # newline after final progress tick

    return results


# ── Training launcher ─────────────────────────────────────────────────────────

def launch_training(
    data_root: Path,
    *,
    model: str             = "distance",
    variant: str           = "interp",
    epochs: int            = 100,
    max_edges_per_batch: int = 500_000,
    hidden_dim: int        = 128,
    n_rbf: int             = 16,
    n_heads: int           = 4,
    n_bond_radial_rounds: int = 2,
    n_aq_rounds: int       = 3,
    n_qq_rounds: int       = 2,
    lr: float              = 3e-4,
    weight_decay: float    = 1e-4,
    pearson_weight: float  = 0.1,
    sample_frac: float     = 0.05,
    num_workers: int       = 4,
    resume: Path | None    = None,
    train_script: Path | None = None,
) -> int:
    """
    Launch ``scripts/08_train.py`` as a subprocess with the given settings.

    Training always runs on ``--all`` proteins in *data_root*; graph
    pre-building is expected to be complete before calling this.

    Args:
        data_root:   Root of the external data directory.
        train_script: Path to 08_train.py.  Defaults to
                      ``<project_root>/scripts/08_train.py``.
        (remaining): Mirror the CLI flags of 08_train.py.

    Returns:
        Subprocess exit code (0 = success).
    """
    if train_script is None:
        train_script = Path(__file__).parent.parent.parent / "scripts" / "08_train.py"

    cmd: list[str] = [
        sys.executable, str(train_script),
        "--all",
        "--data-root",            str(data_root),
        "--model",                model,
        "--variant",              variant,
        "--epochs",               str(epochs),
        "--max-edges-per-batch",  str(max_edges_per_batch),
        "--hidden-dim",           str(hidden_dim),
        "--n-rbf",                str(n_rbf),
        "--n-heads",              str(n_heads),
        "--n-bond-radial-rounds", str(n_bond_radial_rounds),
        "--n-aq-rounds",          str(n_aq_rounds),
        "--n-qq-rounds",          str(n_qq_rounds),
        "--lr",                   str(lr),
        "--weight-decay",         str(weight_decay),
        "--pearson-weight",       str(pearson_weight),
        "--sample-frac",          str(sample_frac),
        "--num-workers",          str(num_workers),
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]

    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode
