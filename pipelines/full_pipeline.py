"""
pipeline/full_pipeline.py

Full end-to-end pipeline for the ProteinStructure ESP project.

Runs all pipeline steps sequentially for each selected protein fragment:
    1. Download AlphaFold structures (all fragments per UniProt ID)
    2. Run PDB2PQR (charge/radius assignment)
    3. Run APBS (electrostatic potential calculation)
    4. Generate PQR surface mesh via MSMS
    5. Sample ESP onto the PQR mesh (interpolated + Laplacian)
    6. Evaluate predictions (Pearson r, RMSE → metadata)

Each step is skipped if its output files already exist.
Failed steps are logged to the per-protein log file.
A single terminal line is printed when each protein completes or fails.
Pipeline status (pipeline_complete, pipeline_steps) is written to metadata.

Usage:
    python pipeline/full_pipeline.py --id-file data/protein_ids.txt
    python pipeline/full_pipeline.py --id-file data/protein_ids.txt --data-root /path/to/data
"""

import argparse
from pathlib import Path

from src.analysis.metrics import evaluate_protein
from src.electrostatics.run_apbs import process_apbs
from src.electrostatics.run_pdb2pqr import process_pdb2pqr
from src.structure.af_api import download_protein, find_downloaded_protein_ids, read_uniprot_ids
from src.surface.esp_mapping import sample_esp
from src.surface.mesh import build_mesh
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger, notify, timer
from src.utils.io import update_metadata
from src.utils.parallel import run_parallel
from src.utils.paths import ProteinPaths


# ── Per-protein pipeline ──────────────────────────────────────────────────────

def _run_protein(protein_id: str, data_root: Path, pipeline_log) -> dict:
    """
    Run all pipeline steps for a single protein fragment.
    All detailed logging goes to the per-protein log file.
    Returns a dict mapping step name to 'success', 'skipped', or 'failed'.
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = pipeline_log

    step_results = {}

    with timer() as t_total:

        # ── Step 2: PDB2PQR ───────────────────────────────────────────────────
        if p.pqr_path.exists():
            step_results["pdb2pqr"] = "skipped"
        elif not p.pdb_path.exists():
            plog.error("[%s] Step 2: PDB missing", protein_id)
            step_results["pdb2pqr"] = "failed"
        else:
            ok = process_pdb2pqr(protein_id, data_root)
            step_results["pdb2pqr"] = "success" if ok else "failed"

        # ── Step 3: APBS ─────────────────────────────────────────────────────
        if p.dx_path.exists():
            step_results["apbs"] = "skipped"
        elif not p.pqr_path.exists():
            plog.error("[%s] Step 3: PQR missing", protein_id)
            step_results["apbs"] = "failed"
        else:
            ok = process_apbs(protein_id, data_root)
            step_results["apbs"] = "success" if ok else "failed"

        # ── Step 4: PQR mesh generation ───────────────────────────────────────
        if p.pqr_mesh_path.exists():
            step_results["mesh_pqr"] = "skipped"
        elif not p.pqr_path.exists():
            plog.error("[%s] Step 4: PQR missing", protein_id)
            step_results["mesh_pqr"] = "failed"
        else:
            try:
                build_mesh(p.pqr_path, protein_id, data_root)
                step_results["mesh_pqr"] = "success"
            except Exception as e:
                plog.error("[%s] Step 4 PQR mesh failed: %s", protein_id, e)
                step_results["mesh_pqr"] = "failed"

        # ── Step 5: ESP sampling ──────────────────────────────────────────────
        if p.all_sampled_exist():
            step_results["esp_sampling"] = "skipped"
        elif not p.pqr_mesh_path.exists() or not p.dx_path.exists():
            plog.error("[%s] Step 5: missing PQR mesh or DX", protein_id)
            step_results["esp_sampling"] = "failed"
        else:
            ok = sample_esp(protein_id, data_root)
            step_results["esp_sampling"] = "success" if ok else "failed"

        # ── Step 6: Evaluate ──────────────────────────────────────────────────
        if p.is_evaluated():
            step_results["evaluate"] = "skipped"
        elif not p.all_sampled_exist():
            plog.error("[%s] Step 6: sampled files missing", protein_id)
            step_results["evaluate"] = "failed"
        else:
            try:
                evaluate_protein(protein_id, data_root, write_metadata=True)
                step_results["evaluate"] = "success"
            except Exception as e:
                plog.error("[%s] Step 6 failed: %s", protein_id, e)
                step_results["evaluate"] = "failed"

    try:
        update_metadata(protein_id, data_root=data_root, data={
            "time_total_sec": t_total.rounded,
        })
    except Exception:
        pass

    any_failed   = any(v == "failed" for v in step_results.values())
    status       = "failed" if any_failed else "complete"
    failed_steps = [k for k, v in step_results.items() if v == "failed"]
    detail       = f"{t_total.rounded}s" + (f"  failed: {', '.join(failed_steps)}" if failed_steps else "")

    notify(protein_id, status, detail)
    plog.info("[%s] Done in %.1f s  steps: %s", protein_id, t_total.seconds, step_results)

    return step_results


def _run_protein_worker(protein_id: str, data_root_str: str) -> tuple[str, dict]:
    """
    Process-pool-safe wrapper around _run_protein.

    Takes only picklable primitive arguments and creates its own logger
    so it can be submitted to ProcessPoolExecutor via run_parallel.

    Returns (protein_id, step_results).
    """
    from pathlib import Path
    from src.utils.config import get_config
    from src.utils.helpers import get_pipeline_logger

    data_root = Path(data_root_str)
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    steps     = _run_protein(protein_id, data_root, log)
    return protein_id, steps


# ── Summary ───────────────────────────────────────────────────────────────────

def _log_summary(all_results: dict, log) -> None:
    steps = ["pdb2pqr", "apbs", "mesh_pqr", "esp_sampling", "evaluate"]
    log.info("═" * 70)
    log.info("PIPELINE SUMMARY")
    log.info("═" * 70)
    for step in steps:
        counts = {"success": 0, "skipped": 0, "failed": 0}
        for step_results in all_results.values():
            outcome = step_results.get(step, "n/a")
            if outcome in counts:
                counts[outcome] += 1
        log.info("  %-20s  success=%d  skipped=%d  failed=%d",
                 step, counts["success"], counts["skipped"], counts["failed"])
    log.info("═" * 70)


# ── Public API ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full end-to-end ESP pipeline."
    )
    parser.add_argument("--id-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes for the per-protein pipeline "
             "(default: 1). Set >1 to process multiple proteins concurrently.",
    )
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log       = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    if not args.id_file.exists():
        log.error("ID file not found: %s", args.id_file)
        return

    uniprot_ids = read_uniprot_ids(args.id_file)
    if not uniprot_ids:
        log.warning("No UniProt IDs found in %s", args.id_file)
        return

    log.info("Starting pipeline for %d UniProt IDs", len(uniprot_ids))
    all_results = {}

    with timer() as t_run:
        for uniprot_id in uniprot_ids:

            # ── Step 1: Download all fragments ────────────────────────────────
            protein_ids = find_downloaded_protein_ids(uniprot_id, data_root)
            if protein_ids:
                log.info("[%s] Already downloaded: %s — skipping download",
                         uniprot_id, ", ".join(protein_ids))
            else:
                ok = download_protein(uniprot_id, data_root)
                if not ok:
                    notify(uniprot_id, "failed", "download")
                    log.error("[%s] Download failed — skipping all steps", uniprot_id)
                    continue
                protein_ids = find_downloaded_protein_ids(uniprot_id, data_root)
                if not protein_ids:
                    log.error("[%s] Could not resolve any protein_id after download", uniprot_id)
                    notify(uniprot_id, "failed", "protein_id resolution")
                    continue
                notify(uniprot_id, "complete", f"download ({len(protein_ids)} fragments)")

            # ── Steps 2-6: Run pipeline for each fragment ─────────────────────
            if args.workers == 1:
                for protein_id in protein_ids:
                    step_results = _run_protein(protein_id, data_root, log)
                    all_results[protein_id] = step_results
                    try:
                        update_metadata(protein_id, data_root=data_root, data={
                            "pipeline_steps"    : step_results,
                            "pipeline_complete" : not any(v == "failed" for v in step_results.values()),
                        })
                    except Exception:
                        pass
            else:
                parallel_results = run_parallel(
                    _run_protein_worker,
                    [(pid, str(data_root)) for pid in protein_ids],
                    n_workers=args.workers,
                    label=f"proteins (workers={args.workers})",
                )
                for protein_id, outcome in parallel_results:
                    if isinstance(outcome, Exception):
                        log.error("[%s] Worker exception: %s", protein_id, outcome)
                        all_results[protein_id] = {}
                        continue
                    _, step_results = outcome
                    all_results[protein_id] = step_results
                    try:
                        update_metadata(protein_id, data_root=data_root, data={
                            "pipeline_steps"    : step_results,
                            "pipeline_complete" : not any(v == "failed" for v in step_results.values()),
                        })
                    except Exception:
                        pass

    _log_summary(all_results, log)
    log.info("Total pipeline time: %.1f s", t_run.seconds)


if __name__ == "__main__":
    main()
