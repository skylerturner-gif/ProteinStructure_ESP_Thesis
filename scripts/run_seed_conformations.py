"""
scripts/run_seed_conformations.py

Seed-conformation pilot, phase B2 (see the approved plan for the full
"Seed-Conformation Pilot" section): for each of 5 best- and 5
worst-predicted pilot proteins (ranked by mean Pearson r across both
full-dataset champions, /home/student/thesis/outputs/seed_pilot_candidates.csv), generate 5
AlphaFold conformations via local ColabFold (--num-seeds 5, one AF2
model), place each as a synthetic protein_id in the normal data_root
layout, and run the existing, unmodified per-protein data-generation unit
(pipelines/data_gen_pipeline.py's _run_protein: PDB2PQR -> APBS -> mesh ->
ESP sampling -> evaluate) on each.

Deliberately does NOT re-download from the AlphaFold DB API (there is no
multi-seed endpoint there -- see the plan) and does NOT go through
data_gen_pipeline.py's own main()/CLI (which resolves UniProt IDs via the
AF API) -- it calls the same underlying per-protein function that CLI
uses, since our synthetic IDs already have a CIF + metadata.json in place
before this ever runs.

Traceability: every seed's exact seed index is read back from ColabFold's
own output filename (not assumed sequential), and each synthetic
protein's metadata.json records parent_protein_id, seed_index, and the
ColabFold version/commit/config used to produce it. ColabFold's own
config.json and per-seed scores json (full per-residue pLDDT + PAE
matrix) are copied alongside the structure for a complete audit trail.

Resilience: a small fraction of sequences trigger a reproducible NaN bug
in the ColabFold/JAX stack (all pLDDT/pTM come back NaN from recycle 0 --
confirmed not fixable via --disable-unified-memory, --compile-mode fast,
or JAX_DEFAULT_MATMUL_PRECISION=float32 on this environment). Each
candidate's ColabFold output is checked for NaN before being accepted; a
candidate that fails ColabFold entirely (non-zero exit) or produces NaN
predictions is recorded in /home/student/thesis/outputs/seed_pilot_failures.csv and skipped in
favor of the next-ranked candidate from /home/student/thesis/outputs/seed_pilot_candidates.csv
(15 deep per group) -- one bad sequence no longer aborts the whole pilot.

Run in the `protein_esp` conda environment (gemmi, PDB2PQR/APBS/MSMS are
here; ColabFold itself runs via its own self-contained pixi environment,
invoked by absolute binary path regardless of which env this script runs
in).

Usage:
    conda activate protein_esp
    python -u scripts/run_seed_conformations.py
    python -u scripts/run_seed_conformations.py --limit 1   # smoke test: 1 protein per group
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_gen_pipeline import _run_protein
from src.structure.seed_conformations import pdb_to_cif, sequence_from_pqr, write_fasta
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger
from src.utils.io import create_metadata, load_metadata
from src.utils.paths import ProteinPaths

COLABFOLD_BIN = Path("/home/student/thesis/localcolabfold/.pixi/envs/default/bin/colabfold_batch")
NUM_SEEDS = 5
N_PER_GROUP = 5
RANDOM_SEED_BASE = 0
NUM_RECYCLE = 3
GPU_DEVICE = "1"  # pinned to the second A100, keeps clear of anything on GPU 0

SEED_RUN_ROOT = Path("/home/student/thesis/outputs/seed_conformations")
CANDIDATES_CSV = Path("/home/student/thesis/outputs/seed_pilot_candidates.csv")
PILOT_CSV = Path("/home/student/thesis/outputs/seed_pilot_proteins.csv")
MANIFEST_CSV = Path("/home/student/thesis/outputs/seed_conformation_manifest.csv")
FAILURES_CSV = Path("/home/student/thesis/outputs/seed_pilot_failures.csv")
IDS_FILE = Path("/home/student/thesis/outputs/seed_conformation_ids.txt")

PDB_RE = re.compile(r"_unrelaxed_rank_(\d+)_alphafold2_ptm_model_(\d+)_seed_(\d+)\.pdb$")


class ColabFoldNaNError(RuntimeError):
    """Raised when ColabFold completes but produces NaN pLDDT/pTM -- a known,
    reproducible JAX/bfloat16 numerical bug for certain sequences on this
    environment (see module docstring). Treated the same as a hard failure:
    the whole protein is abandoned in favor of the next-ranked candidate."""


def _is_nan(x) -> bool:
    """math.isnan, but tolerant of None/non-numeric (treated as invalid, not an error)."""
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _scores_are_valid(scores: dict) -> bool:
    """True if a ColabFold scores dict has no NaN/None in pLDDT or pTM, and
    pLDDT isn't degenerately all-zero (a second, distinct failure mode seen
    on this environment for very long sequences: ColabFold exits 0 with a
    structurally-empty PDB but a real-looking, all-zero scores dict -- see
    _pdb_has_real_structure, which is the more direct/reliable check)."""
    plddt = scores.get("plddt", [])
    if not plddt or any(_is_nan(x) for x in plddt):
        return False
    if all(float(x) == 0.0 for x in plddt):
        return False
    ptm = scores.get("ptm")
    if ptm is not None and _is_nan(ptm):
        return False
    return True


def _pdb_has_real_structure(pdb_path: Path, min_atoms: int = 10) -> bool:
    """True if a PDB file has actual ATOM records with non-degenerate
    coordinates. Direct structural check, independent of the scores JSON --
    catches the empty-PDB-with-plausible-looking-scores failure mode seen
    for very long sequences on this environment (confirmed: ColabFold exits
    0, writes a 4-line PDB with zero ATOM records, and a scores.json with
    an all-zero pLDDT array that would otherwise look superficially valid).
    """
    coords = set()
    n_atoms = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                n_atoms += 1
                coords.add(line[30:54])  # x/y/z columns, as text (cheap distinctness check)
    return n_atoms >= min_atoms and len(coords) >= min_atoms // 2


def _run_colabfold(parent_id: str, sequence: str, work_dir: Path, log) -> Path:
    """Write a FASTA and run colabfold_batch for one protein. Returns its output dir.

    Raises RuntimeError on a hard ColabFold failure, or ColabFoldNaNError if
    it exits cleanly but every seed's predictions are NaN.
    """
    fasta_path = work_dir / "fasta" / f"{parent_id}.fasta"
    write_fasta(parent_id, sequence, fasta_path)

    out_dir = work_dir / "colabfold_raw" / parent_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COLABFOLD_BIN),
        "--num-models", "1",
        "--num-seeds", str(NUM_SEEDS),
        "--random-seed", str(RANDOM_SEED_BASE),
        "--num-recycle", str(NUM_RECYCLE),
        str(fasta_path), str(out_dir),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_DEVICE}
    log.info("[%s] Running ColabFold: %s", parent_id, " ".join(cmd))
    # Inherit stdout/stderr (not capture_output) so ColabFold's own progress
    # lines stream live into this script's log -- capturing silently until
    # exit hid all progress for multi-minute predictions during testing.
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        # ColabFold's own PAE-plotting code crashes (unhandled) when scores
        # are NaN, so a NaN-prediction run also shows up here as a non-zero
        # exit -- check for the NaN signature among whatever scores files
        # did get written before deciding this is a plain hard failure.
        scores_files = sorted(out_dir.glob("*_scores_rank_*.json"))
        if scores_files and not all(
            _scores_are_valid(json.loads(f.read_text())) for f in scores_files
        ):
            log.error("[%s] ColabFold produced NaN predictions (exit %d).", parent_id, result.returncode)
            raise ColabFoldNaNError(f"NaN predictions for {parent_id}")
        log.error("[%s] ColabFold failed (exit %d) -- see stdout above.", parent_id, result.returncode)
        raise RuntimeError(f"ColabFold failed for {parent_id} (exit {result.returncode})")

    if not _candidate_output_is_valid(out_dir):
        raise ColabFoldNaNError(f"NaN or empty-structure predictions for {parent_id}")
    return out_dir


def _candidate_output_is_valid(out_dir: Path) -> bool:
    """All of a candidate's scores AND PDBs must be valid -- checks both the
    NaN/degenerate-scores signature and the direct empty-structure signature
    (two distinct failure modes observed on this environment, see module
    docstring). Whole-protein pass/fail, matching the observed pattern that
    a bad candidate fails all 5 seeds together, not just some."""
    scores_files = sorted(out_dir.glob("*_scores_rank_*.json"))
    pdb_files = sorted(out_dir.glob("*_unrelaxed_rank_*.pdb"))
    if not scores_files or not pdb_files:
        return False
    if not all(_scores_are_valid(json.loads(f.read_text())) for f in scores_files):
        return False
    if not all(_pdb_has_real_structure(f) for f in pdb_files):
        return False
    return True


def _place_seed(
    parent_id: str, parent_meta: dict, seed_pdb: Path, out_dir: Path,
    data_root: Path, pilot_group: str, log,
) -> dict:
    """Convert one ColabFold seed PDB into a synthetic protein_id and bootstrap its metadata."""
    m = PDB_RE.search(seed_pdb.name)
    if not m:
        raise ValueError(f"Unrecognized ColabFold output filename: {seed_pdb.name}")
    rank, model, seed_index = m.groups()

    scores_path = out_dir / seed_pdb.name.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json")
    config_path = out_dir / "config.json"
    scores = json.loads(scores_path.read_text())
    config = json.loads(config_path.read_text())

    synthetic_id = f"{parent_id}_seed{seed_index}"
    p = ProteinPaths(synthetic_id, data_root)
    p.ensure_dirs()

    pdb_to_cif(seed_pdb, p.cif_path)
    shutil.copy(scores_path, p.structure_dir / f"{synthetic_id}_colabfold_scores.json")
    shutil.copy(config_path, p.structure_dir / f"{synthetic_id}_colabfold_config.json")

    plddt_list = scores.get("plddt", [])
    plddt_mean = float(sum(plddt_list) / len(plddt_list)) if plddt_list else None

    metadata = {
        "uniprot_id":        parent_meta.get("uniprot_id", ""),
        "fragment":          1,
        "protein_name":      parent_meta.get("protein_name", ""),
        "organism":          parent_meta.get("organism", ""),
        "sequence_length":   len(plddt_list),
        "af_model_version":  f"colabfold_local_v{config.get('version', '?')}",
        "plddt_mean":        round(plddt_mean, 4) if plddt_mean is not None else None,
        "plddt_median":      None,
        "plddt_per_residue": plddt_list,
        # Provenance -- not part of af_api.py's normal schema, seed-pilot only.
        "parent_protein_id": parent_id,
        "pilot_group":       pilot_group,
        "seed_index":        int(seed_index),
        "colabfold_rank":    int(rank),
        "colabfold_version": config.get("version"),
        "colabfold_commit":  config.get("commit"),
        "colabfold_random_seed_base": config.get("random_seed"),
        "ptm":               scores.get("ptm"),
        "max_pae":           scores.get("max_pae"),
    }
    try:
        create_metadata(synthetic_id, data=metadata, data_root=data_root)
    except FileExistsError:
        log.warning("[%s] Metadata already exists -- reusing existing.", synthetic_id)

    log.info("[%s] Placed from %s (seed_index=%s, pLDDT=%.1f)", synthetic_id, parent_id, seed_index, plddt_mean or -1)
    return {
        "protein_id": synthetic_id, "parent_protein_id": parent_id, "pilot_group": pilot_group,
        "seed_index": int(seed_index), "colabfold_rank": int(rank),
        "plddt_mean_seed": plddt_mean, "ptm": scores.get("ptm"),
    }


def _process_one_candidate(parent_id: str, pilot_group: str, data_root: Path, force: bool, log) -> list[dict]:
    """Run ColabFold + data-gen for one candidate. Returns manifest rows (one per seed).

    Raises RuntimeError / ColabFoldNaNError if the candidate should be
    abandoned in favor of the next-ranked one -- callers catch these.
    """
    parent_paths = ProteinPaths(parent_id, data_root)
    parent_meta = load_metadata(parent_id, data_root)
    sequence = sequence_from_pqr(parent_paths.pqr_path)
    print(f"  sequence: {len(sequence)} aa")

    out_dir = SEED_RUN_ROOT / "colabfold_raw" / parent_id
    existing_pdbs = sorted(out_dir.glob("*_unrelaxed_rank_*.pdb")) if out_dir.exists() else []
    cached_valid = len(existing_pdbs) >= NUM_SEEDS and _candidate_output_is_valid(out_dir)
    if cached_valid and not force:
        print(f"  ColabFold output already present ({len(existing_pdbs)} structures, validated) -- skipping prediction.")
    else:
        if existing_pdbs and not cached_valid:
            print(f"  Cached ColabFold output exists but is invalid (NaN, empty structure, or incomplete) -- re-running.")
        out_dir = _run_colabfold(parent_id, sequence, SEED_RUN_ROOT, log)
    seed_pdbs = sorted((SEED_RUN_ROOT / "colabfold_raw" / parent_id).glob("*_unrelaxed_rank_*.pdb"))
    print(f"  {len(seed_pdbs)} seed structures produced")

    rows = []
    for seed_pdb in seed_pdbs:
        info = _place_seed(parent_id, parent_meta, seed_pdb, out_dir, data_root, pilot_group, log)

        synthetic_id = info["protein_id"]
        print(f"  [{synthetic_id}] running data-gen pipeline (PDB2PQR -> APBS -> mesh -> ESP)...")
        try:
            steps = _run_protein(synthetic_id, data_root, log)
            info["data_gen_status"] = "complete" if all(v != "failed" for v in steps.values()) else "failed"
            info["data_gen_steps"] = json.dumps(steps)
        except Exception as e:
            log.error("[%s] data-gen raised: %s", synthetic_id, e)
            info["data_gen_status"] = "failed"
            info["data_gen_steps"] = json.dumps({"exception": str(e)})
        print(f"    {info['data_gen_status']}: {info['data_gen_steps']}")
        rows.append(info)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5 AlphaFold seed conformations per pilot protein via "
                     "ColabFold and run the existing data-gen pipeline on each."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                         help="Smoke test: only attempt the first N candidates per group.")
    parser.add_argument("--force", action="store_true", help="Re-run ColabFold even if output exists.")
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    if not COLABFOLD_BIN.exists():
        raise FileNotFoundError(f"colabfold_batch not found at {COLABFOLD_BIN} -- run Phase B0 install first.")

    candidates = pd.read_csv(CANDIDATES_CSV).sort_values(["pilot_group", "candidate_rank"])
    SEED_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    failure_rows: list[dict] = []
    used_parents: list[dict] = []

    for pilot_group, group_df in candidates.groupby("pilot_group"):
        pool = list(group_df.itertuples())
        if args.limit is not None:
            pool = pool[: args.limit]
        n_target = args.limit if args.limit is not None else N_PER_GROUP
        n_secured = 0

        for cand in pool:
            if n_secured >= n_target:
                break
            parent_id = cand.protein_id
            print(f"\n=== {parent_id} ({pilot_group}, candidate_rank={cand.candidate_rank}) ===")
            try:
                rows = _process_one_candidate(parent_id, pilot_group, data_root, args.force, log)
            except (RuntimeError, Exception) as e:
                reason = "nan_predictions" if isinstance(e, ColabFoldNaNError) else "colabfold_error"
                print(f"  SKIPPING {parent_id}: {reason} ({e})")
                log.warning("[%s] Skipping candidate, reason=%s: %s", parent_id, reason, e)
                failure_rows.append({
                    "protein_id": parent_id, "pilot_group": pilot_group,
                    "candidate_rank": cand.candidate_rank, "reason": reason, "error": str(e),
                })
                continue

            manifest_rows.extend(rows)
            used_parents.append({
                "protein_id": parent_id, "pilot_group": pilot_group, "candidate_rank": cand.candidate_rank,
            })
            n_secured += 1

        if n_secured < n_target:
            print(f"\nWARNING: only secured {n_secured}/{n_target} '{pilot_group}' proteins "
                  f"-- candidate pool exhausted (15 deep). Widen /home/student/thesis/outputs/seed_pilot_candidates.csv if needed.")
            log.warning("Only secured %d/%d '%s' proteins", n_secured, n_target, pilot_group)

    pd.DataFrame(used_parents).to_csv(PILOT_CSV, index=False)
    print(f"\nWrote {len(used_parents)} used pilot proteins -> {PILOT_CSV}")

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(FAILURES_CSV, index=False)
        print(f"Wrote {len(failure_rows)} skipped candidates -> {FAILURES_CSV}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    print(f"Wrote {len(manifest_df)} rows -> {MANIFEST_CSV}")

    IDS_FILE.write_text("\n".join(manifest_df["protein_id"]) + "\n")
    print(f"Wrote {len(manifest_df)} synthetic protein IDs -> {IDS_FILE}")

    n_ok = (manifest_df["data_gen_status"] == "complete").sum()
    print(f"\nData-gen complete: {n_ok}/{len(manifest_df)}")
    log.info("run_seed_conformations complete: %d/%d data-gen ok, %d candidates skipped",
              n_ok, len(manifest_df), len(failure_rows))


if __name__ == "__main__":
    main()
