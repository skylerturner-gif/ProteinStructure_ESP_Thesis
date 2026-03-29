"""
src/electrostatics/run_apbs.py

Wrapper for APBS electrostatic potential calculation.

Takes the APBS .in file produced by PDB2PQR and runs APBS to compute
the electrostatic potential on a grid, producing a .dx file (the ESP
ground truth used by all downstream surface sampling).

Updates the protein's metadata JSON with the .dx file path.

Metadata fields added:
    esp_dx_path — path to the .dx ESP grid file

Usage (from a script):
    from src.electrostatics.run_apbs import process_apbs
    process_apbs(protein_id="AF-Q16613-F1", data_root=Path("/data"))
"""

import subprocess
from pathlib import Path

from src.utils.config import get_config
from src.utils.helpers import get_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Core APBS runner ──────────────────────────────────────────────────────────

def _run_apbs(p: ProteinPaths, plog) -> bool:
    """
    Run APBS as a subprocess using the executable path from config.

    Returns:
        True on success, False on failure (error is logged).
    """
    exe    = get_config()["executables"]["apbs"]
    cmd    = [exe, str(p.apbs_in_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=p.data_root)
    if result.returncode != 0:
        plog.error("APBS failed:\n%s", result.stderr)
        return False
    plog.info("APBS stdout:\n%s", result.stdout)
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def process_apbs(protein_id: str, data_root: Path) -> bool:
    """
    Run APBS for a protein and update its metadata JSON.

    Expects:
        <data_root>/<protein_id>/structure/<protein_id>.in

    Produces:
        <data_root>/<protein_id>/electrostatics/<protein_id>.dx

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory

    Returns:
        True on success, False on failure.
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.apbs_in_path.exists():
        plog.error("APBS input file not found: %s", p.apbs_in_path)
        plog.error("Run process_pdb2pqr first.")
        return False

    plog.info("── Running APBS ──")

    with timer() as t:
        ok = _run_apbs(p, plog)
    if not ok:
        return False

    plog.info("APBS complete: %.2f s", t.seconds)

    if not p.dx_path.exists():
        plog.error("APBS ran but .dx file not found at: %s", p.dx_path)
        return False

    update_metadata(protein_id, data_root=data_root, data={
        "time_apbs_sec": t.rounded,
    })

    plog.info("ESP grid saved → %s", p.dx_path.name)
    return True