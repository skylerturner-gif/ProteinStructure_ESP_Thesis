"""
src/electrostatics/run_pdb2pqr.py

Wrapper for PDB2PQR charge/radius assignment.

Takes a raw .pdb file, runs PDB2PQR to assign partial charges and radii
via the PARSE forcefield at pH 7.0 using PROPKA, and produces:
    - a .pqr file (charges + radii per atom)
    - a cleaned .pdb file
    - an APBS input .in file

Also counts heavy atoms from the .pqr output and updates the protein's
metadata JSON.

Metadata fields added:
    n_heavy_atoms   — number of non-hydrogen atoms
    net_charge      — sum of partial charges across all atoms (elementary charge units)

Usage (from a script):
    from src.electrostatics.run_pdb2pqr import process_pdb2pqr
    process_pdb2pqr(protein_id="AF-Q16613-F1", data_root=Path("/data"))
"""

import re
import subprocess
from pathlib import Path

from src.utils.config import get_config
from src.utils.helpers import get_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Heavy atom count ──────────────────────────────────────────────────────────

def count_heavy_atoms(pqr_path: Path) -> int:
    """
    Count non-hydrogen atoms in a .pqr file.
    Heavy atoms are ATOM/HETATM records whose atom name does not start with H.
    """
    count = 0
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            atom_name = parts[2]
            if not atom_name.startswith("H"):
                count += 1
    return count


def compute_net_charge(pqr_path: Path) -> float:
    """
    Compute the net charge of a protein by summing the charge column
    (column index 8) across all ATOM/HETATM records in a .pqr file.

    Args:
        pqr_path: path to the .pqr file

    Returns:
        Net charge in elementary charge units (e), rounded to 4 decimal places.
    """
    total = 0.0
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                total += float(parts[8])
            except ValueError:
                continue
    return round(total, 4)


# ── APBS input path fixer ─────────────────────────────────────────────────────

def fix_apbs_input(p: ProteinPaths, plog) -> None:
    """
    PDB2PQR writes bare filenames into the .in file.
    Replace them with correct absolute paths so APBS can find everything.

    Args:
        p:    ProteinPaths instance for this protein
        plog: per-protein logger
    """
    text = p.apbs_in_path.read_text()

    text = re.sub(
        r"(mol\s+pqr\s+)\S+",
        lambda m: m.group(1) + str(p.pqr_path),
        text,
    )
    text = re.sub(
        r"(write\s+pot\s+dx\s+)\S+",
        lambda m: m.group(1) + str(p.dx_stem),
        text,
    )

    p.apbs_in_path.write_text(text)
    plog.info("Fixed APBS input paths → %s", p.apbs_in_path.name)


# ── Core PDB2PQR runner ───────────────────────────────────────────────────────

def _run_pdb2pqr(p: ProteinPaths, plog) -> bool:
    """
    Run PDB2PQR as a subprocess using executable and parameters from config.

    Returns:
        True on success, False on failure (error is logged).
    """
    cfg  = get_config()
    exe  = cfg["executables"]["pdb2pqr"]
    elec = cfg["electrostatics"]

    cmd = [
        exe,
        "--ff",                     elec["forcefield"],
        "--titration-state-method", elec["ph_method"],
        "--with-ph",                str(elec["ph_value"]),
        "--apbs-input",             str(p.apbs_in_path),
        "--pdb-output",             str(p.pdb2pqr_path),
        str(p.pdb_path),
        str(p.pqr_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=p.data_root)
    if result.returncode != 0:
        plog.error("PDB2PQR failed:\n%s", result.stderr)
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def process_pdb2pqr(protein_id: str, data_root: Path) -> bool:
    """
    Run PDB2PQR for a protein and update its metadata JSON.

    Expects:
        <data_root>/<protein_id>/structure/<protein_id>.pdb

    Produces:
        <data_root>/<protein_id>/structure/<protein_id>.pqr
        <data_root>/<protein_id>/structure/<protein_id>_pdb2pqr.pdb
        <data_root>/<protein_id>/structure/<protein_id>.in

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory

    Returns:
        True on success, False on failure.
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.pdb_path.exists():
        plog.error("PDB file not found: %s", p.pdb_path)
        return False

    plog.info("── Running PDB2PQR ──")

    with timer() as t:
        ok = _run_pdb2pqr(p, plog)
    if not ok:
        return False

    plog.info("PDB2PQR complete: %.2f s", t.seconds)
    fix_apbs_input(p, plog)

    n_heavy = count_heavy_atoms(p.pqr_path)
    net_charge = compute_net_charge(p.pqr_path)
    plog.info("Heavy atoms: %d  Net charge: %.4f e", n_heavy, net_charge)

    update_metadata(protein_id, data_root=data_root, data={
        "n_heavy_atoms"    : n_heavy,
        "net_charge"       : net_charge,
        "time_pdb2pqr_sec" : t.rounded,
    })

    return True