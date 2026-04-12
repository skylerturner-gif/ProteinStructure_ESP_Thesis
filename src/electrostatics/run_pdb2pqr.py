"""
src/electrostatics/run_pdb2pqr.py

Wrapper for PDB2PQR charge/radius assignment.

Takes the canonical .cif file, converts it to a temporary .pdb in memory,
runs PDB2PQR to assign partial charges and radii via the PARSE forcefield
at pH 7.0 using PROPKA, and produces:
    - a .pqr file (charges + radii per atom)
    - an APBS input .in file

The temporary .pdb is deleted immediately after PDB2PQR finishes.

Metadata fields added:
    n_heavy_atoms   — number of non-hydrogen atoms
    net_charge      — sum of partial charges across all atoms (elementary charge units)

Usage (from a script):
    from src.electrostatics.run_pdb2pqr import process_pdb2pqr
    process_pdb2pqr(protein_id="AF-Q16613-F1", data_root=Path("/data"))
"""

import re
import subprocess
import tempfile
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


# ── PQR column fixer ─────────────────────────────────────────────────────────

def fix_pqr_columns(pqr_path: Path, plog) -> int:
    """
    Repair merged coordinate/charge/radius columns in a .pqr file.

    PDB2PQR occasionally omits whitespace between adjacent negative numbers
    when coordinate magnitudes exceed 99.999 Å (e.g. '-19.098-100.028').
    APBS and downstream parsers both require whitespace-separated columns.

    Inserts a space before any '-' that immediately follows a digit or '.'.
    Only ATOM/HETATM lines are modified; all other lines are passed through.

    Returns:
        Number of lines that were corrected.
    """
    lines = pqr_path.read_text().splitlines(keepends=True)
    fixed_lines = []
    n_fixed = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            new_line = re.sub(r'(?<=[0-9.])-', ' -', line)
            if new_line != line:
                n_fixed += 1
            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)
    if n_fixed:
        pqr_path.write_text("".join(fixed_lines))
        plog.info("Fixed %d merged-column line(s) in %s", n_fixed, pqr_path.name)
    return n_fixed


# ── Core PDB2PQR runner ───────────────────────────────────────────────────────

def _run_pdb2pqr(p: ProteinPaths, tmp_pdb: Path, plog) -> bool:
    """
    Run PDB2PQR as a subprocess using executable and parameters from config.

    Args:
        p:       ProteinPaths for this protein.
        tmp_pdb: Path to a temporary PDB file (converted from .cif) used as
                 PDB2PQR input.  The caller is responsible for cleanup.

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
        str(tmp_pdb),
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

    Reads the canonical .cif structure file, converts it to a temporary .pdb
    in a TemporaryDirectory (deleted on completion), and runs PDB2PQR.

    Expects:
        <data_root>/<protein_id>/structure/<protein_id>.cif

    Produces:
        <data_root>/<protein_id>/structure/<protein_id>.pqr
        <data_root>/<protein_id>/structure/<protein_id>.in

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory

    Returns:
        True on success, False on failure.
    """
    import gemmi

    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.cif_path.exists():
        plog.error("CIF file not found: %s", p.cif_path)
        return False

    plog.info("── Running PDB2PQR ──")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdb = Path(tmpdir) / f"{protein_id}.pdb"
        try:
            structure = gemmi.read_structure(str(p.cif_path))
            structure.write_pdb(str(tmp_pdb))
            plog.info("CIF → temp PDB: %s", tmp_pdb.name)
        except Exception as e:
            plog.error("CIF → PDB conversion failed: %s", e)
            return False

        with timer() as t:
            ok = _run_pdb2pqr(p, tmp_pdb, plog)
        # tmp_pdb deleted when TemporaryDirectory exits here

    if not ok:
        return False

    fix_pqr_columns(p.pqr_path, plog)
    fix_apbs_input(p, plog)
    plog.info("PDB2PQR complete: %.2f s", t.seconds)

    n_heavy    = count_heavy_atoms(p.pqr_path)
    net_charge = compute_net_charge(p.pqr_path)
    plog.info("Heavy atoms: %d  Net charge: %.4f e", n_heavy, net_charge)

    update_metadata(protein_id, data_root=data_root, data={
        "n_heavy_atoms"    : n_heavy,
        "net_charge"       : net_charge,
        "time_pdb2pqr_sec" : t.rounded,
    })

    return True