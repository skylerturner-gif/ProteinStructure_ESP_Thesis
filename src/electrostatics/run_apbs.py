"""
src/electrostatics/run_apbs.py

Wrapper for APBS electrostatic potential calculation.

Takes the APBS .in file produced by PDB2PQR and runs APBS to compute
the electrostatic potential on a grid.

By default (keep_dx=False) the .dx output is written to a TemporaryDirectory,
read into memory as numpy arrays, and immediately deleted.  The caller
receives (axes, grid) and can pass them directly to sample_esp, avoiding
any permanent .dx file on disk.

When keep_dx=True the .dx is written to the permanent electrostatics/
directory and the same (axes, grid) tuple is returned so the caller still
has the data in memory.

Metadata fields added:
    time_apbs_sec — wall-clock time for the APBS run

Usage (from a script):
    from src.electrostatics.run_apbs import process_apbs
    result = process_apbs(protein_id="AF-Q16613-F1", data_root=Path("/data"))
    if result is None:
        ...  # APBS failed
    axes, grid = result
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.surface.esp_mapping import read_dx
from src.utils.config import get_config
from src.utils.helpers import get_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Core APBS runners ────────────────────────────────────────────────────────

def _run_apbs_to_dir(p: ProteinPaths, out_dir: Path, plog) -> bool:
    """
    Run APBS writing output to *out_dir* instead of p.electrostatics_dir.

    A patched copy of the .in file is written to *out_dir* with the
    ``write pot dx`` line redirected to a file inside *out_dir*.
    APBS is run with cwd=out_dir so that io.mc also lands there.

    Returns:
        True on success, False on failure.
    """
    tmp_dx_stem = str(out_dir / p.protein_id)
    patched_in  = re.sub(
        r"(write\s+pot\s+dx\s+)\S+",
        lambda m: m.group(1) + tmp_dx_stem,
        p.apbs_in_path.read_text(),
    )
    tmp_in_path = out_dir / p.apbs_in_path.name
    tmp_in_path.write_text(patched_in)

    exe    = get_config()["executables"]["apbs"]
    result = subprocess.run(
        [exe, str(tmp_in_path)],
        capture_output=True, text=True,
        cwd=out_dir,   # io.mc lands here and is deleted with the directory
    )
    if result.returncode != 0:
        plog.error("APBS failed:\n%s", result.stderr)
        return False
    plog.info("APBS stdout:\n%s", result.stdout)
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def process_apbs(
    protein_id: str,
    data_root: Path,
    *,
    keep_dx: bool = False,
) -> tuple[tuple, np.ndarray] | None:
    """
    Run APBS for a protein and return the ESP grid as numpy arrays.

    Expects:
        <data_root>/<protein_id>/structure/<protein_id>.in

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory
        keep_dx:    if True, write the .dx file permanently to
                    <data_root>/<protein_id>/electrostatics/<protein_id>.dx
                    in addition to returning the grid.
                    if False (default), the .dx is written to a
                    TemporaryDirectory and deleted after reading.

    Returns:
        (axes, grid) on success — axes is a (x, y, z) tuple of 1-D coordinate
        arrays; grid is a (nx, ny, nz) float32 ESP array in kT/e.
        None on failure.
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.apbs_in_path.exists():
        plog.error("APBS input file not found: %s", p.apbs_in_path)
        plog.error("Run process_pdb2pqr first.")
        return None

    plog.info("── Running APBS  keep_dx=%s ──", keep_dx)

    with timer() as t:
        if keep_dx:
            # Write .dx to the permanent electrostatics directory.
            ok = _run_apbs_to_dir(p, p.electrostatics_dir, plog)
            if not ok:
                return None
            if not p.dx_path.exists():
                plog.error("APBS ran but .dx not found at: %s", p.dx_path)
                return None
            axes, grid = read_dx(p.dx_path)
            plog.info("ESP grid saved → %s", p.dx_path.name)
        else:
            # Write .dx to a TemporaryDirectory; read and discard.
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                ok = _run_apbs_to_dir(p, tmpdir_path, plog)
                if not ok:
                    return None
                tmp_dx = tmpdir_path / (p.protein_id + ".dx")
                if not tmp_dx.exists():
                    plog.error("APBS ran but temp .dx not found in %s", tmpdir)
                    return None
                axes, grid = read_dx(tmp_dx)
            plog.info("ESP grid loaded into memory (%.0f kB) — .dx not saved",
                      grid.nbytes / 1024)

    plog.info("APBS complete: %.2f s", t.seconds)
    update_metadata(protein_id, data_root=data_root, data={"time_apbs_sec": t.rounded})
    return axes, grid