"""
src/surface/mesh.py

Universal SES mesh builder.

Accepts any supported structure file (.pdb or .pqr), builds a solvent-
excluded surface (SES) mesh via MSMS, saves the mesh as .npz and .vtk,
and updates the protein's metadata JSON with mesh metrics.

Supported input formats:
    .pdb  — atom radii looked up from MDAnalysis vdW tables
    .pqr  — atom radii read directly from file (col 9)

Output .npz contains:
    verts     (N, 3) float32  — vertex coordinates
    normals   (N, 3) float32  — vertex normals
    faces     (F, 3) int      — triangle face indices
    ses_area  scalar float    — solvent-excluded surface area (Å²)
    n_verts   scalar int      — number of vertices

Metadata fields added:
    mesh_<suffix>_path   — path to .npz file
    vtk_<suffix>_path    — path to .vtk file
    n_vertices_<suffix>  — number of surface vertices
    ses_area_<suffix>    — SES area in Å²

where <suffix> is 'pdb' or 'pqr' depending on input type.

Usage (from a script):
    from src.surface.mesh import build_mesh
    build_mesh(input_file=p.pdb_path, protein_id="AF-Q16613-F1",
               data_root=Path("/data"))
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.topology.tables import vdwradii as mda_vdwradii

from src.utils.config import get_config
from src.utils.helpers import get_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── Bounding box helpers ──────────────────────────────────────────────────────

def filter_vertices_to_bbox(
    verts: np.ndarray,
    min_coords: np.ndarray,
    max_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (filtered_verts, boolean_mask) for vertices inside the bounding box."""
    mask = np.all((verts >= min_coords) & (verts <= max_coords), axis=1)
    return verts[mask], mask


def filter_faces(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only faces where all three vertices passed the bounding box mask."""
    keep = np.all(mask[faces], axis=1)
    return faces[keep]


# ── Structure parsers ─────────────────────────────────────────────────────────

def xyzr_from_pqr(pqr_file: Path, plog) -> tuple[list[str], np.ndarray]:
    """
    Parse a .pqr file and return MSMS-ready xyzr lines and atom positions.
    """
    xyzr_lines = []
    positions  = []

    with open(pqr_file) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            try:
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                radius   = float(parts[9])
            except (IndexError, ValueError) as e:
                plog.warning("Skipping malformed PQR line: %s (%s)", line.strip(), e)
                continue
            if radius <= 0:
                continue
            xyzr_lines.append(f"{x:.3f} {y:.3f} {z:.3f} {radius:.3f}")
            positions.append([x, y, z])

    if not xyzr_lines:
        raise ValueError(f"No valid atoms parsed from PQR file: {pqr_file}")

    plog.info("Parsed %d atoms from PQR file", len(xyzr_lines))
    return xyzr_lines, np.array(positions, dtype=np.float32)


def xyzr_from_pdb(pdb_file: Path, plog) -> tuple[list[str], np.ndarray]:
    """
    Parse a .pdb file via MDAnalysis and return MSMS-ready xyzr lines and
    atom positions. Atom radii are looked up from MDAnalysis vdW tables.
    """
    clean_lines = [
        line for line in open(pdb_file).readlines()
        if (line.startswith("ATOM") or line.startswith("HETATM"))
        and line[76:78].strip()
    ]

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
        tmp.writelines(clean_lines)
        tmp_path = Path(tmp.name)

    try:
        u     = mda.Universe(str(tmp_path))
        atoms = u.select_atoms("all")

        xyzr_lines = []
        for atom in atoms:
            element = atom.element.upper() if atom.element else ""
            radius  = mda_vdwradii.get(element)
            if radius is None:
                raise ValueError(
                    f"No vdW radius for element '{element}' "
                    f"(atom: {atom.name}, residue: {atom.resname} {atom.resid})"
                )
            xyzr_lines.append(
                f"{atom.position[0]:.3f} {atom.position[1]:.3f} "
                f"{atom.position[2]:.3f} {radius:.3f}"
            )
        positions = atoms.positions.copy()
    finally:
        tmp_path.unlink(missing_ok=True)

    plog.info("Parsed %d atoms from PDB file", len(xyzr_lines))
    return xyzr_lines, positions.astype(np.float32)


# ── MSMS runner ───────────────────────────────────────────────────────────────

def run_msms(
    xyzr_lines: list[str],
    positions: np.ndarray,
    plog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Write a temporary .xyzr file, run MSMS, parse output, apply bounding box
    filter, and return (verts, normals, faces, ses_area).
    """
    cfg          = get_config()["surface"]
    exe          = get_config()["executables"]["msms"]
    msms_density = cfg["msms_density"]
    probe_radius = cfg["probe_radius"]

    with tempfile.NamedTemporaryFile(suffix=".xyzr", delete=False, mode="w") as tmp_xyzr:
        tmp_xyzr.write("\n".join(xyzr_lines) + "\n")
        tmp_xyzr_path = Path(tmp_xyzr.name)

    tmp_out = tempfile.NamedTemporaryFile(delete=False)
    tmp_out_path = Path(tmp_out.name)
    tmp_out.close()

    try:
        result = subprocess.run(
            [
                exe,
                "-if",    str(tmp_xyzr_path),
                "-of",    str(tmp_out_path),
                "-d",     str(msms_density),
                "-probe", str(probe_radius),
            ],
            check=True, capture_output=True, text=True,
        )
        plog.info("MSMS stdout:\n%s", result.stdout)
        plog.info("MSMS stderr:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        plog.error("MSMS failed: %s", e)
        raise
    finally:
        tmp_xyzr_path.unlink(missing_ok=True)

    verts_file = str(tmp_out_path) + ".vert"
    face_file  = str(tmp_out_path) + ".face"

    try:
        vert_data = np.loadtxt(verts_file, usecols=(0, 1, 2, 3, 4, 5), skiprows=3)
        verts     = vert_data[:, :3].astype(np.float32)
        normals   = vert_data[:, 3:6].astype(np.float32)
        faces     = np.loadtxt(face_file, usecols=(0, 1, 2), skiprows=3, dtype=int) - 1
    finally:
        Path(verts_file).unlink(missing_ok=True)
        Path(face_file).unlink(missing_ok=True)
        tmp_out_path.unlink(missing_ok=True)

    plog.info("MSMS: %d vertices, %d faces (before bbox filter)", len(verts), len(faces))

    margin     = 3.0 * probe_radius
    min_coords = positions.min(axis=0) - margin
    max_coords = positions.max(axis=0) + margin
    verts, mask = filter_vertices_to_bbox(verts, min_coords, max_coords)
    normals     = normals[mask]
    faces       = filter_faces(faces, mask)

    plog.info("MSMS: %d vertices, %d faces (after bbox filter)", len(verts), len(faces))

    ses_area   = 0.0
    in_section = False
    for line in result.stdout.splitlines():
        if "NUMERICAL VOLUMES AND AREA" in line:
            in_section = True
            continue
        if in_section:
            m = re.match(r"^\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
            if m:
                ses_area = float(m.group(1))
                break
    if ses_area == 0.0:
        plog.warning("Could not parse SES area from MSMS output.")
    plog.info("MSMS SES area: %.3f Å²", ses_area)

    return verts, normals, faces, ses_area


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_npz_mesh(
    out_path: Path,
    verts: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    ses_area: float,
    plog,
) -> None:
    """Save mesh data to a compressed .npz file."""
    np.savez_compressed(
        out_path,
        verts=verts,
        normals=normals,
        faces=faces,
        ses_area=np.float32(ses_area),
        n_verts=np.int32(len(verts)),
    )
    plog.info("Saved mesh → %s  (%d vertices, %d faces)", out_path.name, len(verts), len(faces))


def export_vtk(out_path: Path, verts: np.ndarray, faces: np.ndarray, plog) -> None:
    """Export mesh as an ASCII VTK PolyData file for external visualization."""
    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{out_path.stem} SES surface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(verts)} float\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write(f"POLYGONS {len(faces)} {len(faces) * 4}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    plog.info("Saved VTK → %s", out_path.name)


# ── Public API ────────────────────────────────────────────────────────────────

def build_mesh(input_file: Path, protein_id: str, data_root: Path) -> Path:
    """
    Build a SES mesh from any supported structure file and save outputs to
    the protein's mesh/ subdirectory. Updates the protein's metadata JSON.

    Args:
        input_file: path to a .pdb or .pqr structure file
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory

    Returns:
        Path to the saved .npz mesh file

    Raises:
        ValueError: if the file extension is not supported
    """
    input_file = Path(input_file)
    ext        = input_file.suffix.lower()

    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    plog.info("── Building mesh from %s ──", input_file.name)

    if ext == ".pdb":
        suffix     = "pdb"
        npz_path   = p.pdb_mesh_path
        vtk_path   = p.pdb_vtk_path
        xyzr_lines, positions = xyzr_from_pdb(input_file, plog)
    elif ext == ".pqr":
        suffix     = "pqr"
        npz_path   = p.pqr_mesh_path
        vtk_path   = p.pqr_vtk_path
        xyzr_lines, positions = xyzr_from_pqr(input_file, plog)
    else:
        raise ValueError(
            f"Unsupported structure file type '{ext}'. Expected .pdb or .pqr."
        )

    with timer() as t:
        verts, normals, faces, ses_area = run_msms(xyzr_lines, positions, plog)
    plog.info("MSMS total: %.2f s", t.seconds)

    save_npz_mesh(npz_path, verts, normals, faces, ses_area, plog)
    export_vtk(vtk_path, verts, faces, plog)

    update_metadata(protein_id, data_root=data_root, data={
        f"n_vertices_{suffix}"      : int(len(verts)),
        f"ses_area_{suffix}"        : float(ses_area),
        f"time_mesh_{suffix}_sec"   : t.rounded,
    })

    plog.info("Mesh complete: %s (%s)", protein_id, suffix)
    return npz_path