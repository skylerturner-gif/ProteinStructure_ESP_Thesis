"""
src/surface/mesh.py

SES mesh builder from PQR files.

Reads a PDB2PQR-generated .pqr file, builds a solvent-excluded surface
(SES) mesh via MSMS, saves the mesh as .npz and .vtk, and updates the
protein's metadata JSON with mesh metrics.

Output .npz contains:
    verts     (N, 3) float32  — vertex coordinates
    normals   (N, 3) float32  — vertex normals
    faces     (F, 3) int      — triangle face indices
    ses_area  scalar float    — solvent-excluded surface area (Å²)
    n_verts   scalar int      — number of vertices

Metadata fields added:
    n_vertices  — number of surface vertices
    ses_area    — SES area in Å²

Usage (from a script):
    from src.surface.mesh import build_mesh
    build_mesh(pqr_file=p.pqr_path, protein_id="AF-Q16613-F1",
               data_root=Path("/data"))
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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


# ── Geometry validity and singularity correction ──────────────────────────────

# Penetration depth above which a vertex is treated as a genuine MSMS
# singularity rather than expected floating-point noise from re-entrant patches.
# MSMS's own docs (Sanner, Olson & Spehner, Biopolymers 1996) confirm that
# singularity vertex normals point arbitrarily to a probe centre, so the mesh
# normal is untrustworthy at these locations.  Vertices with sub-threshold
# penetration (~10 % of all vertices, ordinary re-entrant patches) are left
# untouched; the correction is a true no-op for them.
GEOM_THRESHOLD = 0.3  # Å


def atom_coords_radii_from_pqr(pqr_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (atom_xyz (N,3), atom_radii (N,)) from a PDB2PQR .pqr file.

    Zero-radius atoms (PARSE hydrogens) are excluded to match xyzr_from_pqr —
    the function that feeds MSMS. Including them lets the KDTree return a H as
    nearest neighbour; penetration = 0 - dist is then always negative, masking
    real heavy-atom defects.
    """
    coords, radii = [], []
    with open(pqr_file) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                radius = float(parts[9])
                if radius <= 0:
                    continue
                coords.append((float(parts[5]), float(parts[6]), float(parts[7])))
                radii.append(radius)
            except ValueError:
                continue
    return np.array(coords, dtype=np.float32), np.array(radii, dtype=np.float32)


def geom_validity(
    verts: np.ndarray,
    atom_xyz: np.ndarray,
    atom_radii: np.ndarray,
    threshold: float = GEOM_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-vertex penetration depth into the nearest atom's VdW sphere.

    Returns:
        penetration  (N,) float32 — positive where vertex is inside an atom
        needs_fix    (N,) bool    — True where penetration > threshold
    """
    tree = cKDTree(atom_xyz)
    dist, nearest = tree.query(verts, k=1)
    penetration = (atom_radii[nearest] - dist).astype(np.float32)
    return penetration, penetration > threshold


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals recomputed from face geometry."""
    normals = np.zeros_like(verts, dtype=np.float64)
    v0 = verts[faces[:, 0]].astype(np.float64)
    v1 = verts[faces[:, 1]].astype(np.float64)
    v2 = verts[faces[:, 2]].astype(np.float64)
    face_normals = np.cross(v1 - v0, v2 - v0)  # magnitude = 2 × face area
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (normals / norms).astype(np.float32)


def resolve_singularities(
    verts: np.ndarray,
    atom_xyz: np.ndarray,
    atom_radii: np.ndarray,
    margin: float = 0.15,
    max_passes: int = 15,
    max_displacement: float = 4.0,
) -> tuple[np.ndarray, int]:
    """
    Push each vertex outside all atom VdW spheres by displacing along the
    atom-centre → vertex vector (not the unreliable mesh normal).

    Iterates up to max_passes times because escaping one atom can move a
    vertex into a tightly-packed neighbour.  Total displacement is capped at
    max_displacement Å as a safety valve against pathological geometry.

    Returns:
        fixed_verts   (N, 3) float32 — corrected vertex positions
        n_corrected   int             — number of vertices actually moved
    """
    tree = cKDTree(atom_xyz)
    fixed  = verts.copy().astype(np.float32)
    origin = verts.copy().astype(np.float32)
    moved  = np.zeros(len(fixed), dtype=bool)

    for _ in range(max_passes):
        dist, nearest = tree.query(fixed, k=1)
        violated = dist < atom_radii[nearest]
        if not violated.any():
            break
        for i in np.where(violated)[0]:
            ai        = int(nearest[i])
            direction = fixed[i] - atom_xyz[ai]
            d         = float(np.linalg.norm(direction))
            if d < 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                d = 1.0
            target = atom_xyz[ai] + (atom_radii[ai] + margin) * (direction / d)
            # Clamp total displacement from original position
            disp = np.linalg.norm(target - origin[i])
            if disp > max_displacement:
                target = origin[i] + max_displacement * (target - origin[i]) / disp
            fixed[i] = target
            moved[i] = True

    return fixed, int(moved.sum())


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

def build_mesh(pqr_file: Path, protein_id: str, data_root: Path) -> Path:
    """
    Build a SES mesh from a .pqr file and save outputs to the protein's
    mesh/ subdirectory. Updates the protein's metadata JSON.

    Args:
        pqr_file:   path to the PDB2PQR-generated .pqr structure file
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory

    Returns:
        Path to the saved .npz mesh file
    """
    pqr_file = Path(pqr_file)

    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    plog.info("── Building mesh from %s ──", pqr_file.name)

    xyzr_lines, positions = xyzr_from_pqr(pqr_file, plog)

    with timer() as t:
        verts, normals, faces, ses_area = run_msms(xyzr_lines, positions, plog)
    plog.info("MSMS total: %.2f s", t.seconds)

    atom_xyz, atom_radii = atom_coords_radii_from_pqr(pqr_file)
    pen, needs_fix = geom_validity(verts, atom_xyz, atom_radii, GEOM_THRESHOLD)
    if needs_fix.any():
        fixed_subset, n_corrected = resolve_singularities(
            verts[needs_fix], atom_xyz, atom_radii
        )
        verts[needs_fix] = fixed_subset
        normals = _compute_vertex_normals(verts, faces)
        plog.info(
            "Singularity correction: %d vertices moved (max pen=%.3f Å)",
            n_corrected, float(pen[needs_fix].max()),
        )

    save_npz_mesh(p.mesh_path, verts, normals, faces, ses_area, plog)
    export_vtk(p.vtk_path, verts, faces, plog)

    update_metadata(protein_id, data_root=data_root, data={
        "n_vertices"    : int(len(verts)),
        "ses_area"      : float(ses_area),
        "time_mesh_sec" : t.rounded,
    })

    plog.info("Mesh complete: %s", protein_id)
    return p.mesh_path