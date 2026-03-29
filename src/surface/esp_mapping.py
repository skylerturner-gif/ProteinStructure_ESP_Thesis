"""
src/surface/esp_mapping.py

ESP surface sampling — interpolated and Laplacian reconstruction.

For a given protein ID, loads the PDB mesh (.npz), PQR mesh (_pqr.npz),
and shared .dx ESP grid, then runs both no-H and with-H sampling
sequentially. Each variant produces two output files:

    {id}_pdb_mesh_interp.npz     — nearest-neighbor ESP, no H
    {id}_pdb_mesh_laplacian.npz  — Laplacian-reconstructed ESP, no H
    {id}_pqr_mesh_interp.npz     — nearest-neighbor ESP, with H
    {id}_pqr_mesh_laplacian.npz  — Laplacian-reconstructed ESP, with H

All outputs are saved to the protein's esp/ subdirectory.
Metrics (Pearson r, RMSE) are computed separately via src.analysis.metrics.
No metadata fields are written — paths are derived from ProteinPaths.

Config keys used:
    esp_mapping.normal_offset
    esp_mapping.subsample_n

Usage (from a script):
    from src.surface.esp_mapping import sample_esp
    sample_esp(protein_id="AF-Q16613-F1", data_root=Path("/data"))
"""

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.utils.config import get_config
from src.utils.helpers import get_logger, timer
from src.utils.io import update_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)


# ── DX I/O ────────────────────────────────────────────────────────────────────

def read_dx(dx_file: Path) -> tuple[tuple, np.ndarray]:
    """
    Parse an OpenDX scalar field file and return the grid axes and data.

    Returns:
        axes: tuple of (x, y, z) 1-D coordinate arrays
        grid: (nx, ny, nz) float32 array of ESP values in kT/e

    Raises:
        ValueError: if the file is malformed or data is truncated
    """
    nx = ny = nz = None
    origin    = []
    deltas    = []
    dx_values = []

    with open(dx_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("object 1 class gridpositions counts"):
                nx, ny, nz = map(int, line.split()[-3:])
            elif line.startswith("origin"):
                origin = list(map(float, line.split()[1:4]))
            elif line.startswith("delta"):
                deltas.append(list(map(float, line.split()[1:4])))
            elif line.startswith("object") or line.startswith("attribute") or line.startswith("component"):
                continue
            elif re.match(r"^[-+]?\d", line):
                dx_values.extend([float(x) for x in line.split()])

    if None in (nx, ny, nz) or len(origin) != 3 or len(deltas) != 3:
        raise ValueError(f"Malformed DX file: {dx_file}")
    if len(dx_values) < nx * ny * nz:
        raise ValueError(
            f"DX data truncated: expected {nx * ny * nz}, got {len(dx_values)}"
        )

    grid = np.array(dx_values[: nx * ny * nz], dtype=np.float32).reshape((nx, ny, nz))
    x    = origin[0] + np.arange(nx) * deltas[0][0]
    y    = origin[1] + np.arange(ny) * deltas[1][1]
    z    = origin[2] + np.arange(nz) * deltas[2][2]

    log.info("DX grid shape: %s  range: [%.3f, %.3f] kT/e",
             grid.shape, grid.min(), grid.max())
    return (x, y, z), grid


# ── Normal offset ─────────────────────────────────────────────────────────────

def offset_points(points: np.ndarray, normals: np.ndarray, offset: float) -> np.ndarray:
    """Shift surface points outward along their normals by offset Å."""
    return (points + offset * normals).astype(np.float32)


# ── Nearest-neighbor ESP lookup ───────────────────────────────────────────────

def nearest_neighbor_esp(axes: tuple, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Snap each point to the nearest DX grid voxel and return its ESP value.

    Args:
        axes:   (x, y, z) coordinate arrays from read_dx
        grid:   (nx, ny, nz) ESP array from read_dx
        points: (N, 3) float32 query coordinates

    Returns:
        (N,) float32 array of ESP values in kT/e
    """
    x_ax, y_ax, z_ax = axes

    dx = x_ax[1] - x_ax[0]
    dy = y_ax[1] - y_ax[0]
    dz = z_ax[1] - z_ax[0]

    ix = np.rint((points[:, 0] - x_ax[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - y_ax[0]) / dy).astype(int)
    iz = np.rint((points[:, 2] - z_ax[0]) / dz).astype(int)

    ix = np.clip(ix, 0, len(x_ax) - 1)
    iy = np.clip(iy, 0, len(y_ax) - 1)
    iz = np.clip(iz, 0, len(z_ax) - 1)

    return grid[ix, iy, iz].astype(np.float32)


# ── Face interpolation ────────────────────────────────────────────────────────

def interpolate_faces_from_verts(faces: np.ndarray, esp_verts: np.ndarray) -> np.ndarray:
    """Average the ESP at each face's three vertices to get a per-face value."""
    v0 = esp_verts[faces[:, 0]]
    v1 = esp_verts[faces[:, 1]]
    v2 = esp_verts[faces[:, 2]]
    return ((v0 + v1 + v2) / 3.0).astype(np.float32)


# ── Cotangent Laplacian ───────────────────────────────────────────────────────

def build_cotangent_laplacian(verts: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    """
    Build the cotangent-weighted Laplacian matrix for curvature estimation.

    For each directed edge (v1, v2) opposite angle at v0:
        weight = cot(angle at v0) = dot(a, b) / |cross(a, b)|
    """
    n    = len(verts)
    I, J, W = [], [], []

    for f in faces:
        for i in range(3):
            v0 = f[i]
            v1 = f[(i + 1) % 3]
            v2 = f[(i + 2) % 3]
            a  = verts[v1] - verts[v0]
            b  = verts[v2] - verts[v0]
            cross_norm = np.linalg.norm(np.cross(a, b))
            if cross_norm < 1e-12:
                continue
            cot = np.dot(a, b) / cross_norm
            I.extend([v1, v2])
            J.extend([v2, v1])
            W.extend([cot, cot])

    L_off = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    diag  = -np.array(L_off.sum(axis=1)).flatten()
    return L_off + sp.diags(diag)


def vertex_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Approximate mean curvature magnitude via cotangent Laplacian:
        H_i = ||L * verts||_2  (row-wise 2-norm)
    """
    L  = build_cotangent_laplacian(verts, faces)
    Lv = L @ verts
    return np.linalg.norm(Lv, axis=1).astype(np.float32)


# ── Curvature-biased sampling ─────────────────────────────────────────────────

def curvature_sampling(verts: np.ndarray, faces: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k vertex indices with probability proportional to curvature.
    High-curvature regions (pockets, ridges) are sampled more densely.

    Returns:
        Sorted index array of length k.
    """
    curv = vertex_curvature(verts, faces)
    prob = curv + 1e-6
    prob /= prob.sum()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(verts), size=k, replace=False, p=prob)
    return np.sort(idx)


# ── Laplacian reconstruction ──────────────────────────────────────────────────

def laplacian_reconstruct(
    verts: np.ndarray,
    faces: np.ndarray,
    known_idx: np.ndarray,
    known_values: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct ESP at all vertices by solving the combinatorial Laplacian
    system with known vertices constrained to their interpolated values.

    Uses unweighted (combinatorial) Laplacian for robustness — all edge
    weights are 1, avoiding singularities from degenerate triangles.
    Cotangent weights are used only for curvature-biased sampling above.

    Solves:  L_uu * x_unknown = -L_uk * x_known

    Returns:
        (N,) float32 array of reconstructed ESP values at all vertices.
    """
    n = len(verts)

    rows, cols = [], []
    for f in faces:
        for i in range(3):
            v0 = f[i]
            v1 = f[(i + 1) % 3]
            rows.extend([v0, v1])
            cols.extend([v1, v0])

    data = np.ones(len(rows))
    A    = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    D    = sp.diags(np.array(A.sum(axis=1)).flatten())
    L    = D - A

    mask        = np.ones(n, dtype=bool)
    mask[known_idx] = False           # True = unknown, False = known

    L_uu = L[mask][:, mask]
    L_uk = L[mask][:, ~mask]
    rhs  = -L_uk @ known_values

    log.info("Laplacian solve: %d unknowns, %d known", mask.sum(), (~mask).sum())
    with timer() as t:
        x_unknown = spla.spsolve(L_uu, rhs)
    log.info("Laplacian solve: %.2f s", t.seconds)

    x         = np.zeros(n, dtype=np.float32)
    x[~mask]  = known_values
    x[mask]   = x_unknown
    return x


# ── Save ──────────────────────────────────────────────────────────────────────

def _save_npz(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    esp_verts: np.ndarray,
    esp_faces: np.ndarray,
    subsample_n: int = None,
) -> None:
    kwargs = dict(verts=verts, faces=faces,
                  esp_verts=esp_verts, esp_faces=esp_faces)
    if subsample_n is not None:
        kwargs["subsample_n"] = np.array(subsample_n)
    np.savez_compressed(path, **kwargs)
    log.info("Saved → %s  (%d verts, %d faces)", path.name, len(verts), len(faces))


# ── Per-variant sampling ──────────────────────────────────────────────────────

def _sample_variant(
    mesh_npz: Path,
    p: ProteinPaths,
    suffix: str,
    normal_offset: float,
    subsample_n: int,
    plog,
) -> float:
    """
    Run interpolated and Laplacian ESP sampling for one mesh variant.

    Args:
        mesh_npz:      path to the mesh .npz file
        p:             ProteinPaths instance for this protein
        suffix:        'pdb' or 'pqr'
        normal_offset: outward normal offset in Å
        subsample_n:   keep every Nth vertex as known for Laplacian
        plog:          per-protein logger

    Returns:
        elapsed time in seconds
    """
    stem = mesh_npz.stem

    data    = np.load(mesh_npz)
    verts   = data["verts"]
    normals = data["normals"]
    faces   = data["faces"]
    plog.info("[%s] Loaded mesh: %d verts, %d faces", suffix, len(verts), len(faces))

    axes, grid = read_dx(p.dx_path)

    sample_pts = offset_points(verts, normals, normal_offset)

    with timer() as t:
        # ── Interpolated ESP ──────────────────────────────────────────────────
        esp_verts_interp = nearest_neighbor_esp(axes, grid, sample_pts)
        esp_faces_interp = interpolate_faces_from_verts(faces, esp_verts_interp)

        interp_path = p.esp_dir / f"{stem}_interp.npz"
        _save_npz(interp_path, verts, faces, esp_verts_interp, esp_faces_interp)
        plog.info("[%s] Interp ESP  verts [%.3f, %.3f]  faces [%.3f, %.3f]",
                  suffix,
                  esp_verts_interp.min(), esp_verts_interp.max(),
                  esp_faces_interp.min(), esp_faces_interp.max())

        # ── Laplacian reconstruction ──────────────────────────────────────────
        k = len(verts) // subsample_n
        plog.info("[%s] Curvature sampling: k=%d of %d vertices (N=%d)",
                  suffix, k, len(verts), subsample_n)

        known_idx     = curvature_sampling(verts, faces, k)
        known_values  = esp_verts_interp[known_idx]
        esp_verts_lap = laplacian_reconstruct(verts, faces, known_idx, known_values)
        esp_faces_lap = interpolate_faces_from_verts(faces, esp_verts_lap)

        lap_path = p.esp_dir / f"{stem}_laplacian.npz"
        _save_npz(lap_path, verts, faces, esp_verts_lap, esp_faces_lap,
                  subsample_n=subsample_n)
        plog.info("[%s] Laplacian ESP  verts [%.3f, %.3f]  faces [%.3f, %.3f]",
                  suffix,
                  esp_verts_lap.min(), esp_verts_lap.max(),
                  esp_faces_lap.min(), esp_faces_lap.max())

    plog.info("[%s] ESP sampling complete: %.2f s", suffix, t.seconds)
    return t.rounded


# ── Public API ────────────────────────────────────────────────────────────────

def sample_esp(protein_id: str, data_root: Path) -> bool:
    """
    Run ESP surface sampling for both PDB (no-H) and PQR (with-H) mesh
    variants for a protein, and update its metadata JSON with timing.

    Expects:
        <data_root>/<protein_id>/mesh/<protein_id>_pdb_mesh.npz
        <data_root>/<protein_id>/mesh/<protein_id>_pqr_mesh.npz
        <data_root>/<protein_id>/electrostatics/<protein_id>.dx

    Produces (per variant):
        <data_root>/<protein_id>/esp/<protein_id>_pdb_mesh_interp.npz
        <data_root>/<protein_id>/esp/<protein_id>_pdb_mesh_laplacian.npz
        <data_root>/<protein_id>/esp/<protein_id>_pqr_mesh_interp.npz
        <data_root>/<protein_id>/esp/<protein_id>_pqr_mesh_laplacian.npz

    Args:
        protein_id: e.g. "AF-Q16613-F1"
        data_root:  root of the external data directory

    Returns:
        True on success, False if any required input file is missing.
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    missing = [f for f in [p.pdb_mesh_path, p.pqr_mesh_path, p.dx_path] if not f.exists()]
    if missing:
        for f in missing:
            plog.error("Missing input file: %s", f)
        return False

    cfg           = get_config()["esp_mapping"]
    normal_offset = cfg["normal_offset"]
    subsample_n   = cfg["subsample_n"]

    plog.info("── ESP sampling  normal_offset=%.2f Å  subsample_n=%d ──",
              normal_offset, subsample_n)

    timing = {}
    for mesh_npz, suffix in [(p.pdb_mesh_path, "pdb"), (p.pqr_mesh_path, "pqr")]:
        elapsed = _sample_variant(
            mesh_npz, p, suffix, normal_offset, subsample_n, plog
        )
        timing[f"time_esp_sampling_{suffix}_sec"] = elapsed

    update_metadata(protein_id, data_root=data_root, data=timing)
    plog.info("ESP sampling complete")
    return True