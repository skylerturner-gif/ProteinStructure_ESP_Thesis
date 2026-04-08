"""
src/surface/esp_mapping.py

ESP surface sampling — interpolated and Laplacian reconstruction.

For a given protein ID, loads the PQR mesh (.npz) and shared .dx ESP grid,
then produces two output files:

    {id}_pqr_mesh_interp.npz     — nearest-neighbor ESP (with H)
    {id}_pqr_mesh_laplacian.npz  — Laplacian-reconstructed ESP (with H)

The Laplacian reconstruction uses 5% of vertices selected by a
curvature-prioritised sampler with minimum spacing (see curvature_sampling).

All outputs are saved to the protein's esp/ subdirectory.
Metrics (Pearson r, RMSE) are computed separately via src.analysis.metrics.
No metadata fields are written here — paths are derived from ProteinPaths.

Config keys used:
    esp_mapping.normal_offset
    esp_mapping.sample_frac

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


# ── Curvature ─────────────────────────────────────────────────────────────────

def vertex_curvature_fast(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Vectorized cotangent-Laplacian mean curvature magnitude.
        H_i = ||L * verts||_2  (row-wise 2-norm)
    """
    n = len(verts)
    all_rows, all_cols, all_w = [], [], []

    for i in range(3):
        v0, v1, v2 = faces[:, i], faces[:, (i + 1) % 3], faces[:, (i + 2) % 3]
        a           = verts[v1] - verts[v0]
        b           = verts[v2] - verts[v0]
        cross_norm  = np.linalg.norm(np.cross(a, b), axis=1)
        dot         = np.einsum("ij,ij->i", a, b)
        valid       = cross_norm > 1e-12
        cot         = np.where(valid, dot / np.where(valid, cross_norm, 1.0), 0.0)

        v1v, v2v, cotv = v1[valid], v2[valid], cot[valid]
        all_rows.extend([v1v, v2v])
        all_cols.extend([v2v, v1v])
        all_w.extend([cotv, cotv])

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    w    = np.concatenate(all_w)

    L_off = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    diag  = -np.array(L_off.sum(axis=1)).flatten()
    L     = L_off + sp.diags(diag)

    Lv = L @ verts
    return np.linalg.norm(Lv, axis=1).astype(np.float32)


# ── Curvature-prioritised sampling with minimum spacing ───────────────────────

def curvature_sampling(
    verts: np.ndarray,
    faces: np.ndarray,
    k: int,
    ses_area: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sample k vertex indices using curvature-prioritised selection with
    minimum spacing enforcement.

    Vertices are visited in descending curvature order; a vertex is accepted
    only if no already-selected vertex lies within radius r, where
    r = sqrt(ses_area / π·k).  If the greedy pass yields fewer than k
    vertices, remaining slots are filled from the rejected pool (still in
    curvature order) with no spacing constraint.

    Args:
        verts:    (N, 3) float32 vertex positions
        faces:    (F, 3) int64 face indices
        k:        target number of vertices to select
        ses_area: solvent-excluded surface area in Å² (used to scale r)
        rng:      optional numpy Generator for tie-breaking; defaults to seed 0

    Returns:
        Sorted int64 index array of length k (or fewer if mesh is small).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    curv = vertex_curvature_fast(verts, faces)
    n    = len(verts)
    k    = min(k, n)

    r    = np.sqrt(ses_area / (np.pi * k))
    r2   = float(r * r)
    cs   = float(r) + 1e-9
    mins = verts.min(axis=0)

    offsets = [(ox, oy, oz)
               for ox in (-1, 0, 1)
               for oy in (-1, 0, 1)
               for oz in (-1, 0, 1)]

    noise       = rng.random(n) * 1e-9
    visit_order = np.argsort(-(curv + noise))

    grid_h   = {}
    selected = []
    rejected = []

    for idx in visit_order:
        vx = float(verts[idx, 0])
        vy = float(verts[idx, 1])
        vz = float(verts[idx, 2])
        cx = int((vx - mins[0]) / cs)
        cy = int((vy - mins[1]) / cs)
        cz = int((vz - mins[2]) / cs)

        ok = True
        for ox, oy, oz in offsets:
            pts = grid_h.get((cx + ox, cy + oy, cz + oz))
            if pts is None:
                continue
            for px, py, pz in pts:
                if (vx - px) ** 2 + (vy - py) ** 2 + (vz - pz) ** 2 < r2:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            selected.append(idx)
            cell = (cx, cy, cz)
            if cell in grid_h:
                grid_h[cell].append((vx, vy, vz))
            else:
                grid_h[cell] = [(vx, vy, vz)]
            if len(selected) == k:
                break
        else:
            rejected.append(idx)

    if len(selected) < k:
        need = k - len(selected)
        selected.extend(rejected[:need])
        log.info("Curvature sampling: spacing-pass gave %d / %d — filled %d from rejected pool",
                 len(selected) - need, k, need)

    return np.sort(np.array(selected, dtype=np.int64))


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
    mask[known_idx] = False

    L_uu = L[mask][:, mask]
    L_uk = L[mask][:, ~mask]
    rhs  = -L_uk @ known_values

    log.info("Laplacian solve: %d unknowns, %d known", mask.sum(), (~mask).sum())
    with timer() as t:
        x_unknown = spla.spsolve(L_uu, rhs)
    log.info("Laplacian solve: %.2f s", t.seconds)

    x        = np.zeros(n, dtype=np.float32)
    x[~mask] = known_values
    x[mask]  = x_unknown
    return x


# ── Save ──────────────────────────────────────────────────────────────────────

def _save_npz(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    esp_verts: np.ndarray,
    esp_faces: np.ndarray,
    sample_frac: float = None,
) -> None:
    kwargs = dict(verts=verts, faces=faces,
                  esp_verts=esp_verts, esp_faces=esp_faces)
    if sample_frac is not None:
        kwargs["sample_frac"] = np.array(sample_frac)
    np.savez_compressed(path, **kwargs)
    log.info("Saved → %s  (%d verts, %d faces)", path.name, len(verts), len(faces))


# ── Public API ────────────────────────────────────────────────────────────────

def sample_esp(protein_id: str, data_root: Path) -> bool:
    """
    Run ESP surface sampling for the PQR mesh variant of a protein.

    Expects:
        <data_root>/<protein_id>/mesh/<protein_id>_pqr_mesh.npz
        <data_root>/<protein_id>/electrostatics/<protein_id>.dx

    Produces:
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

    missing = [f for f in [p.pqr_mesh_path, p.dx_path] if not f.exists()]
    if missing:
        for f in missing:
            plog.error("Missing input file: %s", f)
        return False

    cfg           = get_config()["esp_mapping"]
    normal_offset = cfg["normal_offset"]
    sample_frac   = cfg["sample_frac"]

    plog.info("── ESP sampling  normal_offset=%.2f Å  sample_frac=%.2f ──",
              normal_offset, sample_frac)

    data    = np.load(p.pqr_mesh_path)
    verts   = data["verts"]
    normals = data["normals"]
    faces   = data["faces"]
    ses_area = float(data["ses_area"])
    plog.info("Loaded PQR mesh: %d verts, %d faces", len(verts), len(faces))

    axes, grid = read_dx(p.dx_path)
    sample_pts = offset_points(verts, normals, normal_offset)

    with timer() as t:
        # ── Interpolated ESP ──────────────────────────────────────────────────
        esp_verts_interp = nearest_neighbor_esp(axes, grid, sample_pts)
        esp_faces_interp = interpolate_faces_from_verts(faces, esp_verts_interp)

        _save_npz(p.pqr_interp_path, verts, faces, esp_verts_interp, esp_faces_interp)
        plog.info("Interp ESP  verts [%.3f, %.3f]  faces [%.3f, %.3f]",
                  esp_verts_interp.min(), esp_verts_interp.max(),
                  esp_faces_interp.min(), esp_faces_interp.max())

        # ── Laplacian reconstruction ──────────────────────────────────────────
        k = max(1, round(sample_frac * len(verts)))
        plog.info("Curvature sampling: k=%d of %d vertices (%.1f%%)",
                  k, len(verts), sample_frac * 100)

        known_idx     = curvature_sampling(verts, faces, k, ses_area)
        known_values  = esp_verts_interp[known_idx]
        esp_verts_lap = laplacian_reconstruct(verts, faces, known_idx, known_values)
        esp_faces_lap = interpolate_faces_from_verts(faces, esp_verts_lap)

        _save_npz(p.pqr_laplacian_path, verts, faces, esp_verts_lap, esp_faces_lap,
                  sample_frac=sample_frac)
        plog.info("Laplacian ESP  verts [%.3f, %.3f]  faces [%.3f, %.3f]",
                  esp_verts_lap.min(), esp_verts_lap.max(),
                  esp_faces_lap.min(), esp_faces_lap.max())

    update_metadata(protein_id, data_root=data_root, data={
        "time_esp_sampling_sec": t.rounded,
    })
    plog.info("ESP sampling complete: %.2f s", t.seconds)
    return True
