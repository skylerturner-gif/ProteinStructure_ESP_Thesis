"""
src/surface/esp_mapping.py

ESP surface sampling — trilinear interpolation from APBS DX grid.

For a given protein ID, loads the PQR mesh (.npz) and APBS .dx ESP grid,
then trilinearly interpolates ESP onto all surface vertices and selects a
curvature-prioritised ~5% query-node subset.  The result is the canonical
ground-truth label file used during training and evaluation.

Output:
    {id}_esp.npz  — ESP at all vertices (kT/e) + curvature-sampled query_idx,
                    saved to esp/

The query_idx array in the .npz is the single authoritative vertex selection:
    - src.data.graph_builder loads it to place query nodes
    - pipelines/05_evaluate_esp.py loads it to run the RBF interpolation baseline
    - scripts/analyze_model.py loads it to reconstruct model predictions

reconstruct_full_mesh is also exported from this module — it reconstructs ESP at
all mesh vertices from sparse query-point predictions via RBF or 1-NN.

Config keys used:
    esp_mapping.normal_offset
    esp_mapping.sample_frac   (default 0.05)

Usage:
    from src.surface.esp_mapping import sample_esp, reconstruct_full_mesh
    sample_esp(protein_id="AF-Q16613-F1", data_root=Path("/data"))
"""

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree, KDTree

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


# ── DX interpolation ─────────────────────────────────────────────────────────

def trilinear_esp(axes: tuple, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation of ESP values from the DX grid.

    Weights the 8 surrounding voxels by relative distance — C0-continuous,
    no jumps at voxel boundaries unlike nearest-neighbour.  Points outside
    the grid are extrapolated via the boundary value (fill_value=None).

    Args:
        axes:   (x, y, z) 1-D coordinate arrays from read_dx
        grid:   (nx, ny, nz) ESP array in kT/e
        points: (N, 3) query coordinates in Å

    Returns:
        (N,) float32 ESP values
    """
    interp_fn = RegularGridInterpolator(
        axes, grid, method="linear", bounds_error=False, fill_value=None,
    )
    return interp_fn(points).astype(np.float32)


# ── Face interpolation ────────────────────────────────────────────────────────

def interpolate_faces_from_verts(faces: np.ndarray, esp_verts: np.ndarray) -> np.ndarray:
    """Average the ESP at each face's three vertices to get a per-face value."""
    v0 = esp_verts[faces[:, 0]]
    v1 = esp_verts[faces[:, 1]]
    v2 = esp_verts[faces[:, 2]]
    return ((v0 + v1 + v2) / 3.0).astype(np.float32)


# ── Curvature ─────────────────────────────────────────────────────────────────

def vertex_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
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

    curv = vertex_curvature(verts, faces)
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


# ── RBF mesh reconstruction ──────────────────────────────────────────────────

def _rbf_epsilon(known_verts: np.ndarray) -> float:
    """1 / mean nearest-neighbour distance among known (sampled) vertices."""
    tree = cKDTree(known_verts)
    dists, _ = tree.query(known_verts, k=2, workers=-1)  # k=2 to skip self
    return 1.0 / max(float(dists[:, 1].mean()), 1e-9)


def rbf_reconstruct(
    verts: np.ndarray,
    esp_known: np.ndarray,
    sample_idx: np.ndarray,
    kernel: str = "multiquadric",
    neighbors: int = 50,
) -> np.ndarray:
    """
    Reconstruct ESP at all mesh vertices from a sparse curvature-sampled
    subset using RBF interpolation.

    Epsilon is set to 1 / mean nearest-neighbour distance among the sampled
    vertices, scaling the kernel to transition at the typical inter-sample
    spacing (see notebooks/decisions/04_interpolation_strategy.ipynb).

    At 5% sampling (default), multiquadric RBF gives Pearson r ≈ 0.983 and
    RMSE ≈ 0.47 kT/e vs. 1-NN r ≈ 0.954 / RMSE ≈ 0.77 kT/e.

    Args:
        verts:      (N_v, 3) float32  — all mesh vertex positions
        esp_known:  (N_q,)   float32  — ESP at the sampled vertices (kT/e)
        sample_idx: (N_q,)   int64    — indices into verts of the sampled subset
        kernel:     RBF kernel ("multiquadric" | "gaussian" | "linear" | …).
                    "multiquadric" φ(r) = √(1 + (ε·r)²) is the default.
        neighbors:  Number of nearest neighbours used for local support.
                    50 gives a good accuracy/speed tradeoff at 5% sampling.

    Returns:
        (N_v,) float32 — reconstructed ESP at every mesh vertex.
    """
    known_verts = verts[sample_idx].astype(np.float64)
    eps = _rbf_epsilon(known_verts)

    rbf = RBFInterpolator(
        known_verts,
        esp_known.astype(np.float64),
        kernel=kernel,
        epsilon=eps,
        neighbors=neighbors,
    )
    return rbf(verts.astype(np.float64)).astype(np.float32)


# ── Full-mesh reconstruction ─────────────────────────────────────────────────

def reconstruct_full_mesh(
    query_pos: np.ndarray,
    query_esp: np.ndarray,
    mesh_verts: np.ndarray,
    method: str = "multiquadric",
) -> np.ndarray:
    """
    Reconstruct ESP at all mesh vertices from sparse query-point values.

    Query points are a curvature-prioritised subset of mesh vertices.  Their
    positions are matched back to vertex indices via KDTree (exact match since
    query_pos is a strict subset of mesh_verts), then passed to the chosen
    interpolation method.

    Methods:
        "multiquadric"  — RBF φ(r)=√(1+(ε·r)²), ε=1/mean-nn-dist.
                          Best accuracy (r≈0.983, RMSE≈0.47 kT/e at 5%).
        "gaussian"      — RBF φ(r)=exp(-(ε·r)²), same ε. Slightly lower accuracy.
        "nearest"       — 1-NN, instant but coarser (r≈0.954, RMSE≈0.77).

    Args:
        query_pos:  (N_q, 3) positions of query nodes
        query_esp:  (N_q,)   ESP at those positions (kT/e)
        mesh_verts: (N_v, 3) all mesh vertex positions
        method:     reconstruction method (default "multiquadric")

    Returns:
        (N_v,) float32 ESP at every mesh vertex
    """
    tree = KDTree(mesh_verts)
    _, sample_idx = tree.query(query_pos, k=1, workers=-1)
    sample_idx = sample_idx.astype(np.int64)

    if method == "nearest":
        q_tree = KDTree(query_pos)
        _, nn_idx = q_tree.query(mesh_verts, k=1, workers=-1)
        return query_esp[nn_idx].astype(np.float32)

    return rbf_reconstruct(mesh_verts, query_esp, sample_idx, kernel=method)


# ── Save ──────────────────────────────────────────────────────────────────────

def _save_npz(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    esp_verts: np.ndarray,
    esp_faces: np.ndarray,
    query_idx: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        verts=verts,
        faces=faces,
        esp_verts=esp_verts,
        esp_faces=esp_faces,
        query_idx=query_idx,
    )
    log.info(
        "Saved → %s  (%d verts, %d faces, %d query pts)",
        path.name, len(verts), len(faces), len(query_idx),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def sample_esp(
    protein_id: str,
    data_root: Path,
    *,
    grid_data: tuple | None = None,
    sample_frac: float | None = None,
) -> bool:
    """
    Trilinearly interpolate APBS ESP onto all PQR mesh vertices and select
    the curvature-prioritised query-node subset used for training/evaluation.

    Expects:
        <data_root>/<protein_id>/mesh/<protein_id>_mesh.npz
        Either grid_data=(axes, grid) passed in-memory, or
        <data_root>/<protein_id>/electrostatics/<protein_id>.dx on disk.

    Produces:
        <data_root>/<protein_id>/esp/<protein_id>_esp.npz  with keys:
            esp_verts  — ESP at every vertex (kT/e), ground truth
            esp_faces  — ESP averaged over faces (kT/e), ground truth
            query_idx  — curvature-sampled vertex indices for graph building
                         and evaluation (single authoritative selection)

    Args:
        protein_id:  e.g. "AF-Q16613-F1"
        data_root:   root of the external data directory
        grid_data:   optional (axes, grid) tuple returned by process_apbs.
                     When provided the .dx file is not read from disk.
                     When None (default), reads from p.dx_path.
        sample_frac: fraction of mesh vertices to select as query nodes.
                     Defaults to config esp_mapping.sample_frac or 0.05.

    Returns:
        True on success, False if any required input file is missing.
    """
    p    = ProteinPaths(protein_id, data_root)
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)

    if not p.mesh_path.exists():
        plog.error("Missing input file: %s", p.mesh_path)
        return False

    if grid_data is None:
        if not p.dx_path.exists():
            plog.error("Missing .dx file and no grid_data supplied: %s", p.dx_path)
            return False

    cfg           = get_config()
    normal_offset = cfg["esp_mapping"]["normal_offset"]
    if sample_frac is None:
        sample_frac = cfg.get("esp_mapping", {}).get("sample_frac", 0.05)

    plog.info(
        "── ESP sampling  normal_offset=%.2f Å  sample_frac=%.3f ──",
        normal_offset, sample_frac,
    )

    mesh_data = np.load(p.mesh_path)
    verts     = mesh_data["verts"]
    normals   = mesh_data["normals"]
    faces     = mesh_data["faces"]
    ses_area  = float(mesh_data["ses_area"])
    plog.info("Loaded mesh: %d verts, %d faces", len(verts), len(faces))

    axes, grid = grid_data if grid_data is not None else read_dx(p.dx_path)
    sample_pts = offset_points(verts, normals, normal_offset)

    with timer() as t:
        esp_verts = trilinear_esp(axes, grid, sample_pts)
        esp_faces = interpolate_faces_from_verts(faces, esp_verts)

        n_query   = max(1, int(len(verts) * sample_frac))
        query_idx = curvature_sampling(verts, faces, n_query, ses_area)

        _save_npz(p.esp_path, verts, faces, esp_verts, esp_faces, query_idx)
        plog.info(
            "ESP  verts [%.3f, %.3f]  faces [%.3f, %.3f]  query=%d",
            esp_verts.min(), esp_verts.max(),
            esp_faces.min(), esp_faces.max(),
            len(query_idx),
        )

    update_metadata(protein_id, data_root=data_root, data={
        "time_esp_sampling_sec": t.rounded,
        "n_query_nodes":         int(len(query_idx)),
    })
    plog.info("ESP sampling complete: %.2f s", t.seconds)
    return True
