"""
src/data/graph_builder.py

Builds the bipartite atom ↔ query-point heterogeneous graph for EGNN training.

Graph node types
----------------
  atom  — all atoms (heavy + H) from the PQR file
  query — curvature-sampled surface mesh vertices (loaded from esp .npz query_idx)

Graph edge types (all directed)
--------------------------------
  ('atom',  'bond',  'atom')  — covalent bonds, MDAnalysis-detected
  ('atom',  'radial', 'atom')  — radial supplementary kNN=16, bond pairs excluded
  ('atom',  'aq',   'query') — kNN=32 atom→query (query-centric)
  ('query', 'qq',   'query') — kNN=8  query→query

Atom node features (integer indices — fed to nn.Embedding in the model)
------------------------------------------------------------------------
  atom_type    — element → index in ELEMENT_VOCAB  (unknown = N_ELEMENT_TYPES-1)
  residue_type — residue → index in RESIDUE_VOCAB  (unknown = N_RESIDUE_TYPES-1)
  bond_count   — number of covalent bonds (scalar int)
  pos          — (3,) float32 position in Å

Query node features
-------------------
  pos — (3,) float32 position in Å
  y   — float32 target ESP value in kT/e  (absent if ESP file not found)

Edge attributes
---------------
  bond  : [bond_order (1)] ++ [RBF distance (N_RBF)]  →  N_RBF+1 floats
  radial : [RBF distance (N_RBF)]
  aq   : [RBF distance (N_RBF)]
  qq   : [RBF distance (N_RBF)]

RBF distance ranges (Å)
  bond  [0.9, 1.8]   radial [1.8, 8.0]   aq [0.0, 12.0]   qq [0.0, 8.0]

Public API
----------
  build_graph(protein_id, data_root, *, knn_radial, knn_aq, knn_qq, n_rbf) -> HeteroData
  ELEMENT_VOCAB, RESIDUE_VOCAB, N_ELEMENT_TYPES, N_RESIDUE_TYPES  (model constants)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from MDAnalysis.topology.guessers import guess_types
from scipy.spatial import KDTree
from torch_geometric.data import HeteroData

import MDAnalysis as mda

from src.utils.config import get_config
from src.utils.helpers import get_logger
from src.utils.paths import ProteinPaths

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

log = get_logger(__name__)

# ── Atom / residue vocabularies ───────────────────────────────────────────────

ELEMENT_VOCAB: dict[str, int] = {
    "H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "P": 5,
}
N_ELEMENT_TYPES: int = len(ELEMENT_VOCAB) + 1   # 7  (index 6 = unknown)

RESIDUE_VOCAB: dict[str, int] = {
    "ALA": 0, "ARG": 1,  "ASN": 2,  "ASP": 3,  "CYS": 4,
    "GLN": 5, "GLU": 6,  "GLY": 7,  "HIS": 8,  "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
N_RESIDUE_TYPES: int = len(RESIDUE_VOCAB) + 1   # 21  (index 20 = unknown)

# ── Bond order heuristic ──────────────────────────────────────────────────────

_AROMATIC_ATOMS: dict[str, set[str]] = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
}


def _assign_bond_order(bond, elements: list[str]) -> float:
    """Heuristic bond order: 1.0 (single/X-H), 1.5 (aromatic/peptide), 2.0 (C=O)."""
    a1, a2 = bond.atoms
    e1, e2 = elements[a1.index], elements[a2.index]

    # 1. Hydrogens
    if "H" in (e1, e2):
        return 1.0

    # 2. Aromatic rings
    ring = _AROMATIC_ATOMS.get(a1.resname)
    if ring and a1.name in ring and a2.name in ring:
        return 1.5

    # 3. Peptide / amide C–N
    if {e1, e2} == {"C", "N"}:
        c_atom = a1 if e1 == "C" else a2
        has_carbonyl_oxygen = any(
            elements[nbr.index] == "O" and len(nbr.bonds) == 1
            for nbr in c_atom.bonded_atoms
        )
        if has_carbonyl_oxygen:
            return 1.5

    # 4. Carbonyl C=O
    if {e1, e2} == {"C", "O"}:
        o_atom = a1 if e1 == "O" else a2
        if len(o_atom.bonds) == 1:
            return 2.0

    # 5. Default
    return 1.0


# ── MDAnalysis bond graph ─────────────────────────────────────────────────────

def _build_bond_graph(
    pqr_path: Path,
) -> tuple[
    np.ndarray,   # atom_xyz      (N, 3)  float32
    list[str],    # atom_names    (N,)
    list[str],    # atom_resnames (N,)
    np.ndarray,   # bond_src       (E,)    int64
    np.ndarray,   # bond_dst       (E,)    int64
    np.ndarray,   # bond_orders   (E,)    float32
    np.ndarray,   # bond_dists    (E,)    float32
    set[tuple],   # bond_set      undirected pairs (min,max)
]:
    u = mda.Universe(str(pqr_path), to_guess=["bonds"])
    elements      = list(guess_types(u.atoms.names))
    u.add_TopologyAttr("elements", elements)

    atom_xyz      = u.atoms.positions.astype(np.float32)
    atom_names    = list(u.atoms.names)
    atom_resnames = list(u.atoms.resnames)

    src_list, dst_list, order_list, dist_list = [], [], [], []
    bond_set: set[tuple[int, int]] = set()

    for bond in u.bonds:
        i, j   = bond.atoms[0].index, bond.atoms[1].index
        order  = _assign_bond_order(bond, elements)
        dist   = float(bond.length())
        src_list += [i, j]
        dst_list += [j, i]
        order_list += [order, order]
        dist_list  += [dist,  dist]
        bond_set.add((min(i, j), max(i, j)))

    return (
        atom_xyz, atom_names, atom_resnames,
        np.array(src_list,   dtype=np.int64),
        np.array(dst_list,   dtype=np.int64),
        np.array(order_list, dtype=np.float32),
        np.array(dist_list,  dtype=np.float32),
        bond_set,
    )


# ── Edge builders ─────────────────────────────────────────────────────────────

def _knn_self(xyz: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed kNN within one node set (self-edges excluded)."""
    k = min(k, len(xyz) - 1)
    tree = KDTree(xyz)
    dists, idx = tree.query(xyz, k=k + 1)
    n = len(xyz)
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    tgt = idx[:, 1:].flatten().astype(np.int64)
    d   = dists[:, 1:].flatten().astype(np.float32)
    return src, tgt, d


def _knn_bipartite(
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Query-centric bipartite kNN: for each target node find its k nearest
    source nodes.  Returns (src_idx, tgt_idx, distances).
    """
    k = min(k, len(src_xyz))
    tree = KDTree(src_xyz)
    dists, idx = tree.query(tgt_xyz, k=k)
    n_tgt = len(tgt_xyz)
    tgt   = np.repeat(np.arange(n_tgt, dtype=np.int64), k)
    src   = idx.flatten().astype(np.int64)
    return src, tgt, dists.flatten().astype(np.float32)


def _knn_radial(
    atom_xyz: np.ndarray,
    k: int,
    bond_set: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    radial supplementary kNN: k nearest atom-atom edges with bonded pairs excluded.
    Uses a look-ahead buffer of k+8 to absorb exclusions.
    """
    buf  = k + 8
    k    = min(k, len(atom_xyz) - 1)
    tree = KDTree(atom_xyz)
    dists_all, idx_all = tree.query(atom_xyz, k=min(buf + 1, len(atom_xyz)))
    dists_all = dists_all[:, 1:]   # drop self
    idx_all   = idx_all[:, 1:]

    src_list, tgt_list, d_list = [], [], []
    for i in range(len(atom_xyz)):
        count = 0
        for rank in range(idx_all.shape[1]):
            j = int(idx_all[i, rank])
            if (min(i, j), max(i, j)) in bond_set:
                continue
            src_list.append(i)
            tgt_list.append(j)
            d_list.append(float(dists_all[i, rank]))
            count += 1
            if count == k:
                break

    return (
        np.array(src_list, dtype=np.int64),
        np.array(tgt_list, dtype=np.int64),
        np.array(d_list,   dtype=np.float32),
    )


def _rbf_encode(
    dists: np.ndarray,
    n_rbf: int,
    d_min: float,
    d_max: float,
) -> np.ndarray:
    """Gaussian RBF encoding. Returns (len(dists), n_rbf) float32 array."""
    centers = np.linspace(d_min, d_max, n_rbf, dtype=np.float32)
    sigma   = (d_max - d_min) / max(n_rbf - 1, 1)
    d       = dists[:, None].astype(np.float32)
    return np.exp(-((d - centers) ** 2) / (sigma ** 2))


# ── Surface geometry helpers ──────────────────────────────────────────────────

def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals from face geometry.

    For each face the unnormalised normal (whose magnitude equals twice the
    face area) is accumulated into each of the three incident vertices.
    Vertex normals are then unit-normalised.

    Args:
        verts: (V, 3) float32 vertex positions
        faces: (F, 3) int   face vertex indices

    Returns:
        (V, 3) float32 unit vertex normals
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)          # (F, 3) — magnitude = 2 × area
    vn = np.zeros_like(verts, dtype=np.float64)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    norms = np.linalg.norm(vn, axis=-1, keepdims=True)
    return (vn / np.maximum(norms, 1e-8)).astype(np.float32)


def _compute_mean_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Discrete mean curvature magnitude via the cotangent Laplacian.

    Uses the standard formula H_i = |Δ x_i| / 2 where Δ is the
    cotangent-weighted Laplace–Beltrami operator.  Mixed area is
    approximated as one third of the sum of incident triangle areas.

    Args:
        verts: (V, 3) float32 vertex positions
        faces: (F, 3) int   face vertex indices

    Returns:
        (V,) float32 mean curvature magnitudes (always ≥ 0)
    """
    N  = len(verts)
    v0 = verts[faces[:, 0]].astype(np.float64)
    v1 = verts[faces[:, 1]].astype(np.float64)
    v2 = verts[faces[:, 2]].astype(np.float64)

    # Edge vectors originating at each corner
    e01, e02 = v1 - v0, v2 - v0
    e10, e12 = v0 - v1, v2 - v1
    e20, e21 = v0 - v2, v1 - v2

    def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cotangent of angle between vectors a and b (per-face array)."""
        cross_sq = (np.cross(a, b) ** 2).sum(-1)
        cos_val  = (a * b).sum(-1)
        return cos_val / (np.sqrt(cross_sq) + 1e-8)

    cot0 = _cot(e01, e02)   # cot at corner 0 (across from edge 1–2)
    cot1 = _cot(e10, e12)   # cot at corner 1 (across from edge 0–2)
    cot2 = _cot(e20, e21)   # cot at corner 2 (across from edge 0–1)

    # Laplacian of positions (Σ_j cot_weights × (x_j - x_i))
    L = np.zeros((N, 3), dtype=np.float64)
    np.add.at(L, faces[:, 0], cot2[:, None] * (v1 - v0) + cot1[:, None] * (v2 - v0))
    np.add.at(L, faces[:, 1], cot2[:, None] * (v0 - v1) + cot0[:, None] * (v2 - v1))
    np.add.at(L, faces[:, 2], cot1[:, None] * (v0 - v2) + cot0[:, None] * (v1 - v2))

    # Mixed area ≈ one third of sum of incident triangle areas
    face_areas = 0.5 * np.linalg.norm(np.cross(e01, e02), axis=-1)
    A = np.zeros(N, dtype=np.float64)
    np.add.at(A, faces[:, 0], face_areas / 3.0)
    np.add.at(A, faces[:, 1], face_areas / 3.0)
    np.add.at(A, faces[:, 2], face_areas / 3.0)

    # H = |L x| / (2 A)
    Hn = L / (2.0 * A[:, None] + 1e-8)
    return np.linalg.norm(Hn, axis=-1).astype(np.float32)


# ── Main public function ──────────────────────────────────────────────────────

def build_graph(
    protein_id: str,
    data_root: Path,
    *,
    knn_radial: int = 16,
    knn_aq: int = 32,
    knn_qq: int = 8,
    n_rbf: int = 16,
) -> HeteroData:
    """
    Build a PyG HeteroData graph for one protein.

    Query nodes are loaded from the canonical query_idx stored in the protein's
    ESP .npz (written by sample_esp via curvature_sampling).  This guarantees
    that the same vertex subset is used across graph building, interpolation
    evaluation, and model training/testing.

    Args:
        protein_id:  e.g. "AF-Q16613-F1"
        data_root:   root of the external data directory
        knn_radial:  k for radial supplementary atom-atom kNN (bond pairs excluded)
        knn_aq:      k for atom→query edges (query-centric)
        knn_qq:      k for query→query edges
        n_rbf:       number of Gaussian RBF basis functions per edge

    Returns:
        HeteroData with node types 'atom' and 'query', and edge types
        ('atom','bond','atom'), ('atom','radial','atom'),
        ('atom','aq','query'), ('query','qq','query').
        Optional query node attributes (enabled via config.yaml features:):
          query.curvature — (N_q,)    mean curvature magnitude
          query.normal    — (N_q, 3)  unit surface normal
        data.feature_spec — dict copy of the features: config block.

    Raises:
        FileNotFoundError: if PQR, mesh, or ESP file is missing.
    """
    feat_cfg = get_config().get("features", {})
    p = ProteinPaths(protein_id, Path(data_root))

    if not p.pqr_path.exists():
        raise FileNotFoundError(f"PQR not found: {p.pqr_path}")
    if not p.mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {p.mesh_path}")
    if not p.esp_path.exists():
        raise FileNotFoundError(f"ESP not found: {p.esp_path}")

    # ── 1. Bond graph from PQR ────────────────────────────────────────────────
    log.info("%s  building bond graph", protein_id)
    (atom_xyz, atom_names, atom_resnames,
     bond_src, bond_dst, bond_orders, bond_dists, bond_set) = _build_bond_graph(p.pqr_path)

    n_atoms = len(atom_xyz)

    # ── 2. Atom feature indices ───────────────────────────────────────────────
    elements = list(guess_types(atom_names))
    atom_type_idx = np.array(
        [ELEMENT_VOCAB.get(e, N_ELEMENT_TYPES - 1) for e in elements],
        dtype=np.int64,
    )
    res_type_idx = np.array(
        [RESIDUE_VOCAB.get(r, N_RESIDUE_TYPES - 1) for r in atom_resnames],
        dtype=np.int64,
    )
    bond_count = np.zeros(n_atoms, dtype=np.int64)
    for i, j in bond_set:
        bond_count[i] += 1
        bond_count[j] += 1

    # ── 3. Mesh → query points (loaded from canonical query_idx in ESP npz) ───
    mesh     = np.load(p.mesh_path)
    verts    = mesh["verts"]
    faces    = mesh["faces"]

    esp_data  = np.load(p.esp_path)
    q_idx     = esp_data["query_idx"].astype(np.int64)
    query_xyz = verts[q_idx].astype(np.float32)
    n_query   = len(query_xyz)
    log.info("%s  loaded %d query points from esp npz", protein_id, n_query)

    # ── 4. ESP target ─────────────────────────────────────────────────────────
    query_esp = esp_data["esp_verts"][q_idx].astype(np.float32)

    # ── 5. radial supplementary kNN edges ────────────────────────────────────────────
    log.info("%s  building radial edges (k=%d)", protein_id, knn_radial)
    radial_src, radial_dst, radial_dists = _knn_radial(atom_xyz, knn_radial, bond_set)

    # ── 6. Atom→query edges (query-centric) ──────────────────────────────────
    log.info("%s  building AQ edges (k=%d)", protein_id, knn_aq)
    aq_src, aq_dst, aq_dists = _knn_bipartite(atom_xyz, query_xyz, knn_aq)

    # ── 7. Query→query edges ──────────────────────────────────────────────────
    log.info("%s  building QQ edges (k=%d)", protein_id, knn_qq)
    qq_src, qq_dst, qq_dists = _knn_self(query_xyz, knn_qq)

    # ── 8. RBF encode distances ───────────────────────────────────────────────
    bond_rbf  = _rbf_encode(bond_dists,  n_rbf, d_min=0.9,  d_max=1.8)
    radial_rbf = _rbf_encode(radial_dists, n_rbf, d_min=1.8,  d_max=8.0)
    aq_rbf   = _rbf_encode(aq_dists,   n_rbf, d_min=0.0,  d_max=12.0)
    qq_rbf   = _rbf_encode(qq_dists,   n_rbf, d_min=0.0,  d_max=8.0)

    # ── 9. Assemble HeteroData ────────────────────────────────────────────────
    data = HeteroData()

    # Atom nodes
    data["atom"].pos          = torch.tensor(atom_xyz,       dtype=torch.float)
    data["atom"].atom_type    = torch.tensor(atom_type_idx,  dtype=torch.long)
    data["atom"].residue_type = torch.tensor(res_type_idx,   dtype=torch.long)
    data["atom"].bond_count   = torch.tensor(bond_count,     dtype=torch.long)

    # Query nodes
    data["query"].pos = torch.tensor(query_xyz, dtype=torch.float)
    data["query"].y   = torch.tensor(query_esp, dtype=torch.float)

    # Optional surface geometry features (controlled by config.yaml features:)
    if feat_cfg.get("query_normal", False):
        normals = _compute_vertex_normals(verts, faces)
        data["query"].normal = torch.tensor(normals[q_idx], dtype=torch.float)
        log.info("%s  attached surface normals to query nodes", protein_id)

    if feat_cfg.get("query_curvature", False):
        curvature = _compute_mean_curvature(verts, faces)
        data["query"].curvature = torch.tensor(curvature[q_idx], dtype=torch.float)
        log.info("%s  attached mean curvature to query nodes", protein_id)

    # Covalent edges: [bond_order | rbf_dist]
    bond_attr = np.concatenate([bond_orders[:, None], bond_rbf], axis=1)
    data["atom", "bond", "atom"].edge_index = torch.tensor(
        np.stack([bond_src, bond_dst]), dtype=torch.long
    )
    data["atom", "bond", "atom"].edge_attr = torch.tensor(bond_attr, dtype=torch.float)

    # radial supplementary kNN edges
    data["atom", "radial", "atom"].edge_index = torch.tensor(
        np.stack([radial_src, radial_dst]), dtype=torch.long
    )
    data["atom", "radial", "atom"].edge_attr = torch.tensor(radial_rbf, dtype=torch.float)

    # Atom→query edges
    data["atom", "aq", "query"].edge_index = torch.tensor(
        np.stack([aq_src, aq_dst]), dtype=torch.long
    )
    data["atom", "aq", "query"].edge_attr = torch.tensor(aq_rbf, dtype=torch.float)

    # Query→query edges
    data["query", "qq", "query"].edge_index = torch.tensor(
        np.stack([qq_src, qq_dst]), dtype=torch.long
    )
    data["query", "qq", "query"].edge_attr = torch.tensor(qq_rbf, dtype=torch.float)

    # Metadata
    data.protein_id   = protein_id
    data.n_atoms      = n_atoms
    data.n_query      = n_query
    data.feature_spec = {k: bool(v) for k, v in feat_cfg.items()}

    log.info(
        "%s  graph built: atoms=%d  query=%d  bond=%d  radial=%d  aq=%d  qq=%d",
        protein_id, n_atoms, n_query,
        bond_src.shape[0], radial_src.shape[0], aq_src.shape[0], qq_src.shape[0],
    )

    return data
