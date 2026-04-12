"""
src/utils/paths.py

Centralized per-protein path construction.

All file and directory paths for a protein are defined here.
No other module should construct protein paths manually — always
import from this module instead.

Directory structure per protein:
    <data_root>/
    └── <protein_id>/
        ├── structure/          .cif, .pqr, .in, _pae.json
        ├── electrostatics/     .dx, APBS output files
        ├── mesh/               _mesh.npz, _mesh.vtk
        ├── esp/                _esp.npz
        ├── logs/               <protein_id>.log
        └── <protein_id>_metadata.json

Usage:
    from src.utils.paths import ProteinPaths
    p = ProteinPaths("AF-Q16613-F1", data_root)

    p.mesh_path   # mesh/AF-Q16613-F1_mesh.npz
    p.esp_path    # esp/AF-Q16613-F1_esp.npz
    p.ensure_dirs()     # create all subdirectories
"""

from pathlib import Path


class ProteinPaths:
    """
    All paths for a single protein, derived from protein_id and data_root.

    Attributes are lazy Path objects — no I/O is performed until you
    actually use them. Call ensure_dirs() to create all subdirectories.
    """

    def __init__(self, protein_id: str, data_root: Path):
        self.protein_id  = protein_id
        self.data_root   = Path(data_root)

        # ── Top-level protein directory ───────────────────────────────────────
        self.protein_dir = self.data_root / protein_id

        # ── Subdirectories ────────────────────────────────────────────────────
        self.structure_dir      = self.protein_dir / "structure"
        self.electrostatics_dir = self.protein_dir / "electrostatics"
        self.mesh_dir           = self.protein_dir / "mesh"
        self.esp_dir            = self.protein_dir / "esp"
        self.logs_dir           = self.protein_dir / "logs"

        # ── Metadata ──────────────────────────────────────────────────────────
        self.metadata_path = self.protein_dir / f"{protein_id}_metadata.json"
        self.metadata_lock = self.protein_dir / f"{protein_id}_metadata.lock"

        # ── Per-protein log ───────────────────────────────────────────────────
        self.log_path = self.logs_dir / f"{protein_id}.log"

        # ── Structure files ───────────────────────────────────────────────────
        self.cif_path     = self.structure_dir / f"{protein_id}.cif"
        self.pqr_path     = self.structure_dir / f"{protein_id}.pqr"
        self.apbs_in_path = self.structure_dir / f"{protein_id}.in"
        self.pae_path     = self.structure_dir / f"{protein_id}_pae.json"

        # ── Electrostatics files ──────────────────────────────────────────────
        self.dx_path        = self.electrostatics_dir / f"{protein_id}.dx"
        # dx_stem is passed to APBS — it appends .dx automatically
        self.dx_stem        = self.electrostatics_dir / protein_id

        # ── Mesh files ────────────────────────────────────────────────────────
        self.mesh_path  = self.mesh_dir / f"{protein_id}_mesh.npz"
        self.vtk_path   = self.mesh_dir / f"{protein_id}_mesh.vtk"

        # ── ESP sampled files ─────────────────────────────────────────────────
        self.esp_path = self.esp_dir / f"{protein_id}_esp.npz"

        # ── Graph cache files ─────────────────────────────────────────────────
        self.graph_dir = self.protein_dir / "graph"

    def graph_path(self) -> Path:
        """Path to the cached PyG HeteroData graph."""
        return self.graph_dir / f"{self.protein_id}_graph.pt"

    def ensure_dirs(self) -> None:
        """Create all protein subdirectories. Safe to call if they exist."""
        for d in [
            self.protein_dir,
            self.structure_dir,
            self.electrostatics_dir,
            self.mesh_dir,
            self.esp_dir,
            self.logs_dir,
            self.graph_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def esp_exists(self) -> bool:
        """Return True if the ESP sampled output file exists."""
        return self.esp_path.exists()

    def is_evaluated(self) -> bool:
        """
        Return True if ESP evaluation metrics have been written to metadata.
        Checks for pearson_r key in the metadata JSON.
        Returns False if metadata does not exist or is missing the key.
        """
        if not self.metadata_path.exists():
            return False
        try:
            import json
            meta = json.loads(self.metadata_path.read_text())
            return "pearson_r" in meta
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"ProteinPaths(protein_id={self.protein_id!r}, data_root={self.data_root!r})"
