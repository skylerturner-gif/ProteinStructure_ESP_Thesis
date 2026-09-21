"""
src/utils/io.py

Thread-safe per-protein metadata JSON read/write.

Each protein gets its own directory under the external data root:
    <data_root>/<protein_id>/<protein_id>_metadata.json

The metadata JSON is created once by the AlphaFold API stage and then
updated incrementally by later pipeline stages (mesh, APBS, sampling).
A filelock companion (.lock) prevents concurrent write collisions when
multiple worker processes update the same protein's metadata.
"""

import json
from pathlib import Path
from filelock import FileLock, Timeout

from src.utils.helpers import get_logger

log = get_logger(__name__)

# How long (seconds) to wait for a lock before raising Timeout
LOCK_TIMEOUT = 30


def _metadata_paths(protein_id: str, data_root: Path) -> tuple[Path, Path]:
    """Return (metadata_path, lock_path) for a given protein."""
    protein_dir = Path(data_root) / protein_id
    meta_path   = protein_dir / f"{protein_id}_metadata.json"
    lock_path   = protein_dir / f"{protein_id}_metadata.lock"
    return meta_path, lock_path


def create_metadata(protein_id: str, data: dict, data_root: Path) -> None:
    """
    Create a new metadata JSON for a protein.
    Called once by the AlphaFold API stage with initial biological fields.

    Args:
        protein_id: e.g. "AF_Q16613-1_8638_207"
        data:       dict of initial fields (uniprot_id, protein_name, etc.)
        data_root:  root of the external data directory

    Raises:
        FileExistsError: if metadata already exists (use update_metadata to modify)
        Timeout:         if the lock cannot be acquired within LOCK_TIMEOUT seconds
    """
    meta_path, lock_path = _metadata_paths(protein_id, data_root)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        if meta_path.exists():
            raise FileExistsError(
                f"Metadata already exists for {protein_id}: {meta_path}\n"
                "Use update_metadata() to modify existing records."
            )
        payload = {"protein_id": protein_id, **data}
        meta_path.write_text(json.dumps(payload, indent=2))
        log.info("Created metadata → %s", meta_path)


def update_metadata(protein_id: str, data: dict, data_root: Path) -> None:
    """
    Thread-safe update of an existing metadata JSON.
    Loads the current file, merges new fields, and writes back.
    New fields are added; existing fields are overwritten.

    Args:
        protein_id: e.g. "AF_Q16613-1_8638_207"
        data:       dict of fields to add or update
        data_root:  root of the external data directory

    Raises:
        FileNotFoundError: if metadata does not exist yet
        Timeout:           if the lock cannot be acquired within LOCK_TIMEOUT seconds
    """
    meta_path, lock_path = _metadata_paths(protein_id, data_root)

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata found for {protein_id}: {meta_path}\n"
                "Use create_metadata() to initialise a new record."
            )
        current = json.loads(meta_path.read_text())
        current.update(data)
        meta_path.write_text(json.dumps(current, indent=2))
        log.info("Updated metadata → %s  (fields: %s)", meta_path, list(data.keys()))


def is_graph_stats_contaminated(meta: dict) -> bool:
    """
    True if a protein's graph-build stats were overwritten by the
    resample_query_density.py / rebuild_graphs_for_ids.py symlink bug (see
    scripts/repair_graph_stats.py): that experiment symlinked metadata.json
    from an isolated dest-root back to the source data_root, so a later
    graph rebuild against the dest-root wrote graph stats through the
    symlink into the shared source metadata.

    Detection signature: num_query_nodes (graph-build stage) disagrees with
    n_query_nodes (ESP-sampling stage, authoritative) — confirmed
    dataset-wide to coincide exactly with num_query_nodes == n_vertices
    (the contaminating full-density graph's query count).
    """
    return meta.get("num_query_nodes") != meta.get("n_query_nodes")


def load_metadata(protein_id: str, data_root: Path) -> dict:
    """
    Read and return the metadata dict for a protein.
    No lock needed for reads (JSON writes are atomic at the OS level
    via write_text, which is a single write syscall on all major OS).

    Args:
        protein_id: e.g. "AF_Q16613-1_8638_207"
        data_root:  root of the external data directory

    Returns:
        dict of all stored metadata fields

    Raises:
        FileNotFoundError: if metadata does not exist
    """
    meta_path, _ = _metadata_paths(protein_id, data_root)
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found for {protein_id}: {meta_path}")
    return json.loads(meta_path.read_text())