"""
src/utils/filter.py

Protein selector — filter proteins from data_root based on metadata fields.

Scans all protein directories in data_root, loads each metadata JSON,
and returns a list of protein IDs matching the specified criteria.
Proteins with missing or malformed metadata are logged and skipped.

Supported filters:
    min_sequence_length  — minimum sequence length (inclusive)
    max_sequence_length  — maximum sequence length (inclusive)
    min_plddt            — minimum mean pLDDT score (inclusive)
    max_plddt            — maximum mean pLDDT score (inclusive)
    min_surface_area     — minimum SES area in Å² (inclusive, uses pdb mesh)
    max_surface_area     — maximum SES area in Å² (inclusive, uses pdb mesh)

All filters are optional and combinable. Passing no filters with
select_all=True returns all proteins with valid metadata.

Usage:
    from src.utils.filter import get_protein_ids

    # All proteins
    ids = get_protein_ids(data_root, select_all=True)

    # Sequence length > 100
    ids = get_protein_ids(data_root, min_sequence_length=100)

    # Sequence length > 100 AND mean pLDDT < 70
    ids = get_protein_ids(data_root, min_sequence_length=100, max_plddt=70.0)
"""

from pathlib import Path

from src.utils.helpers import get_logger
from src.utils.io import load_metadata

log = get_logger(__name__)


def get_protein_ids(
    data_root: Path,
    select_all: bool = False,
    min_sequence_length: int = None,
    max_sequence_length: int = None,
    min_plddt: float = None,
    max_plddt: float = None,
    min_surface_area: float = None,
    max_surface_area: float = None,
) -> list[str]:
    """
    Return a sorted list of protein IDs from data_root matching all
    specified filter criteria.

    Args:
        data_root:            root of the external data directory
        select_all:           if True, return all proteins with valid metadata
                              (other filters still apply if provided)
        min_sequence_length:  minimum sequence length (inclusive)
        max_sequence_length:  maximum sequence length (inclusive)
        min_plddt:            minimum mean pLDDT score (inclusive)
        max_plddt:            maximum mean pLDDT score (inclusive)
        min_surface_area:     minimum SES area in Å² (inclusive, pdb mesh)
        max_surface_area:     maximum SES area in Å² (inclusive, pdb mesh)

    Returns:
        Sorted list of protein_id strings passing all filters.

    Raises:
        ValueError: if no filters are specified and select_all is False.
    """
    data_root = Path(data_root)

    any_filter = any(f is not None for f in [
        min_sequence_length, max_sequence_length,
        min_plddt, max_plddt,
        min_surface_area, max_surface_area,
    ])

    if not select_all and not any_filter:
        raise ValueError(
            "No filters specified. Pass select_all=True to return all proteins, "
            "or provide at least one filter criterion."
        )

    # Find all protein directories (those containing a metadata JSON)
    protein_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and (d / f"{d.name}_metadata.json").exists()
    ])

    if not protein_dirs:
        log.warning("No protein directories with metadata found in %s", data_root)
        return []

    log.info("Scanning %d protein directories in %s", len(protein_dirs), data_root)

    selected = []
    skipped  = 0

    for protein_dir in protein_dirs:
        protein_id = protein_dir.name

        try:
            meta = load_metadata(protein_id, data_root)
        except FileNotFoundError:
            log.warning("No metadata for %s — skipping", protein_id)
            skipped += 1
            continue
        except Exception as e:
            log.warning("Failed to load metadata for %s: %s — skipping", protein_id, e)
            skipped += 1
            continue

        # ── Apply filters ──────────────────────────────────────────────────────
        if not _passes_filters(
            meta, protein_id,
            min_sequence_length, max_sequence_length,
            min_plddt, max_plddt,
            min_surface_area, max_surface_area,
        ):
            continue

        selected.append(protein_id)

    log.info(
        "Selected %d / %d proteins  (%d skipped due to missing/bad metadata)",
        len(selected), len(protein_dirs), skipped,
    )
    return selected


def _passes_filters(
    meta: dict,
    protein_id: str,
    min_sequence_length: int,
    max_sequence_length: int,
    min_plddt: float,
    max_plddt: float,
    min_surface_area: float,
    max_surface_area: float,
) -> bool:
    """
    Return True if a protein's metadata passes all specified filters.
    Logs a warning and returns False if a required metadata field is missing.
    """
    # ── Sequence length ────────────────────────────────────────────────────────
    if min_sequence_length is not None or max_sequence_length is not None:
        seq_len = meta.get("sequence_length")
        if seq_len is None:
            log.warning("[%s] Missing sequence_length — skipping", protein_id)
            return False
        if min_sequence_length is not None and seq_len < min_sequence_length:
            return False
        if max_sequence_length is not None and seq_len > max_sequence_length:
            return False

    # ── Mean pLDDT ────────────────────────────────────────────────────────────
    if min_plddt is not None or max_plddt is not None:
        plddt = meta.get("plddt_mean")
        if plddt is None:
            log.warning("[%s] Missing plddt_mean — skipping", protein_id)
            return False
        if min_plddt is not None and plddt < min_plddt:
            return False
        if max_plddt is not None and plddt > max_plddt:
            return False

    # ── Surface area ──────────────────────────────────────────────────────────
    if min_surface_area is not None or max_surface_area is not None:
        area = meta.get("ses_area_pdb")
        if area is None:
            log.warning("[%s] Missing ses_area_pdb — skipping", protein_id)
            return False
        if min_surface_area is not None and area < min_surface_area:
            return False
        if max_surface_area is not None and area > max_surface_area:
            return False

    return True


def add_filter_args(parser) -> None:
    """
    Add standard filter arguments to an argparse.ArgumentParser.
    Call this in each script to get consistent CLI filtering across all scripts.

    Usage in a script:
        from src.utils.filter import add_filter_args, get_protein_ids_from_args
        add_filter_args(parser)
        args = parser.parse_args()
        protein_ids = get_protein_ids_from_args(args, data_root)
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", dest="select_all",
        help="Run on all proteins with valid metadata in data_root.",
    )
    group.add_argument(
        "--filter", action="store_true", dest="use_filter",
        help="Run on proteins matching the filter criteria below.",
    )

    parser.add_argument("--min-sequence-length", type=int, default=None,
                        help="Minimum sequence length (inclusive).")
    parser.add_argument("--max-sequence-length", type=int, default=None,
                        help="Maximum sequence length (inclusive).")
    parser.add_argument("--min-plddt", type=float, default=None,
                        help="Minimum mean pLDDT score (inclusive).")
    parser.add_argument("--max-plddt", type=float, default=None,
                        help="Maximum mean pLDDT score (inclusive).")
    parser.add_argument("--min-surface-area", type=float, default=None,
                        help="Minimum SES surface area in Å² (inclusive).")
    parser.add_argument("--max-surface-area", type=float, default=None,
                        help="Maximum SES surface area in Å² (inclusive).")


def get_protein_ids_from_args(args, data_root: Path) -> list[str]:
    """
    Call get_protein_ids using parsed argparse arguments.

    Args:
        args:      parsed argparse.Namespace from a script using add_filter_args
        data_root: root of the external data directory

    Returns:
        List of matching protein IDs.
    """
    return get_protein_ids(
        data_root=data_root,
        select_all=args.select_all,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        min_plddt=args.min_plddt,
        max_plddt=args.max_plddt,
        min_surface_area=args.min_surface_area,
        max_surface_area=args.max_surface_area,
    )