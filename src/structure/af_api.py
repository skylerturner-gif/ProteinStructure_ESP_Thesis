"""
src/structure/af_api.py

AlphaFold API client and structure downloader.

Reads a .txt file of UniProt accession IDs (one per line), queries the
AlphaFold EBI API for all predicted structure fragments, downloads each
fragment's mmCIF, PAE, and confidence data, converts mmCIF to PDB via
gemmi, and initializes per-fragment metadata JSONs.

API endpoint:
    GET https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}

Downloads per fragment (e.g. F1, F2, ...):
    {protein_id}.cif       — mmCIF structure file (canonical structure)
    {protein_id}_pae.json  — predicted aligned error matrix (from API)

Metadata fields created:
    protein_id          — AlphaFold model ID (e.g. AF-Q16613-F1)
    uniprot_id          — UniProt accession ID
    fragment            — fragment number (1, 2, ...)
    protein_name        — protein description from AlphaFold
    organism            — source organism
    sequence_length     — number of residues in this fragment
    plddt_mean          — mean per-residue pLDDT score
    plddt_median        — median per-residue pLDDT score
    plddt_per_residue   — full per-residue pLDDT array (list of floats)
    af_model_version    — AlphaFold model version string

Usage (from a script):
    from src.structure.af_api import download_structures
    download_structures(
        id_file=Path("data/protein_ids.txt"),
        data_root=Path("external_data"),
    )
"""

import json
import time
from pathlib import Path

import numpy as np
import requests

from src.utils.helpers import get_logger, ensure_dirs
from src.utils.io import create_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)

# ── AlphaFold API ─────────────────────────────────────────────────────────────
AF_API_BASE     = "https://alphafold.ebi.ac.uk/api/prediction"
REQUEST_TIMEOUT = 30   # seconds per HTTP request
RETRY_DELAY     = 5    # seconds to wait before retrying after a failure


# ── ID file reader ────────────────────────────────────────────────────────────

def read_uniprot_ids(id_file: Path) -> list[str]:
    """
    Read UniProt accession IDs from a plain text file, one per line.
    Blank lines and lines starting with '#' are ignored.

    Args:
        id_file: path to the .txt file of UniProt IDs

    Returns:
        list of UniProt accession strings (e.g. ["Q16613", "P12345"])
    """
    ids = []
    with open(id_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    log.info("Read %d UniProt IDs from %s", len(ids), id_file)
    return ids


# ── API query ─────────────────────────────────────────────────────────────────

def _fetch_all_fragments(uniprot_id: str) -> list[dict]:
    """
    Query the AlphaFold API for a UniProt ID and return all fragment entries.

    For proteins with multiple fragments (very long sequences) the API returns
    one entry per fragment (F1, F2, ...).  All fragments are returned so that
    each can be downloaded and processed independently.

    Returns:
        list of API metadata dicts, one per fragment, or [] on failure.
    """
    url = f"{AF_API_BASE}/{uniprot_id}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            log.warning("UniProt ID not found in AlphaFold DB: %s", uniprot_id)
        else:
            log.error("HTTP error for %s: %s", uniprot_id, e)
        return []
    except requests.exceptions.RequestException as e:
        log.error("Network error for %s: %s", uniprot_id, e)
        return []

    results = response.json()
    if not results:
        log.warning("Empty API response for %s", uniprot_id)
        return []

    if len(results) > 1:
        log.info("%s has %d fragments", uniprot_id, len(results))

    return results


# ── File downloaders ──────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, plog) -> bool:
    """
    Download a file from url and save to dest.

    Returns:
        True on success, False on failure.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        dest.write_bytes(response.content)
        plog.info("Downloaded → %s", dest.name)
        return True
    except requests.exceptions.RequestException as e:
        plog.error("Failed to download %s: %s", url, e)
        return False



# ── pLDDT extraction ──────────────────────────────────────────────────────────

def _extract_plddt(
    api_entry: dict,
    plog,
) -> tuple[list[float], float, float] | None:
    """
    Download and parse per-residue pLDDT scores from the plddtDocUrl.
    Mean pLDDT is also available as globalMetricValue in the main response
    and is used as a fallback if the download fails.

    Returns:
        (plddt_list, mean, median) or None if unavailable.
    """
    plddt_mean_api = api_entry.get("globalMetricValue")

    plddt_url = api_entry.get("plddtDocUrl")
    if not plddt_url:
        plog.warning("No plddtDocUrl in API response.")
        if plddt_mean_api is not None:
            plog.info("Using globalMetricValue as plddt_mean: %.2f", plddt_mean_api)
            return [], float(plddt_mean_api), None
        return None

    try:
        response = requests.get(plddt_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        plddt_data = response.json()
    except Exception as e:
        plog.warning("Failed to download pLDDT JSON: %s", e)
        if plddt_mean_api is not None:
            return [], float(plddt_mean_api), None
        return None

    plddt_list = plddt_data.get("confidenceScore") or plddt_data.get("plddt")
    if not plddt_list:
        plog.warning("No confidenceScore found in pLDDT JSON.")
        if plddt_mean_api is not None:
            return [], float(plddt_mean_api), None
        return None

    arr    = np.array(plddt_list, dtype=float)
    mean   = float(np.mean(arr))
    median = float(np.median(arr))
    return plddt_list, mean, median


# ── Per-fragment downloader ───────────────────────────────────────────────────

def _download_fragment(api_entry: dict, data_root: Path) -> bool:
    """
    Download all data for one AlphaFold fragment and initialize its metadata JSON.

    Args:
        api_entry: single fragment dict from the AlphaFold API response
        data_root: root of the external data directory

    Returns:
        True on success, False on any failure (logged and skipped).
    """
    protein_id = api_entry.get("entryId")
    uniprot_id = api_entry.get("uniprotAccession", "")
    if not protein_id:
        log.error("No entryId in API entry: %s", api_entry)
        return False

    # Extract fragment number from entryId (e.g. "AF-Q16613-F2" → 2)
    try:
        fragment = int(protein_id.rsplit("-F", 1)[1])
    except (IndexError, ValueError):
        fragment = 1

    p    = ProteinPaths(protein_id, data_root)
    p.ensure_dirs()
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)
    plog.info("── Processing %s (fragment %d) ──", uniprot_id, fragment)

    # Download mmCIF
    cif_url = api_entry.get("cifUrl")
    if not cif_url:
        plog.error("No CIF URL in API response")
        return False

    if not _download_file(cif_url, p.cif_path, plog):
        return False

    # Download PAE JSON
    pae_url   = api_entry.get("paeDocUrl")
    pae_saved = False
    if pae_url and pae_url.endswith(".json"):
        pae_saved = _download_file(pae_url, p.pae_path, plog)
    else:
        plog.warning("No PAE JSON URL found — skipping PAE download.")

    # Extract pLDDT
    plddt_result = _extract_plddt(api_entry, plog)
    if plddt_result is None:
        plddt_list, plddt_mean, plddt_median = [], None, None
    else:
        plddt_list, plddt_mean, plddt_median = plddt_result

    # Create metadata JSON
    metadata = {
        "uniprot_id"        : uniprot_id,
        "fragment"          : fragment,
        "protein_name"      : api_entry.get("uniprotDescription", ""),
        "organism"          : api_entry.get("organismScientificName", ""),
        "sequence_length"   : api_entry.get("sequenceEnd"),
        "af_model_version"  : api_entry.get("latestVersion", ""),
        "plddt_mean"        : round(plddt_mean, 4) if plddt_mean is not None else None,
        "plddt_median"      : round(plddt_median, 4) if plddt_median is not None else None,
        "plddt_per_residue" : plddt_list,
    }

    try:
        create_metadata(protein_id, data=metadata, data_root=data_root)
    except FileExistsError:
        plog.warning("Metadata already exists — skipping creation.")

    plog.info("SUCCESS %s → %s", uniprot_id, protein_id)
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def find_downloaded_protein_ids(uniprot_id: str, data_root: Path) -> list[str]:
    """
    Search data_root for all protein directories matching a UniProt ID.
    Returns a list of protein_ids (directory names) for which a PDB file exists.

    Args:
        uniprot_id: UniProt accession ID (e.g. "Q16613")
        data_root:  root of the external data directory

    Returns:
        Sorted list of protein_id strings (e.g. ["AF-Q16613-F1", "AF-Q16613-F2"]).
    """
    found = []
    for protein_dir in Path(data_root).iterdir():
        if protein_dir.is_dir() and uniprot_id in protein_dir.name:
            if (protein_dir / "structure" / f"{protein_dir.name}.pdb").exists():
                found.append(protein_dir.name)
    return sorted(found)


def find_downloaded_protein_id(uniprot_id: str, data_root: Path) -> str | None:
    """
    Return the first downloaded protein_id for a UniProt ID, or None.
    Convenience wrapper around find_downloaded_protein_ids for single-fragment use.
    """
    ids = find_downloaded_protein_ids(uniprot_id, data_root)
    return ids[0] if ids else None


def download_protein(uniprot_id: str, data_root: Path) -> bool:
    """
    Download AlphaFold structures for all fragments of a UniProt ID.

    Queries the AlphaFold API, downloads mmCIF and PAE files for every
    available fragment (F1, F2, ...), converts each to PDB, and initializes
    per-fragment metadata JSONs.

    Args:
        uniprot_id: UniProt accession ID (e.g. "Q16613")
        data_root:  root of the external data directory

    Returns:
        True if at least one fragment was successfully downloaded.
    """
    fragments = _fetch_all_fragments(uniprot_id)
    if not fragments:
        log.warning("── Skipping %s (API query failed or no fragments) ──", uniprot_id)
        return False

    results = [_download_fragment(entry, Path(data_root)) for entry in fragments]
    return any(results)


def download_structures(id_file: Path, data_root: Path) -> dict:
    """
    Download AlphaFold structures for all UniProt IDs in a text file.

    Reads UniProt IDs from id_file, queries the AlphaFold API for each,
    downloads all fragment mmCIF and PAE files, converts to PDB, and
    initializes per-fragment metadata JSONs under data_root.

    Failed proteins are logged and skipped — the pipeline continues.

    Args:
        id_file:   path to .txt file with one UniProt ID per line
        data_root: root of the external data directory

    Returns:
        dict with keys:
            "success": list of uniprot_ids where at least one fragment succeeded
            "failed":  list of uniprot_ids where all fragments failed
    """
    data_root = Path(data_root)
    ensure_dirs(data_root)

    uniprot_ids = read_uniprot_ids(id_file)
    if not uniprot_ids:
        log.warning("No UniProt IDs found in %s", id_file)
        return {"success": [], "failed": []}

    results = {"success": [], "failed": []}

    log.info("══ AlphaFold download: %d proteins ══", len(uniprot_ids))
    t_total = time.perf_counter()

    for uniprot_id in uniprot_ids:
        ok = download_protein(uniprot_id, data_root)
        if ok:
            results["success"].append(uniprot_id)
        else:
            results["failed"].append(uniprot_id)
        time.sleep(0.5)

    elapsed = time.perf_counter() - t_total
    log.info("══ Download complete: %d succeeded, %d failed  (%.1f s total) ══",
             len(results["success"]), len(results["failed"]), elapsed)

    if results["failed"]:
        log.warning("Failed UniProt IDs: %s", ", ".join(results["failed"]))

    return results
