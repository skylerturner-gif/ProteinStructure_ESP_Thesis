"""
src/structure/af_api.py

AlphaFold API client and structure downloader.

Reads a .txt file of UniProt accession IDs (one per line), downloads
the predicted structure and confidence data for each from the AlphaFold
EBI API, converts mmCIF to PDB via gemmi, and initializes the per-protein
metadata JSON.

API endpoint:
    GET https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}

Downloads per protein:
    {protein_id}.cif       — mmCIF structure file
    {protein_id}.pdb       — PDB format (converted from mmCIF via gemmi)
    {protein_id}_pae.json  — predicted aligned error matrix (from API)

Metadata fields created:
    protein_id          — AlphaFold model ID (e.g. AF-Q16613-F1-model_v4)
    uniprot_id          — UniProt accession ID
    protein_name        — protein description from AlphaFold
    organism            — source organism
    sequence_length     — number of residues
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

import gemmi
import numpy as np
import requests

from src.utils.helpers import get_logger, ensure_dirs
from src.utils.io import create_metadata
from src.utils.paths import ProteinPaths

log = get_logger(__name__)

# ── AlphaFold API ─────────────────────────────────────────────────────────────
AF_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
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

def _fetch_api_metadata(uniprot_id: str) -> dict | None:
    """
    Query the AlphaFold API for a UniProt ID and return the F1 fragment entry.

    The API returns a list of fragments. For proteins with multiple fragments
    (very long sequences) we take only F1. Most proteins have a single entry.

    Returns:
        dict of API metadata fields for F1, or None on failure.
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
        return None
    except requests.exceptions.RequestException as e:
        log.error("Network error for %s: %s", uniprot_id, e)
        return None

    results = response.json()
    if not results:
        log.warning("Empty API response for %s", uniprot_id)
        return None

    # Filter for F1 fragment only
    f1_entries = [r for r in results if r.get("modelEntityId", "").endswith("-F1")]
    if not f1_entries:
        log.warning("No F1 fragment found for %s — taking first result", uniprot_id)
        return results[0]

    if len(results) > 1:
        log.info("%s has %d fragments — using F1 only", uniprot_id, len(results))

    return f1_entries[0]


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


# ── mmCIF → PDB conversion ────────────────────────────────────────────────────

def _convert_cif_to_pdb(cif_path: Path, pdb_path: Path, plog) -> bool:
    """
    Convert a mmCIF structure file to PDB format using gemmi.

    Returns:
        True on success, False on failure.
    """
    try:
        structure = gemmi.read_structure(str(cif_path))
        structure.write_pdb(str(pdb_path))
        plog.info("Converted CIF → PDB: %s", pdb_path.name)
        return True
    except Exception as e:
        plog.error("CIF to PDB conversion failed: %s", e)
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


# ── Per-protein downloader ────────────────────────────────────────────────────

def _download_protein(uniprot_id: str, data_root: Path) -> bool:
    """
    Download all data for one protein and initialize its metadata JSON.

    Args:
        uniprot_id: UniProt accession ID (e.g. "Q16613")
        data_root:  root of the external data directory

    Returns:
        True on success, False on any failure (logged and skipped).
    """
    # Step 1: Query API (we need protein_id before we can set up paths/logger)
    api_entry = _fetch_api_metadata(uniprot_id)
    if api_entry is None:
        log.warning("── Skipping %s (API query failed) ──", uniprot_id)
        return False

    protein_id = api_entry.get("entryId")
    if not protein_id:
        log.error("No entryId in API response for %s", uniprot_id)
        return False

    # Set up per-protein paths and logger
    p    = ProteinPaths(protein_id, data_root)
    p.ensure_dirs()
    plog = get_logger(f"protein.{protein_id}", log_file=p.log_path)
    plog.info("── Processing %s ──", uniprot_id)

    # Step 2: Download mmCIF
    cif_url = api_entry.get("cifUrl")
    if not cif_url:
        plog.error("No CIF URL in API response")
        return False

    if not _download_file(cif_url, p.cif_path, plog):
        return False

    # Step 3: Convert to PDB
    if not _convert_cif_to_pdb(p.cif_path, p.pdb_path, plog):
        return False

    # Step 4: Download PAE JSON
    pae_url   = api_entry.get("paeDocUrl")
    pae_saved = False
    if pae_url and pae_url.endswith(".json"):
        pae_saved = _download_file(pae_url, p.pae_path, plog)
    else:
        plog.warning("No PAE JSON URL found — skipping PAE download.")

    # Step 5: Extract pLDDT
    plddt_result = _extract_plddt(api_entry, plog)
    if plddt_result is None:
        plddt_list, plddt_mean, plddt_median = [], None, None
    else:
        plddt_list, plddt_mean, plddt_median = plddt_result

    # Step 6: Create metadata JSON
    metadata = {
        "uniprot_id"        : uniprot_id,
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

def download_protein(uniprot_id: str, data_root: Path) -> bool:
    """
    Download AlphaFold structure for a single UniProt ID.

    Queries the AlphaFold API, downloads mmCIF and PAE files, converts
    to PDB, and initializes the per-protein metadata JSON.

    Args:
        uniprot_id: UniProt accession ID (e.g. "Q16613")
        data_root:  root of the external data directory

    Returns:
        True on success, False on failure.
    """
    return _download_protein(uniprot_id, Path(data_root))

def download_structures(id_file: Path, data_root: Path) -> dict:
    """
    Download AlphaFold structures for all UniProt IDs in a text file.

    Reads UniProt IDs from id_file, queries the AlphaFold API for each,
    downloads mmCIF and PAE files, converts to PDB, and initializes
    per-protein metadata JSONs under data_root.

    Failed proteins are logged and skipped — the pipeline continues.

    Args:
        id_file:   path to .txt file with one UniProt ID per line
        data_root: root of the external data directory

    Returns:
        dict with keys:
            "success": list of protein_ids successfully processed
            "failed":  list of uniprot_ids that failed
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
        ok = _download_protein(uniprot_id, data_root)
        if ok:
            results["success"].append(uniprot_id)
        else:
            results["failed"].append(uniprot_id)
        # Brief pause to be respectful to the API
        time.sleep(0.5)

    elapsed = time.perf_counter() - t_total
    log.info("══ Download complete: %d succeeded, %d failed  (%.1f s total) ══",
             len(results["success"]), len(results["failed"]), elapsed)

    if results["failed"]:
        log.warning("Failed UniProt IDs: %s", ", ".join(results["failed"]))

    return results