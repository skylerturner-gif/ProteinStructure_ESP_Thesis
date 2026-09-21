"""
src/structure/seed_conformations.py

Support code for the seed-conformation pilot (SUMMER_PLAN.md's
"Conformational Sampling via AlphaFold Seeds", scoped here to a 10-protein
x 5-seed pilot run through ColabFold): deriving a FASTA sequence from an
already-processed protein's PQR file (no new UniProt/network dependency)
and converting ColabFold's PDB output back into the mmCIF format the rest
of the pipeline (`pipelines/02_run_esp_calculations.py` onward) expects.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["sequence_from_pqr", "write_fasta", "pdb_to_cif"]

# Standard 20 amino acids only -- this dataset's PQR files use canonical
# 3-letter residue names throughout (verified empirically: PDB2PQR always
# writes the base residue name, e.g. "HIS" regardless of protonation-state
# tautomer -- see src/analysis/charge_probe.py's load_parse_charge_reference
# docstring for the full tautomer-naming wrinkle this dataset already works
# around). Any residue name outside this table raises rather than silently
# producing a wrong sequence.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def sequence_from_pqr(pqr_path: Path) -> str:
    """
    Reconstruct the amino-acid sequence from a PQR file's ATOM records, in
    resid order, one residue per unique resid. Avoids a new UniProt/network
    dependency for ColabFold input -- the PQR is already on disk and is the
    exact structure the rest of the pipeline treats as ground truth.

    Raises ValueError on any non-standard residue name (this dataset's PQR
    files only ever contain the 20 canonical residues -- see THREE_TO_ONE's
    docstring note -- so an unexpected name means something upstream
    changed and should not be silently mapped to the wrong letter).
    """
    seen: dict[int, str] = {}
    order: list[int] = []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            resname, resid = fields[3], int(fields[4])
            if resid not in seen:
                seen[resid] = resname
                order.append(resid)

    letters = []
    for resid in order:
        resname = seen[resid]
        if resname not in THREE_TO_ONE:
            raise ValueError(
                f"{pqr_path}: unrecognized residue name {resname!r} at resid {resid} "
                "-- refusing to guess a one-letter code."
            )
        letters.append(THREE_TO_ONE[resname])
    return "".join(letters)


def write_fasta(protein_id: str, sequence: str, fasta_path: Path) -> None:
    """Write a single-sequence FASTA file for ColabFold input."""
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(f">{protein_id}\n{sequence}\n")


def pdb_to_cif(pdb_path: Path, cif_path: Path) -> None:
    """
    Convert a PDB structure file to mmCIF via gemmi -- the inverse of
    src/electrostatics/run_pdb2pqr.py's existing CIF -> temp-PDB
    conversion, using the same library so both directions stay consistent.
    """
    import gemmi

    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    cif_path.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(cif_path))
