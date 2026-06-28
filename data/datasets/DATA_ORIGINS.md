# Data Origins

This document describes how proteins were sourced, processed, and partitioned across the three dataset directories used in this project.

---

## Directory Overview

| Directory | Proteins | Purpose |
|---|---|---|
| `~/external_protein_data/` | 8,461 | Main dataset — train / val / test |
| `~/pdb_comparison_data/` | 82 (41 AF + 41 PDB) | AF vs. experimental structure comparison |
| `~/rejected_dataset_proteins/` | 7,901 | All proteins removed from active datasets |

Summarized TSV logs for each: [main_dataset.tsv](main_dataset.tsv), [pdb_analysis.tsv](pdb_analysis.tsv), [rejected_data.tsv](rejected_data.tsv).

---

## Main Dataset (`external_protein_data/`)

### 1. Candidate pool — UniProt query

All candidates were drawn from UniProt SwissProt (reviewed entries only) via the UniProt REST API (`scripts/fetch_uniprot_ids.py`, `scripts/fetch_stratified_ids.py`, `scripts/build_nonredundant_pool.py`). The query applied at time of fetch:

```
reviewed:true
AND database:alphafolddb
AND length:[50 TO 1000]
AND NOT keyword:KW-0472   # membrane proteins
AND NOT keyword:KW-0479   # metal-binding proteins
```

**Why these filters:** membrane proteins require lipid bilayer context that APBS cannot model in implicit solvent; metal-binding proteins have coordination geometry that the PARSE force field does not parameterize. The 50–1000 residue window captures the majority of globular soluble proteins while staying within MSMS mesh and GPU memory limits.

Candidates were fetched in stratified 50-residue bins (50–99, 100–149, …, 950–1000) with up to ~600 IDs per bin, then shuffled within each bin to break UniProt's default annotation-score ordering (which over-represents *E. coli* and *H. sapiens* model organisms).

### 2. Pipeline processing

Each candidate was run through the full data-generation pipeline:

1. **Download AlphaFold structure** — CIF from EBI; PAE JSON saved alongside
2. **PDB2PQR** — PARSE force field, pH 7.0, with proton placement (hydrogens included)
3. **APBS** — linear Poisson-Boltzmann solver, solvent-excluded surface boundary conditions
4. **MSMS** — solvent-excluded surface mesh built from the PQR file
5. **ESP sampling** — APBS grid interpolated onto mesh vertices; curvature-subsampled ~5% designated as query nodes
6. **Graph build** — heterogeneous PyG HeteroData cached to disk

All meshes and graph vertices are derived from PQR files (with hydrogens and physico-chemically correct per-atom radii). The earlier PDB mesh track (no hydrogens) was dropped.

### 3. Rejection and pruning

Proteins were moved to `rejected_dataset_proteins/` under five distinct criteria. The move log (`data/redundant_moved_log.tsv`) records each protein, its reason, and the timestamp.

| Reason | Count | Criterion |
|---|---|---|
| Redundant (MMseqs2) | ~5,975 | Sequence identity ≥ 30% to another protein in the cluster (bidirectional 80% coverage); shorter/non-representative member removed |
| Outlier — charge or length | ~520 total | `\|net_charge\|` > 50 at pH 7.0 (PARSE), or sequence length > 1000, or interpolation Pearson r below threshold, or extreme ESP range |
| Incomplete pipeline | ~75 | `pipeline_complete` flag absent or `False` in metadata JSON (a pipeline stage failed or was skipped) |
| Surplus PDB structures | ~1,331 | PDB chain structures processed with `--process-all` during comparison-set preparation but not selected into the final 41-pair set |
| Unknown origin | ~45 | Numeric-format protein IDs (`AF-0000000000...`); believed to be from an early data-generation batch before standard `AF-<UniProt>-F1` naming was adopted. Origin is unconfirmed. |

**MMseqs2 clustering detail:** `easy-cluster` was run at 30% minimum sequence identity with 80% bidirectional coverage (cov-mode 0). For each cluster, the longest member was kept to preserve length-bin diversity. Non-representative members were moved to rejected.

**Outlier detail:** multiple criteria were applied; any protein failing at least one was moved:
- sequence length > 1000 aa (`find_outliers.py`)
- `|net_charge|` > 50 at pH 7.0 (`find_outliers.py`)
- interpolation Pearson r below the quality threshold (`find_esp_pearson_outliers.py`)
- ESP value range exceeding ±100 kT/e (`find_esp_pearson_outliers.py`)

### 4. Resulting dataset

After all rejection passes, 8,461 proteins remain in `external_protein_data/`. Sequence length distribution:

| Length bin | Count |
|---|---|
| 50–99 | 583 |
| 100–199 | 1,468 |
| 200–299 | 1,365 |
| 300–399 | 1,169 |
| 400–499 | 1,014 |
| 500–599 | 807 |
| 600–699 | 702 |
| 700–799 | 536 |
| 800–899 | 360 |
| 900–1000 | 440 |
| < 50 (legacy) | 17 |

Median sequence length: 364 aa. All proteins are AlphaFold v6 models (with some v1 legacy entries in the rejected set).

### 5. Train / val / test split

Not yet finalized. Planned: 80/10/10 stratified by sequence length bin, possibly with an additional stratification axis. The split assignment will be recorded in the master dataset CSV once finalized.

---

## PDB Comparison Dataset (`pdb_comparison_data/`)

This set supports RQ2: *Do differences between AF-predicted and experimentally-determined structures produce localized ESP differences that the model fails to predict?*

It is **never used for training**. Both the AF and PDB versions of each protein are processed through the identical APBS/MSMS pipeline so that ESP fields are directly comparable.

### Pair selection

Starting from the non-redundant AF proteins in `external_protein_data/`, PDB crystal structure counterparts were identified and filtered (`scripts/fetch_af_pdb_pairs.py`):

**AF-side filters (applied to candidate AF proteins):**
- Mean pLDDT ≥ 80.0

**PDB-side filters (applied to candidate crystal structures):**
- Resolution ≤ 2.0 Å
- R-factor ≤ 0.25
- PDB chain coverage ≥ 95% of UniProt sequence length
- X-ray crystallography only (`source: pdb_xray`)

**Pair ranking:** coverage DESC → Cα RMSD ASC → resolution ASC → mean pLDDT DESC → mean PAE ASC. The best PDB structure per unique AF protein was kept.

**Mesh quality filter:** SES surface area ratio (PDB/AF) and vertex count ratio both required to be within ±5% (0.95–1.05), ensuring mesh geometry is comparable between the AF and PDB structures. This filter was applied after running PDB2PQR + MSMS on all candidate PDB chains (`--process-all` mode).

**Length-stratified selection:** final pairs were chosen to span the length distribution:
- 10 pairs from 0–200 aa
- 10 pairs from 200–400 aa
- 10 pairs from 400–600 aa
- 10 pairs from 600–800 aa
- 1 pair from 800–1000 aa

This yielded **41 paired proteins** (41 AF structures + 41 PDB chain structures = 82 total entries).

### Directory structure

```
pdb_comparison_data/
├── af/         — AlphaFold structures for the 41 selected proteins
│                 (same format as external_protein_data/)
└── pdb/        — PDB chain structures (PDB-<UniProt>-<PDB_ID>-<chain> naming)
                  metadata includes: accession, pdb_id, chain, af_protein_id,
                  sequence_length, ca_rmsd_vs_af, resolution_A, source
```

The 1,331 PDB chain structures that were processed but not selected are stored in `rejected_dataset_proteins/` for potential future use.

---

## Rejected Dataset (`rejected_dataset_proteins/`)

Contains all proteins that were downloaded and processed but removed from active datasets. Retained rather than deleted to allow:
- Post-hoc analysis of why proteins failed
- Reprocessing under revised criteria
- Potential future study of redundant homologs or outlier cases

See `data/redundant_moved_log.tsv` for a complete record of moves with reasons and timestamps.

### Breakdown by reason

| Category | Count (approx.) |
|---|---|
| Redundant (sequence identity ≥ 30%) | 5,975 |
| Outlier (charge, length, ESP quality) | 520 |
| Incomplete pipeline | 75 |
| Surplus PDB candidate structures | 1,331 |
| Unknown / legacy naming | 45 |
| **Total** | **7,946** |

Note: totals may not sum exactly to 7,901 due to rounding and proteins that may have been moved before the logging system was in place.

---

## Summary of Processing Scripts

| Script | Role |
|---|---|
| `scripts/fetch_uniprot_ids.py` | Fetch flat list of UniProt candidates with broad filters |
| `scripts/fetch_stratified_ids.py` | Fetch candidates stratified by length bin |
| `scripts/build_nonredundant_pool.py` | End-to-end: fetch → cluster → dedup vs. existing → stratified output |
| `scripts/separate_redundant_proteins.py` | MMseqs2 cluster existing data root; move non-representatives to rejected |
| `scripts/find_outliers.py` | Flag proteins by sequence length or net charge |
| `scripts/find_esp_pearson_outliers.py` | Flag proteins by interpolation Pearson r or extreme ESP range |
| `scripts/reject_incomplete.py` | Move proteins missing `pipeline_complete: True` |
| `scripts/fetch_af_pdb_pairs.py` | Identify and rank AF-PDB candidate pairs |
| `scripts/select_and_prepare_pdb_pairs.py` | Run PDB2PQR + MSMS on PDB candidates; apply mesh quality filter; select final 41 pairs |
