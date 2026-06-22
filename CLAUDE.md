# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether heterogeneous graph neural networks can learn geometry-conditioned approximations of protein electrostatic potential (ESP) fields from AlphaFold-predicted structures.

**Full pipeline stages:**
1. Download AlphaFold structures via EBI API
2. Compute partial charges and radii with PDB2PQR (PARSE force field, pH 7.0)
3. Compute ESP grids with APBS
4. Generate Solvent-Excluded Surface (SES) meshes via MSMS
5. Sample ESP values onto surface mesh vertices; curvature-subsample ~5% as query nodes
6. Pre-build and cache heterogeneous PyG graphs (atom + query nodes)
7. Train DistanceESPN or AttentionESPN to predict ESP at query nodes

**Two trained model architectures:**
- **DistanceESPN** — geometry encoded via RBF edge features only, mean-aggregation AQ message passing
- **AttentionESPN** — multi-head cross-attention AQ with RBF geometry bias; best current results

## Setup

Two conda environments are required — one per pipeline phase:

```bash
# Stages 1–5: data generation (PDB2PQR, APBS, MSMS — no PyTorch)
conda env create -f environment.yml
conda activate protein_esp
pip install -e .

# Stages 6–7: graph building and model training (PyTorch + PyG, GPU)
conda env create -f pyg_environment.yml
conda activate pyg_env
pip install -e .

# Machine-specific config (executables, data root, log file) — required before running anything
cp config.template.yaml config.yaml
# Edit config.yaml: paths.data_root, paths.log_file, paths.pdb2pqr, paths.apbs, paths.msms
```

`config.yaml` is git-ignored and must be created on each machine.

## Running the Pipeline

```bash
# Full end-to-end (handles env switching internally)
python pipelines/full_pipeline.py --id-file data/test_ids.txt

# Data generation only (stages 1–5, protein_esp env)
python pipelines/data_gen_pipeline.py --id-file data/test_ids.txt --workers 4

# Modeling only (stages 6–7, pyg_env)
python pipelines/model_pipeline.py --id-file data/test_ids.txt

# Individual stage scripts
python pipelines/01_download_structures.py --id-file data/test_ids.txt
python pipelines/02_run_esp_calculations.py --filter --min-sequence-length 100
python pipelines/03_generate_surface.py --all
python pipelines/04_sample_esp.py --filter --min-plddt 70
python pipelines/05_evaluate_esp.py --all
python pipelines/06_build_graphs.py --all
python pipelines/07_train.py --model attention --config configs/default.yaml

# Fetch candidate UniProt IDs for dataset expansion
python scripts/fetch_uniprot_ids.py --output data/candidate_ids.txt

# WandB hyperparameter sweeps
python pipelines/run_sweep.py --sweep sweeps/ablation_example.yaml
```

Protein filtering flags available on stage scripts: `--all`, `--filter --min-sequence-length N`, `--filter --min-plddt F`, `--filter --min-surface-area F`, and their `--max-*` counterparts.

## Architecture

### Data flow

All persistent per-protein data lives in `<data_root>/<protein_id>/` with subdirectories:
- `structure/` — `.cif`, `.pqr`, `.in`, `_pae.json`
- `electrostatics/` — `.dx` APBS grid (optional, processed in memory by default)
- `mesh/` — `_mesh.npz`, `_mesh.vtk`
- `esp/` — `_esp.npz` (mesh vertices + ESP values + `query_idx` for sparse subset)
- `graph/` — `_graph.pt` cached PyG HeteroData object
- `logs/` — per-protein processing log
- `<protein_id>_metadata.json` — accumulates fields across all pipeline stages

Metadata fields include: pLDDT stats, atom count, net charge, mesh vertex count, surface area, RBF baseline Pearson r and RMSE, per-stage timing, pipeline completion status.

### PQR-only pipeline

Meshes and graph vertices are built from PQR files only (hydrogens included via PDB2PQR). The PDB mesh track (no hydrogens) was dropped. All downstream graph and model code assumes PQR-derived structures.

### Graph structure

Heterogeneous PyG `HeteroData` with two node types and four edge types:
- Node types: `atom` (all heavy atoms + H), `query` (curvature-sampled ~5% mesh vertices)
- Edge types: `bond` (covalent bonds), `radial` (atom–atom kNN), `aq` (atom→query kNN=32), `qq` (query–query kNN)
- All edge features: RBF-encoded distances; atom nodes additionally carry element, residue, and bond-type embeddings
- Graphs cached as `.pt` files; `DynamicBatchSampler` packs proteins by edge budget to fit VRAM

### Key utilities

- **`src/utils/paths.py` — `ProteinPaths`**: Single source of truth for all per-protein file paths. Construct with `ProteinPaths(protein_id, data_root)`.
- **`src/utils/config.py`**: Cached YAML config loader. Use `get_config()`, `get_data_root()`, `get_log_file()`.
- **`src/utils/io.py`**: Thread-safe `create_metadata()`, `update_metadata()`, `load_metadata()`.
- **`src/utils/helpers.py`**: Logger setup (`get_logger`, `get_pipeline_logger`), `_Timer` context manager, `notify`.
- **`src/utils/filter.py`**: `get_protein_ids()` resolves `--all`/`--filter` CLI flags to a list of protein IDs.
- **`src/utils/parallel.py`**: `run_parallel()` for process/thread pool execution across proteins.

### Models

- **`src/models/egnn.py`**: Shared building blocks — `_mlp`, `AtomEncoder`, `QueryEncoder`, `MessageLayer`, `_AtomMP`, `_QueryRefine`. Vocabularies: `N_ELEMENT_TYPES`, `N_RESIDUE_TYPES`.
- **`src/models/distance_espn.py` — `DistanceESPN`**: Stage 1 (atom–atom bond + radial MP) → Stage 2 (AQ mean aggregation, RBF geometry) → Stage 3 (QQ refinement).
- **`src/models/attention_espn.py` — `AttentionESPN`**: Same stages but Stage 2 uses multi-head cross-attention with RBF geometry bias instead of mean aggregation.
- **`src/training/trainer.py`**: Training loop, validation, checkpointing (`best_model.pt`, `latest_model.pt`), metric tracking.
- **`src/training/loss.py`**: `ESPLoss` combining MSE + per-protein Pearson correlation.

### Analysis scripts

- **`scripts/analyze_model.py`**: Training curves, vertex/protein-level ESP parity plots, PyVista comparison renders.
- **`scripts/analyze_embeddings.py`**: Atom-type embedding cosine similarity heatmaps, attention weight statistics, cross-model embedding comparison.
- **`scripts/probe_charges.py`**: Trains frozen-backbone MLP to predict per-atom PARSE partial charges from model embeddings — tests whether model encodes chemistry.
- **`scripts/pyvista_visual.py`**: Interactive PyVista surface viewer (ground truth or predicted vs ground truth).
- **`scripts/fetch_uniprot_ids.py`**: Queries UniProt REST API to fetch candidate accession IDs for dataset expansion (filters: reviewed, AlphaFold available, no membrane/metal-binding proteins, length 50–1000).

### Logging convention

Modules use file-only logging (no terminal output during processing); single-line terminal notifications print on completion or failure. Configure log level in `config.yaml` → `logging.level`.

## No Tests Yet

`tests/` contains only a placeholder. No test runner is configured.
