# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether geometric deep learning (EGNN) can learn geometry-conditioned approximations of protein electrostatic potential (ESP) fields from AlphaFold-predicted structures.

**Pipeline stages:**
1. Download AlphaFold structures via API
2. Compute ESP grids with PDB2PQR + APBS
3. Generate Solvent-Excluded Surface (SES) meshes via MSMS
4. Sample ESP values onto surface meshes (interpolation + Laplacian reconstruction)
5. Evaluate and visualize (EGNN training is a future stage — currently stubbed)

## Setup

```bash
conda env create -f environment.yml
conda activate protein_esp
pip install -e .

# Configure machine-specific paths (executables, data root, log file)
cp config.template.yaml config.yaml
# Edit config.yaml before running anything
```

`config.yaml` (git-ignored) must define `paths.data_root` (external directory holding all generated protein data), `paths.log_file`, and paths to `pdb2pqr`, `apbs`, and `msms` executables.

## Running the Pipeline

```bash
# Full end-to-end
python pipelines/full_pipeline.py --id-file data/test_ids.txt

# Individual stages
python scripts/01_download_structure.py --id-file data/test_ids.txt
python scripts/02_run_esp_calculations.py --filter --min-sequence-length 100
python scripts/03_generate_surface.py --all
python scripts/04_sample_esp.py --filter --min-plddt 70
python scripts/05_evaluate.py --all
python scripts/06_visualize.py --protein-id AF-Q16613-F1
```

Protein filtering flags available on all scripts: `--all`, `--filter --min-sequence-length N`, `--filter --min-plddt F`, `--filter --min-surface-area F`, and their `--max-*` counterparts.

## Architecture

### Data flow

All persistent per-protein data lives in `<data_root>/<protein_id>/` with subdirectories `structure/`, `electrostatics/`, `mesh/`, `esp/`, `logs/`, and a `<protein_id>_metadata.json`. The metadata JSON accumulates fields across all pipeline stages (pLDDT, atom count, mesh stats, Pearson r, RMSE, etc.) and is written with `filelock` for thread safety.

### Key utilities

- **`src/utils/paths.py` — `ProteinPaths`**: Single source of truth for all per-protein file paths. Construct with `ProteinPaths(protein_id, data_root)`.
- **`src/utils/config.py`**: Cached YAML config loader. Use `get_config()`, `get_data_root()`, `get_log_file()`.
- **`src/utils/io.py`**: Thread-safe `create_metadata()`, `update_metadata()`, `load_metadata()`.
- **`src/utils/helpers.py`**: Logger setup (`get_logger`, `get_pipeline_logger`) and `_Timer` context manager.
- **`src/utils/filter.py`**: `get_protein_ids()` resolves `--all`/`--filter` CLI flags to a list of protein IDs.

### Logging convention

Modules use file-only logging (no terminal output during processing); single-line terminal notifications print on completion. Configure level in `config.yaml` → `logging.level`.

### ESP variants

Each protein produces 4 ESP surface files: `{pdb,pqr}_mesh_{interp,laplacian}.npz`. PDB mesh uses no hydrogens; PQR mesh includes hydrogens (from PDB2PQR). Two sampling methods: nearest-neighbor interpolation and Laplacian reconstruction.

## Incomplete / Stub Modules

The following are headers + docstrings only and need implementation:
- `src/data/dataset.py` — Dataset & DataLoader
- `src/data/graph_builder.py` — Graph construction for EGNN
- `src/data/transform.py` — Normalization, masking, augmentation
- `src/models/egnn.py` — EGNN architecture

## No Tests Yet

`tests/` contains only a placeholder markdown file. There is no test runner configured.
