# Protein Structure ESP Prediction Thesis

Research project investigating whether geometric deep learning (EGNN) can learn geometry-conditioned approximations of protein electrostatic potential (ESP) fields from AlphaFold-predicted structures.

---

## Project Structure

```
ProteinStructure_ESP_Thesis/
├── src/
│   ├── acquisition/        # AlphaFold API download
│   ├── electrostatics/     # PDB2PQR and APBS wrappers
│   ├── surface/            # SES mesh generation and ESP mapping
│   ├── data/               # Dataset, graph building, transforms (stub)
│   ├── models/             # EGNN architecture (stub)
│   ├── analysis/           # Metrics and visualization
│   └── utils/              # Logging, I/O, config, paths, filtering
├── scripts/                # Numbered pipeline entry points
├── pipelines/              # Full end-to-end orchestrator
├── data/                   # Protein ID lists and placeholders
├── outputs/                # Figures, metrics, logs
├── tests/                  # Placeholder (no tests yet)
├── config.template.yaml
├── environment.yml
└── pyproject.toml
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate protein_esp
pip install -e .

# Copy and edit machine-specific config
cp config.template.yaml config.yaml
```

`config.yaml` is git-ignored. All sections must be filled before running anything.

### Config keys (`config.yaml`)

```yaml
paths:
  data_root: /path/to/external/data       # root for all per-protein output
  log_file:  /path/to/logs/pipeline.log   # pipeline-level log file

executables:
  pdb2pqr: pdb2pqr      # full path if not on PATH
  apbs:    apbs
  msms:    msms

electrostatics:
  forcefield: PARSE     # PARSE, AMBER, CHARMM, etc.
  ph_method:  propka    # propka or pdb2pka
  ph_value:   7.0

surface:
  msms_density:  3.0    # vertices per Å²
  probe_radius:  1.4    # Å (water probe)

esp_mapping:
  normal_offset: 0.5    # Å outward offset before ESP lookup
  subsample_n:   3      # keep every Nth vertex as known for Laplacian

logging:
  level: INFO           # DEBUG, INFO, WARNING, ERROR
```

---

## Running the Pipeline

### Full end-to-end

```bash
python pipelines/full_pipeline.py --id-file data/test_ids.txt
```

Runs all six stages sequentially for each UniProt ID in the file.

### Individual stages

```bash
# 1. Download AlphaFold structures
python scripts/01_download_structure.py --id-file data/test_ids.txt

# 2. Assign charges/radii and compute ESP grid
python scripts/02_run_esp_calculations.py --all
python scripts/02_run_esp_calculations.py --filter --min-sequence-length 100

# 3. Generate SES meshes
python scripts/03_generate_surface.py --all
python scripts/03_generate_surface.py --filter --min-plddt 70

# 4. Sample ESP onto surface meshes
python scripts/04_sample_esp.py --all
python scripts/04_sample_esp.py --filter --min-plddt 70 --min-surface-area 5000

# 5. Evaluate predictions
python scripts/05_evaluate.py --all
python scripts/05_evaluate.py --all --force   # recompute even if metrics exist

# 6. Visualise a single protein
python scripts/06_visualize.py AF-Q16613-F1
python scripts/06_visualize.py AF-Q16613-F1 --clim -10 10
```

### Protein selection flags (scripts 02–05)

| Flag | Type | Description |
|------|------|-------------|
| `--all` | — | Select all proteins that have a metadata JSON |
| `--filter` | — | Select proteins matching the criteria below |
| `--min-sequence-length` | int | Minimum amino acid count |
| `--max-sequence-length` | int | Maximum amino acid count |
| `--min-plddt` | float | Minimum mean pLDDT score |
| `--max-plddt` | float | Maximum mean pLDDT score |
| `--min-surface-area` | float | Minimum SES area (Å²) |
| `--max-surface-area` | float | Maximum SES area (Å²) |

`--all` and `--filter` are mutually exclusive and one is required.

---

## Architecture

### Data flow

```
UniProt ID list
    │
    ▼  01_download_structure.py
AlphaFold mmCIF → PDB  +  pLDDT per residue  →  metadata.json
    │
    ▼  02_run_esp_calculations.py
PDB → PDB2PQR → .pqr  +  .in
                 │
                 ▼  fix_pqr_columns()  ← repairs merged coordinate columns
              APBS  →  .dx (ESP grid)
    │
    ▼  03_generate_surface.py
.pdb / .pqr → MSMS → mesh (.npz + .vtk)  [pdb and pqr variants]
    │
    ▼  04_sample_esp.py
mesh + .dx → ESP sampled onto surface (.npz)  [×4: pdb/pqr × interp/laplacian]
    │
    ▼  05_evaluate.py
sampled .npz → Pearson r + RMSE → metadata.json
```

All per-protein files live under `<data_root>/<protein_id>/` with subdirectories:
`structure/`, `electrostatics/`, `mesh/`, `esp/`, `logs/`.

### ESP variants

Each protein produces four ESP surface files:

| File | Description |
|------|-------------|
| `pdb_mesh_interp.npz` | PDB structure, nearest-neighbour interpolation |
| `pdb_mesh_laplacian.npz` | PDB structure, Laplacian reconstruction |
| `pqr_mesh_interp.npz` | PQR structure (with hydrogens), nearest-neighbour |
| `pqr_mesh_laplacian.npz` | PQR structure, Laplacian reconstruction |

### Metadata fields

`<protein_id>_metadata.json` accumulates fields across stages:

| Field | Stage | Description |
|-------|-------|-------------|
| `protein_id` | Download | AlphaFold fragment ID (e.g. AF-Q16613-F1) |
| `uniprot_id` | Download | UniProt accession |
| `protein_name` | Download | Human-readable name |
| `organism` | Download | Source organism |
| `sequence_length` | Download | Number of amino acid residues |
| `plddt_mean` | Download | Mean per-residue pLDDT |
| `plddt_median` | Download | Median per-residue pLDDT |
| `plddt_per_residue` | Download | Full per-residue pLDDT list |
| `af_model_version` | Download | AlphaFold model version string |
| `n_heavy_atoms` | PDB2PQR | Count of non-hydrogen atoms |
| `net_charge` | PDB2PQR | Sum of partial charges (elementary charge units) |
| `time_pdb2pqr_sec` | PDB2PQR | Wall-clock time for PDB2PQR |
| `time_apbs_sec` | APBS | Wall-clock time for APBS |
| `n_vertices_pdb` | Mesh | Number of SES vertices (PDB variant) |
| `n_vertices_pqr` | Mesh | Number of SES vertices (PQR variant) |
| `ses_area_pdb` | Mesh | Solvent-excluded surface area in Å² (PDB) |
| `ses_area_pqr` | Mesh | Solvent-excluded surface area in Å² (PQR) |
| `time_mesh_pdb_sec` | Mesh | Wall-clock time for PDB mesh |
| `time_mesh_pqr_sec` | Mesh | Wall-clock time for PQR mesh |
| `pearson_r_pdb` | Evaluate | Pearson r between interp and Laplacian ESP (PDB) |
| `pearson_r_pqr` | Evaluate | Pearson r (PQR) |
| `rmse_pdb` | Evaluate | RMSE between interp and Laplacian ESP (PDB) |
| `rmse_pqr` | Evaluate | RMSE (PQR) |

---

## Key Utilities

### `src/utils/paths.py` — `ProteinPaths`

Single source of truth for all per-protein file paths.

```python
from src.utils.paths import ProteinPaths
p = ProteinPaths(protein_id, data_root)
p.ensure_dirs()          # create all subdirectories
p.pqr_path               # Path to .pqr file
p.dx_path                # Path to .dx ESP grid
p.pdb_mesh_path          # Path to PDB mesh .npz
p.pqr_interp_path        # Path to PQR interpolated ESP .npz
p.is_evaluated()         # True if Pearson r metrics exist in metadata
p.all_sampled_exist()    # True if all four ESP .npz files exist
```

### `src/utils/config.py`

```python
from src.utils.config import get_config, get_data_root, get_log_file
cfg       = get_config()      # full config dict (cached)
data_root = get_data_root()   # Path to data_root
log_file  = get_log_file()    # Path to pipeline log file
```

### `src/utils/io.py`

Thread-safe metadata JSON operations (uses `filelock`):

```python
from src.utils.io import create_metadata, update_metadata, load_metadata
create_metadata(protein_id, data={"key": "val"}, data_root=...)
update_metadata(protein_id, data={"new_key": 42},  data_root=...)
meta = load_metadata(protein_id, data_root=...)
```

### `src/utils/filter.py`

```python
from src.utils.filter import get_protein_ids, add_filter_args, get_protein_ids_from_args
ids = get_protein_ids(data_root, select_all=True)
ids = get_protein_ids(data_root, min_plddt=70, min_sequence_length=100)
```

### `src/utils/helpers.py`

```python
from src.utils.helpers import get_logger, timer, notify
log = get_logger(__name__, log_file=p.log_path)
with timer() as t:
    ...
log.info("Elapsed: %.2f s", t.seconds)
notify(protein_id, "complete")   # single-line terminal output
```

---

## Source Module Reference

### `src/acquisition/af_api.py`

| Function | Description |
|----------|-------------|
| `read_uniprot_ids(id_file)` | Read UniProt IDs from a text file (one per line, `#` comments ignored) |
| `download_protein(uniprot_id, data_root)` | Download structure, pLDDT, and create metadata for one protein |
| `download_structures(id_file, data_root)` | Batch download; returns `{"success": [...], "failed": [...]}` |
| `find_downloaded_protein_id(uniprot_id, data_root)` | Find already-downloaded protein directory, returns ID string or `None` |

### `src/electrostatics/run_pdb2pqr.py`

| Function | Description |
|----------|-------------|
| `process_pdb2pqr(protein_id, data_root)` | Run PDB2PQR, fix merged columns, update metadata |
| `fix_pqr_columns(pqr_path, plog)` | Repair coordinate columns merged by PDB2PQR for large proteins (e.g. `-19.098-100.028` → `-19.098 -100.028`) |
| `count_heavy_atoms(pqr_path)` | Count non-hydrogen atoms in a .pqr file |
| `compute_net_charge(pqr_path)` | Sum partial charges across all ATOM/HETATM records |
| `fix_apbs_input(p, plog)` | Replace bare filenames in APBS .in file with absolute paths |

### `src/electrostatics/run_apbs.py`

| Function | Description |
|----------|-------------|
| `process_apbs(protein_id, data_root)` | Run APBS and update metadata with timing |

### `src/surface/mesh.py`

| Function | Description |
|----------|-------------|
| `build_mesh(input_file, protein_id, data_root)` | Build SES mesh from .pdb or .pqr via MSMS; saves .npz and .vtk |
| `xyzr_from_pqr(pqr_file, plog)` | Parse .pqr for MSMS input (x, y, z, radius per atom) |
| `xyzr_from_pdb(pdb_file, plog)` | Parse .pdb for MSMS input using MDAnalysis vdW radii |
| `run_msms(xyzr_lines, positions, plog)` | Run MSMS subprocess and parse vertex/face output |
| `filter_vertices_to_bbox(verts, min_coords, max_coords)` | Filter mesh vertices to atom bounding box |
| `filter_faces(faces, mask)` | Remove faces referencing filtered-out vertices |
| `save_npz_mesh(out_path, ...)` | Save mesh to compressed .npz |
| `export_vtk(out_path, ...)` | Export mesh as ASCII VTK PolyData |

### `src/surface/esp_mapping.py`

| Function | Description |
|----------|-------------|
| `sample_esp(protein_id, data_root)` | Run all four ESP sampling variants; updates metadata |
| `read_dx(dx_file)` | Parse OpenDX scalar field; returns axis arrays and grid |
| `nearest_neighbor_esp(axes, grid, points)` | Snap surface points to nearest DX voxel |
| `laplacian_reconstruct(verts, faces, known_idx, known_values)` | Reconstruct ESP at all vertices via sparse Laplacian solve |
| `build_cotangent_laplacian(verts, faces)` | Build cotangent-weighted Laplacian matrix (sparse CSR) |
| `vertex_curvature(verts, faces)` | Approximate mean curvature magnitude at each vertex |
| `curvature_sampling(verts, faces, k)` | Sample k vertices weighted by curvature |
| `offset_points(points, normals, offset)` | Shift surface points outward along vertex normals |

### `src/analysis/metrics.py`

| Function | Description |
|----------|-------------|
| `evaluate_protein(protein_id, data_root, write_metadata=True)` | Compute Pearson r and RMSE for all four ESP variants |
| `compute_stats(esp_predicted, esp_reference)` | Return `(pearson_r, rmse)` for two arrays |

### `src/analysis/visualization.py`

| Function | Description |
|----------|-------------|
| `plot_esp_comparison(protein_id, data_root, clim=None)` | Render 2×2 PyVista window: interpolated vs Laplacian × PDB vs PQR |

---

## Incomplete / Stub Modules

The following files contain headers and docstrings only — implementation is a future stage:

| Module | Purpose |
|--------|---------|
| `src/data/dataset.py` | PyTorch Dataset and DataLoader |
| `src/data/graph_builder.py` | Graph construction for EGNN input |
| `src/data/transform.py` | Normalization, masking, augmentation |
| `src/models/egnn.py` | EGNN architecture and loss functions |

---

## Logging

All processing modules write to file-only loggers (no terminal output). A single notification line is printed to the terminal on completion or failure. Log level is set in `config.yaml` under `logging.level`.

Per-protein logs: `<data_root>/<protein_id>/logs/<protein_id>.log`
Pipeline log: path set in `config.yaml` under `paths.log_file`

---

## No Tests Yet

`tests/` contains only a placeholder. There is no test runner configured.
