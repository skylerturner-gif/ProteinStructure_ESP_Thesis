# Protein ESP Surface Prediction via Geometric Deep Learning

Thesis project investigating whether heterogeneous graph neural networks can learn geometry-conditioned approximations of protein electrostatic potential (ESP) fields on the solvent-excluded surface, from AlphaFold-predicted atomic structures alone. Two architectures are trained and compared — **AttentionESPN** (multi-head cross-attention atom→query message passing) and **DistanceESPN** (RBF-only, mean-aggregation) — against three non-GNN baselines, on a heterogeneous graph of atom nodes and curvature-sampled surface query nodes, trained to approximate ESP values computed by PDB2PQR + APBS.

> **Note for reviewers:** protein data, ESP grids, mesh files, and model checkpoints live outside this repository (too large for git). Decision notebooks include all outputs as saved cell results and document methodology even though they can't be re-executed without the external data.

---

## Project Structure

```
ProteinStructure_ESP_Thesis/
├── src/
│   ├── structure/          # AlphaFold API download (af_api.py)
│   ├── electrostatics/     # PDB2PQR and APBS wrappers
│   ├── surface/             # SES mesh generation (MSMS) and ESP sampling
│   ├── data/                # Dataset, graph builder, sampler, RBF expansion, transforms
│   ├── models/               # AttentionESPN, DistanceESPN, shared EGNN building blocks
│   ├── training/              # Trainer loop and ESP loss function
│   ├── analysis/               # Metrics, embedding/attention analysis, charge probe, visualization
│   ├── baseline_models/         # Non-GNN reference points: Coulomb, 3D CNN, Surface DGCNN
│   └── utils/                    # Logging, I/O, config, paths, filtering, parallelism
├── pipelines/               # Stage-by-stage (01-07) and end-to-end pipeline entry points
├── scripts/                 # Analysis, evaluation, and one-off audit/repair utilities
├── sweeps/                  # WandB-style sweep config YAMLs (ablation studies; gitignored)
├── notebooks/
│   ├── decisions/           # 16 sequential decision notebooks — every methodological choice
│   └── *.ipynb              # Post-training analysis notebooks (error, embeddings, baselines, ...)
├── configs/                 # (placeholder — experiment configs currently live in sweeps/)
├── data/
│   ├── datasets/            # Dataset index, TSVs, DATA_ORIGINS.md
│   ├── interim/, processed/ # Placeholders
│   └── ablation_subset_ids.txt
├── docs/                    # Development workflow, GitHub setup, testing guide
├── tests/                   # Placeholder — no test suite yet (see docs/testing_guide.md)
├── THESISPROCESSES.md       # Every methodological decision, with rationale — start here
├── SUMMER_PLAN.md            # Tiered research roadmap with DONE/IN PROGRESS status
├── EVAL_REPRODUCIBILITY_INVESTIGATION.md  # One open, unresolved metrics-reproducibility gap
├── config.template.yaml
├── environment.yml          # protein_esp: pipeline stages 1-5
├── pyg_environment.yml      # pyg_env: graph construction and modeling (stages 6-7)
└── pyproject.toml
```

---

## Setup

This project uses **two separate conda environments**, because PyTorch Geometric's dependencies conflict with the mesh-generation toolchain:

| Environment | Use | File |
|-------------|-----|------|
| `protein_esp` | Pipeline stages 1-5: downloading structures, ESP computation, mesh generation, ESP sampling | `environment.yml` |
| `pyg_env` | Stages 6-7: graph construction, model training, all `pyg_env`-tagged analysis notebooks/scripts | `pyg_environment.yml` |

```bash
# Stages 1-5
conda env create -f environment.yml
conda activate protein_esp
pip install -e .

# Stages 6-7
conda env create -f pyg_environment.yml
conda activate pyg_env
pip install -e .

# Machine-specific config — required before running anything
cp config.template.yaml config.yaml
# then edit paths.data_root, paths.log_file, executables.{pdb2pqr,apbs,msms}
```

`config.yaml` is git-ignored. See `config.template.yaml` for every key (paths, executables, electrostatics/surface/esp_mapping parameters, default model architecture, default training hyperparameters) with inline documentation. `pipelines/07_train.py` injects `config.yaml`'s `model:`/`training:` sections as argparse defaults — CLI flags still override them.

---

## Running the Pipeline

```bash
# Full end-to-end (handles conda env switching internally)
python pipelines/full_pipeline.py --id-file data/your_ids.txt

# Data generation only — stages 1-5, protein_esp env
python pipelines/data_gen_pipeline.py --id-file data/your_ids.txt --workers 4

# Modeling only — stages 6-7, pyg_env
python pipelines/model_pipeline.py --id-file data/your_ids.txt

# Individual stages
python pipelines/01_download_structures.py --id-file data/your_ids.txt
python pipelines/02_run_esp_calculations.py --filter --min-sequence-length 100
python pipelines/03_generate_surface.py --all
python pipelines/04_sample_esp.py --filter --min-plddt 70
python pipelines/05_evaluate_esp.py --all
python pipelines/06_build_graphs.py --all
python pipelines/07_train.py --model attention --epochs 120 --bf16

# WandB hyperparameter sweeps
python pipelines/run_sweep.py --sweep sweeps/phase_a_loss_weight.yaml
```

### Protein selection flags (stages 02-06)

| Flag | Type | Description |
|------|------|--------------|
| `--all` | — | Select every protein with a metadata JSON |
| `--filter` | — | Select proteins matching the criteria below |
| `--min-sequence-length` / `--max-sequence-length` | int | Amino acid count |
| `--min-plddt` / `--max-plddt` | float | Mean pLDDT score |
| `--min-surface-area` / `--max-surface-area` | float | SES area (Å²) |

`--all` and `--filter` are mutually exclusive; one is required.

---

## Decision Notebooks

Every methodological decision is documented in `notebooks/decisions/`, in the order it was made — what was tested, what alternatives were considered, and the quantitative rationale. Full narrative + current status of each: **[THESISPROCESSES.md](THESISPROCESSES.md)**.

| # | Notebook | Decision | Env |
|---|----------|----------|-----|
| 01 | `normal_offset_strategy` | ESP sampling offset from the SES surface | `protein_esp` |
| 02 | `ESP_sampling_method_strategy` | Trilinear interpolation vs. nearest-neighbour vs. Laplacian | `protein_esp` |
| 03 | `vertex_sampling_strategy` | Curvature-weighted vs. Poisson-disk vertex subsampling | `protein_esp` |
| 04 | `interpolation_strategy` | Multiquadric RBF full-mesh reconstruction | `protein_esp` |
| 05 | `graph_viability` | Heterogeneous graph definition: node/edge types, VRAM, timing | `pyg_env` |
| 06 | `model_exploration` | Dynamic batching strategy, per-size-class safe batch sizes | `pyg_env` |
| 07 | `loss_function_sweep` | Sweep A — Pearson-weighted loss, winner `pearson_weight=0.5` | `pyg_env` |
| 08 | `message_aggregation_sweep` | Sweep B — MessageLayer aggregation (mean/sum/max/multi) | `pyg_env` |
| 09 | `ema_justification` | EMA weight averaging kept, 120-epoch budget confirmed sufficient | `pyg_env` |
| 10 | `chemistry_layer_ablations` | Sweep F — residue/bond/radial/chemistry ablation ladder | `pyg_env` |
| 11 | `query_features_ablation` | Sweep C — query geometry features re-tested (kept off) | `pyg_env` |
| 12 | `invariance_analysis` | SE(3) rotation/translation robustness | `pyg_env` |
| 13 | `query_rounds_sweep` | Sweep D — QQ round count, full 0→32 sweep | `pyg_env` |
| 14 | `structure_rounds_sweep` | Sweep E — AA/AQ round count (in progress) | `pyg_env` |
| 15 | `full_vs_subset_comparison` | Full-dataset champion configuration | `pyg_env` |
| 16 | `attention_head_analysis` | Attention head count + chemical specialization | `pyg_env` |

`notebooks/decisions/_phase_template.ipynb` is the blank template new sweep notebooks are started from, not a decision itself.

### Post-training analysis notebooks (`notebooks/`, not in `decisions/`)

All use the full-dataset champions (`attention_aa4_aq2_qq16`, `distance_aa8_aq2_qq24`) unless noted. See THESISPROCESSES.md → "Post-Training Analysis Findings" for results.

| Notebook | Topic |
|----------|-------|
| `baseline_comparison.ipynb` | Non-GNN baselines vs. both architectures |
| `alphafold_uncertainty_analysis.ipynb` | pLDDT/PAE vs. model error |
| `error_clustering_analysis.ipynb` | Structural/chemical correlates of high-error proteins |
| `mesh_density_analysis.ipynb` | Direct high-density prediction vs. sparse-predict + RBF-interpolate |
| `pdb_af_comparison_analysis.ipynb` | AlphaFold vs. experimental PDB structures |
| `seed_conformation_analysis.ipynb` | AlphaFold seed conformational-uncertainty pilot |
| `vertex_error_analysis.ipynb` | Vertex-level error breakdown |
| `embedding_analysis.ipynb` | Embedding + attention chemistry analysis |
| `partial_charge_probe.ipynb` | Frozen-backbone probe recovering PARSE partial charges |
| `initial_protein_data_analysis.ipynb`, `data_pipeline_functionality.ipynb` | Early exploratory / pipeline-validation notebooks, predate the `decisions/` series |

---

## Architecture

### Data flow

```
UniProt ID list
    │
    ▼  01_download_structures.py
AlphaFold mmCIF + per-residue pLDDT + PAE  →  metadata.json
    │
    ▼  02_run_esp_calculations.py
mmCIF → PDB2PQR (PARSE, pH 7.0) → .pqr  →  APBS → .dx ESP grid
    │
    ▼  03_generate_surface.py
.pqr → MSMS → SES mesh (.npz + .vtk), singularity-corrected
    │
    ▼  04_sample_esp.py
mesh + .dx → trilinear ESP sample per vertex, curvature-subsample ~5% as query_idx → esp.npz
    │
    ▼  05_evaluate_esp.py
esp.npz → RBF interpolation baseline (interp_rmse / interp_pearson_r) → metadata.json
    │
    ▼  06_build_graphs.py
PQR + mesh + esp.npz → heterogeneous PyG HeteroData, cached as graph.pt
    │
    ▼  07_train.py
graph.pt (DynamicBatchSampler, packed by edge budget) → DistanceESPN / AttentionESPN
```

The PDB-structure mesh track (no explicit hydrogens) was evaluated in notebook 01 and dropped — every mesh, graph, and model downstream is built from PQR files only.

All per-protein data lives under `<data_root>/<protein_id>/`: `structure/` (`.cif`, `.pqr`, `_pae.json`), `electrostatics/` (`.dx`, optional — processed in memory by default), `mesh/` (`_mesh.npz`, `_mesh.vtk`), `esp/` (`_esp.npz`: vertices, ESP values, `query_idx`), `graph/` (`_graph.pt`), `logs/`, plus a top-level `<protein_id>_metadata.json` that accumulates fields across every stage (pLDDT stats, atom count, net charge, mesh vertex count, surface area, RBF baseline r/RMSE, per-stage timing, completion status).

### Graph structure

Heterogeneous PyG `HeteroData`, two node types and four edge types, no weight sharing across edge types:

| Node type | Contents |
|-----------|----------|
| `atom` | All heavy atoms + H, from PQR files. Element + residue + bond-type embeddings. |
| `query` | ~5% curvature-sampled surface vertices — the prediction targets. |

| Edge type | Definition | RBF range |
|-----------|-----------|-----------|
| `bond` | Covalent bonds (MDAnalysis-guessed) | [0.9, 1.8] Å |
| `radial` | Atom-atom kNN=16, covalent pairs excluded | [1.8, 8.0] Å |
| `aq` (atom→query) | kNN=32 per query, query-centric | [0.0, 12.0] Å |
| `qq` (query-query) | kNN=8 | [0.0, 8.0] Å |

Message passing: bond+radial interleaved (Stage 1, n rounds) → AQ (Stage 2, once — mean-aggregation for DistanceESPN, multi-head cross-attention with RBF geometry bias for AttentionESPN) → QQ (Stage 3, n rounds). Cached graphs store one `edge_dist` scalar per edge (plus `bond_order` for bond edges); models expand to Gaussian RBF features on-device at forward time (`src/data/rbf.py`) rather than storing the baked expansion — see THESISPROCESSES.md → "Graph Storage — Slim Edge Attributes" for why (~3.8x smaller cache files).

### Models

- **`src/models/distance_espn.py` — `DistanceESPN`**: geometry encoded via RBF edge features only, mean/sum/max-concat aggregation throughout.
- **`src/models/attention_espn.py` — `AttentionESPN`**: same three-stage structure, but Stage 2 (AQ) uses multi-head cross-attention with RBF geometry bias instead of mean aggregation. Best current results.
- **`src/models/egnn.py`**: shared building blocks (`AtomEncoder`, `QueryEncoder`, `MessageLayer`, `_AtomMP`, `_QueryRefine`) and vocab sizes both architectures import.
- **`src/baseline_models/`**: three non-GNN reference points — `coulomb/` (analytical vacuum-Coulomb physics floor, no learning), `cnn3d/` (dense voxel-grid 3D U-Net), `surface_dgcnn/` (EdgeConv point-cloud model on the mesh surface). See THESISPROCESSES.md → "Baseline Models" for what each isolates.
- **`src/training/trainer.py` — `Trainer`**: training loop, EMA weight averaging (always on), bf16 autocast (always on), checkpointing, validation.
- **`src/training/loss.py` — `ESPLoss`**: MSE + per-protein Pearson correlation, protein-size-weighted (always on).

---

## Key Utilities

### `src/utils/paths.py` — `ProteinPaths`

Single source of truth for every per-protein file path.

```python
from src.utils.paths import ProteinPaths
p = ProteinPaths(protein_id, data_root)
p.ensure_dirs()
p.pqr_path
p.dx_path
p.graph_path()
p.is_evaluated()
```

### `src/utils/config.py`

```python
from src.utils.config import get_config, get_data_root, get_log_file
cfg = get_config()   # cached YAML config dict
```

### `src/utils/io.py`

Thread-safe metadata JSON operations (`filelock`-backed):

```python
from src.utils.io import create_metadata, update_metadata, load_metadata
```

### `src/utils/filter.py`

```python
from src.utils.filter import get_protein_ids, add_filter_args
ids = get_protein_ids(data_root, select_all=True)
ids = get_protein_ids(data_root, min_plddt=70, min_sequence_length=100)
```

### `src/utils/helpers.py`

```python
from src.utils.helpers import get_logger, get_pipeline_logger, notify
```

Modules use file-only loggers (no terminal output during processing); a single-line notification prints on completion or failure. Log level: `config.yaml` → `logging.level`.

---

## Testing

`tests/` is currently a placeholder — no test runner is configured yet. `docs/testing_guide.md` documents the intended strategy (pure-Python logic and parsers under pytest with synthetic fixtures; subprocess calls to APBS/PDB2PQR/MSMS mocked, not exercised in CI; scientific validation stays in notebooks) for when that suite gets built out.

---

## Where to look next

- **[THESISPROCESSES.md](THESISPROCESSES.md)** — every methodological decision, the alternatives considered, and the quantitative rationale, including current full-dataset champion configuration and known open issues.
- **[SUMMER_PLAN.md](SUMMER_PLAN.md)** — tiered research roadmap with per-item DONE/IN PROGRESS status.
- **[EVAL_REPRODUCIBILITY_INVESTIGATION.md](EVAL_REPRODUCIBILITY_INVESTIGATION.md)** — one open, unresolved metrics-reproducibility gap (paused, not resolved — see THESISPROCESSES.md → "Known Issue").
