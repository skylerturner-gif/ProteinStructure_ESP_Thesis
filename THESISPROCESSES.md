# Thesis Processes & Assumptions

This document records every methodological decision made in the pipeline — what was chosen, what the alternatives were, and why. Update this file whenever a new choice is made.

---

## Structure Source

**Decision:** Use AlphaFold Database (v4) structures exclusively.

**Alternatives considered:** Experimental PDB structures.

**Rationale:** Uniform quality and coverage; pLDDT confidence scores available per-residue for downstream filtering; avoids heterogeneity of experimental resolution and missing density.

---

## Charge & Radius Assignment — Forcefield

**Decision:** `PARSE` forcefield via PDB2PQR.

**Config key:** `electrostatics.forcefield`

**Alternatives:** AMBER, CHARMM, OPLS.

**Rationale:** PARSE was designed specifically for implicit-solvent Poisson-Boltzmann electrostatics calculations; it is the standard forcefield recommended alongside APBS for ESP computation. AMBER and CHARMM are more appropriate for MD simulations where bonded terms matter.

---

## Protonation State / pH

**Decision:** Titrate at pH 7.0 using PROPKA.

**Config keys:** `electrostatics.ph_method: propka`, `electrostatics.ph_value: 7.0`

**Alternatives:** pdb2pka (slower, more rigorous); fixed protonation (no titration).

**Rationale:** pH 7.0 is physiological. PROPKA is the standard fast empirical method and is the default recommendation in PDB2PQR documentation.

---

## ESP Solver

**Decision:** APBS (Adaptive Poisson-Boltzmann Solver), linearised PB, implicit solvent.

**Alternatives:** DelPhi, OpenPB, molecular dynamics with explicit solvent.

**Rationale:** APBS is the de-facto standard for protein ESP in structural biology; integrates directly with PDB2PQR output; linearised PB is sufficient for the surface potential ranges encountered at physiological ionic strength.

---

## Surface Representation — Probe Radius

**Decision:** 1.4 Å probe radius (standard water molecule approximation).

**Config key:** `surface.probe_radius`

**Alternatives:** 1.2 Å (smaller probe, more detail in narrow clefts), 1.6 Å.

**Rationale:** 1.4 Å is the universally accepted solvent probe radius for SES construction and directly corresponds to the implicit-solvent boundary used in APBS.

---

## Surface Mesh Resolution

**Decision:** 3.0 vertices per Å² (MSMS density parameter).

**Config key:** `surface.msms_density`

**Alternatives:** 1.0 (coarse, fast), 5.0+ (fine, slow).

**Rationale:** 3.0 is the MSMS default and provides a good balance between mesh quality and file size for large proteins. May be revisited if EGNN training is memory-constrained.

---

## To Be Decided

### ESP Sampling — Structure

**Decision:** Use PQR files (PDB2PQR with PARSE forcefield + PROPKA pH 7.0) exclusively. PDB files discarded.

**Notebooks:** `notebooks/decisions/01_normal_offset_strategy.ipynb`, `notebooks/decisions/02_ESP_sampling_method_strategy.ipynb`

**Rationale:** At the chosen 0.5 Å offset, PDB meshes (no explicit hydrogens) retain std≈4.2 kT/e and range [−38, +39] kT/e — roughly double the dynamic range of the equivalent PQR mesh (std≈2.6, range [−10, +18]). This excess variance is consistent across all ESP sampling methods and all protein sizes tested. Explicit hydrogens from PDB2PQR shift the SES boundary outward, preventing mesh vertices from falling directly at near-atom charge singularities in the APBS DX grid. The PARSE forcefield and PDB2PQR are already required upstream for the ESP computation itself, so there is no additional tooling cost.

### ESP Sampling — Vertices

**Decision:** Curvature-weighted sampling at 5% of total mesh vertices per protein.

**Notebook:** `notebooks/decisions/03_vertex_sampling_strategy.ipynb`

**Rationale:** Swept fractions [1%, 2.5%, 5%, 10%, 25%, 50%] for both Poisson disk and curvature-weighted strategies across three protein sizes (7.6k, 30.5k, 215k vertices). At 5%, curvature-weighted sampling captures 4.4× more high-curvature vertices (recall=0.25 vs 0.056 for Poisson at top-20% curvature threshold) with nearly identical spatial coverage (p90≈1.84–1.94 Å) and ESP fidelity (r=0.917–0.954). Curvature sampling is also 4-6× faster than Poisson disk. The 5% fraction gives k≈1524 for the medium protein (30.5k verts) — a manageable node count for batched attention layers. Subsample size expressed as a fraction of total vertices scales consistently across protein sizes, so absolute node count grows appropriately with surface area.

### ESP Sampling — Normal Offset

**Decision:** 0.5 Å outward offset along vertex normals, PQR mesh only.

**Notebook:** `notebooks/decisions/01_normal_offset_strategy.ipynb`

**Rationale:** Sweeping offsets [0.0, 0.1, 0.25, 0.5, 1.0] Å on `AF-Q16613-F1` showed that sampling at the SES surface (0.0 Å) produces extreme spike values (PQR std=6.9, max=40.9 kT/e) from near-surface singularities in the APBS DX grid. At 0.5 Å, the PQR field narrows to [−10.3, 17.8] kT/e (std=2.6, outlier fraction 0.76%) while retaining Pearson r=0.77 with the surface values. At 1.0 Å, std barely decreases further (2.3) but Pearson r drops to 0.67, indicating over-smoothing. The PDB mesh (no explicit hydrogens) was discarded at all offsets due to persistently high variance (std=4.2 at 0.5 Å vs. 2.6 for PQR).

---

### ESP Sampling — Methods

**Decision:** Trilinear interpolation from the APBS DX grid.

**Notebook:** `notebooks/decisions/02_ESP_sampling_method_strategy.ipynb`

**Rationale:** Trilinear interpolation distance-weights the 8 surrounding voxels to produce a C0-continuous field with no piecewise-constant artifacts. Tested against nearest-neighbour and Laplacian smoothing across three protein sizes (small ~7.6k, medium ~30.5k, large ~215k vertices) on both PDB and PQR meshes. Trilinear achieves r=0.99 vs nearest-neighbour on PQR meshes with negligible runtime overhead (<0.1s even for the largest protein). Nearest-neighbour's low mean edge gradient metric is misleading — it reflects multiple vertices snapping to the same voxel centre, not genuine smoothness. Laplacian smoothing produced mesh-topology artifacts (mean edge gradient ~1000–1400 kT/e/Å) and is 20–100× slower; a seeding bug was also found and corrected during analysis (was incorrectly seeded from trilinear rather than independently from the DX grid).

---

### ESP Reconstruction — Full-Mesh Interpolation Method

**Decision:** Multiquadric RBF with local support (`neighbors=50`), ε = 1 / mean nearest-neighbour distance among sampled vertices.

**Notebook:** `notebooks/decisions/04_interpolation_strategy.ipynb`

**Rationale:** After curvature-weighted subsampling to 5% (k=1,524 for the medium protein), sparse predictions must be reconstructed at all mesh vertices. Tested three methods on AF-Q16613-F1: 1-NN (r=0.954, RMSE=0.766 kT/e), Gaussian RBF (r=0.978, RMSE=0.548 kT/e), and Multiquadric RBF (r=0.983, RMSE=0.466 kT/e). Multiquadric achieves a 39% RMSE reduction over 1-NN and is marginally better than Gaussian at the same ~1.12s runtime. The multiquadric basis √(1+(εr)²) has slower algebraic decay than Gaussian, better capturing the longer-range character of ESP fields between sparsely spaced samples. Runtime of ~1s per protein is acceptable for a preprocessing step.

---

### Protein Filtering — Sequence Length

**No global Min or Max size set yet.** Short peptides (< ~50 residues) may produce degenerate meshes, and large peptides (> ~500 residues) could blow VRAM and storage capabilities; this will be evaluated as data accumulates.

---

## Training — Loss Weighting & Gradient Accumulation

**Decision:** Inverse protein size weighting (`inv_size`) and gradient accumulation (2 steps) applied together (`both_batching` configuration).
**Notebook:** `notebooks/decisions/08_batching_analysis.ipynb`

**Problem addressed:** With dynamic batching over highly variable protein sizes, two issues compound: (1) large proteins dominate the loss gradient proportional to their surface area; (2) small per-protein batches produce high-variance gradient estimates causing train/val loss thrashing.

**Inverse protein size weighting:** Each protein's loss is scaled by `1 / n_query_vertices`, normalized across the batch, giving every protein equal weight in the gradient update regardless of size. A large protein (10,748 query pts) would otherwise contribute ~28× more gradient signal than a small one (380 query pts). Alone: Pearson r 0.766 → 0.776, RMSE 2.979 → 2.915.

**Gradient accumulation (2 steps):** Gradients accumulated over 2 consecutive forward passes before each optimizer step. Effectively doubles logical batch size without additional VRAM. Reduces per-epoch loss thrashing by averaging noisy per-protein gradient estimates. Alone: slight metric dip (r 0.766 → 0.753) but visibly smoother training loss curves. The smoothing is most valuable when paired with the corrected `inv_size` gradient direction.

**Combined result:** `both_batching` achieves r=0.783 (+0.018 over baseline), the best configuration. Train time +46% (1,374 s → 2,004 s) — acceptable given improved stability and metric quality.

---

## Training — Dynamic Batching Strategy

**Decision:** Dynamic batching bucketed by protein size class, with per-bucket safe batch sizes derived from measured peak VRAM.

**Notebook:** `notebooks/decisions/06_model_exploration.ipynb`

**Hardware:** 2× NVIDIA A100 40 GB.

**Rationale:** The dataset spans a ~27× range in per-protein VRAM footprint (small ~470 MB variable cost to large ~10,105 MB for AttentionESPN). A fixed batch size would either OOM on large proteins or waste ~95% of GPU capacity on small ones. Peak training VRAM (weights + gradients + AdamW optimizer states + graph tensors + activations) was measured per protein across 30 epochs on the three benchmark sizes.

**Safe batch sizes per A100 40 GB (AttentionESPN, conservative):**
| Size class | Approx atoms | Variable VRAM | Max proteins/batch |
|------------|-------------|---------------|-------------------|
| Small | ~587 | ~470 MB | ~87 |
| Medium | ~3,270 | ~1,769 MB | ~23 |
| Large | ~14,006 | ~10,105 MB | ~4 |

**Implementation:** Sort proteins ascending by total edge count (proxy for VRAM); bin into size classes; apply per-bucket max batch sizes. With 2 GPUs + gradient accumulation, effective batch size doubles without additional per-device VRAM cost.

---

## Graph Architecture — Heterogeneous Graph Definition

**Decision:** Heterogeneous graph with two node types (atom, query) and four edge types (bond, radial, AQ, QQ), each backed by an independent `MessageLayer` with distinct learned weights.

**Notebook:** `notebooks/decisions/05_graph_viability.ipynb`

**Node types:**
- *Atom nodes* — all atoms including H, sourced from PQR files.
- *Query nodes* — 5% curvature-sampled surface vertices.

**Edge types and RBF ranges:**
- **Bond** — covalent bonds guessed by MDAnalysis; bond order {1.0, 1.5, 2.0} + RBF dist [0.9, 1.8 Å].
- **Radial** — kNN=16 sparse atom–atom, covalent pairs excluded; RBF [1.8, 8.0 Å].
- **Atom→Query (AQ)** — kNN=32 per query, query-centric; RBF [0.0, 12.0 Å].
- **Query–Query (QQ)** — kNN=8; RBF [0.0, 8.0 Å].

**Message passing order:** bond → radial (Stage 1, interleaved, n rounds) → AQ (Stage 2, once) → QQ (Stage 3, n rounds).

**Key design principle:** No weight sharing across edge types. Bond and radial edges differ in distance range, density, and semantic content; a shared layer would conflate fundamentally different geometric relationships.

**Viability:** Graph construction time 0.1 s / 0.3 s / 1.6 s (small / medium / large). Forward-pass VRAM 15 MB / 72 MB / 406 MB — well under 24 GB for all sizes tested.

---

## Model Selection — Architecture & Feature Configuration

**Decision:** AttentionESPN with both optional features enabled: query geometry (`norm_curv`) and multi-aggregation (`multi`), referred to as the `both` configuration.

**Notebook:** `notebooks/decisions/07_model_analysis.ipynb`

**Evaluated on:** 20 proteins, 8 ablation runs (2 model types × 4 feature configs).

**Multi-aggregation:** Replaces single mean-aggregation in all `MessageLayer` updates with mean + sum + max concatenation. Improves AttentionESPN Pearson r by +0.023 over the base config. Decreases DistanceESPN r by −0.028 — the tripled update MLP input overwhelms learned representations in the mean-aggregation backbone at the current hidden dimensionality.

**Query geometry features:** Surface normal (3D) and mean curvature (scalar, log1p-scaled) injected into query node embeddings via `QueryEncoder`. Provides explicit surface-shape inductive bias. Causes a performance dip in several configurations (**feature overload**): the attention model already learns geometry-aware atom weighting implicitly via RBF-biased cross-attention, so explicit geometry injection partially duplicates this signal. The dip is most severe for DistanceESPN (`norm_curv` r=0.720 vs base r=0.776, −0.056).

**Why `both` over `multi` alone:** `attention_multi` has the highest Pearson r in this run (0.787 vs 0.766 for `both`), but the gap is within noise for a 20-protein test set. Query geometry features were retained provisionally. *(Superseded — see QQ Rounds Ablation below.)*

---

## Model Selection — QQ Rounds & Geometry Feature Reversal

**Decision:** QQ rounds kept (qq=2). Geometry features (surface normals + curvature) **dropped**.

**Notebook:** `notebooks/decisions/09_query_layer_analysis.ipynb`

**Final forward configuration:** AttentionESPN + multi-aggregation + inverse size weighting + gradient accumulation + qq=2, no geometry features.

**QQ rounds are essential:** Removing QQ rounds drops Pearson r from 0.783 → 0.667 (−0.116) — the largest single-component degradation across all ablations. The kNN=8 distance cutoff means each QQ pass reaches only immediate surface neighbours (~few Å). Without multi-hop iteration, the model cannot capture long-range surface continuity: inter-residue charge patterns, surface-scale charge asymmetry, and smooth ESP gradients all require information to diffuse across multiple hops. Two QQ rounds act as a limited surface diffusion process that makes these patterns learnable.

**Geometry features dropped:** The QQ ablation isolates the geometry feature effect without lateral propagation — removing normals + curvature when qq=0 *improves* r from 0.667 to 0.748 (+0.081). Without a surface propagation pathway, explicit geometry injection is disruptive rather than informative. Combined with the marginal and inconsistent gains seen in notebook 07 (geometry features underperformed multi-only by −0.021 r), the pattern is clear: the attention model's RBF-biased cross-attention already encodes geometric context implicitly, and adding explicit surface geometry features introduces competing gradient signals regardless of QQ depth.

---

## Model Validation — Partial Charge Probe

**Finding:** Atom embeddings after Stage 1 message passing encode per-atom partial charges with mean R² = 0.9947 (RMSE = 0.020 e) from a frozen-backbone 3-layer MLP probe.

**Notebook:** `notebooks/decisions/10_weight_charge_analysis.ipynb`

**Significance:** Partial charges are the direct physical source of ESP — the APBS solver computes ESP by treating each atom as a point charge at its PARSE-assigned partial charge. A frozen probe recovering these charges at ~99.5% explained variance confirms the model is not operating as a geometric interpolator; instead, atom representations after the bond and radial message passing rounds encode the charge distribution required for ESP prediction. The probe converged rapidly (30 epochs, MSE 0.0077 → 0.0004), indicating this information is linearly accessible from the embeddings.

**Per-environment accuracy:** Best-predicted environments are H (RMSE=0.012), backbone amide N (RMSE=0.016), and aliphatic C (RMSE=0.025). Hardest is aromatic C (RMSE=0.043), where ring-position-dependent charge variation requires finer chemical context. Several environments show NaN Pearson r because PARSE assigns a fixed charge to all atoms of that type — low RMSE with NaN r indicates correct constant prediction, not failure.

