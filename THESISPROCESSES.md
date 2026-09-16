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

## Mesh Singularity Correction

**Decision:** Automatically detect and correct mesh vertices that penetrate too deep into a nearby atom's VdW sphere (a "singularity" — the vertex sits inside the excluded volume rather than on its boundary), inline during mesh generation.

**Config:** `GEOM_THRESHOLD = 0.3` Å in `src/surface/mesh.py`.

**Rationale:** MSMS occasionally produces vertices with penetration depth > ~0.3 Å into the nearest atom's VdW radius, most often in tightly packed or concave regions. Left uncorrected, ESP sampled at that vertex (via trilinear interpolation from the APBS grid) picks up field values from deep inside the low-dielectric interior instead of the true surface value. `build_mesh()` now runs `geom_validity()` against every vertex right after MSMS generation, and any vertex flagged by `needs_fix` is corrected via `resolve_singularities()` before the mesh is saved — this is unconditional, not a flag, so every mesh generated from here on is already clean.

**Retrofit vs. audit:** Structures meshed *before* this fix landed can still carry the defect. `scripts/retrofit_mesh_singularities.py` (fed by `scripts/survey_mesh_singularities.py`) exists to fix those in place, but is run manually, on demand — it is not part of the normal pipeline, which always skips proteins with an existing cached mesh and so will never re-surface this on its own. `scripts/check_mesh_singularities.py` is a separate, read-only audit (writes no ID file, triggers no retrofit) that reports how many already-downloaded structures are affected. As of the last check, 0/1,045 currently-downloaded proteins are affected — the historical retrofit already covers the full active dataset.

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

**Decision (historical):** Inverse protein size weighting (`inv_size`) and gradient accumulation (2 steps) applied together (`both_batching` configuration).
**Notebook:** _retired during the notebook restructuring — no longer in the repo._

> **Superseded** — protein-size-weighted loss is now hardcoded always-on and gradient accumulation is no longer part of the active sweep plan; see "Current Core Training Configuration" below. Kept here for the historical rationale.

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

**Notebook:** _original feature-ablation notebook (retired — superseded by the staged ablation plan; the current notebook 07 covers a different axis, message-passing aggregation)_

**Evaluated on:** 20 proteins, 8 ablation runs (2 model types × 4 feature configs).

**Multi-aggregation:** Replaces single mean-aggregation in all `MessageLayer` updates with mean + sum + max concatenation. Improves AttentionESPN Pearson r by +0.023 over the base config. Decreases DistanceESPN r by −0.028 — the tripled update MLP input overwhelms learned representations in the mean-aggregation backbone at the current hidden dimensionality.

**Query geometry features:** Surface normal (3D) and mean curvature (scalar, log1p-scaled) injected into query node embeddings via `QueryEncoder`. Provides explicit surface-shape inductive bias. Causes a performance dip in several configurations (**feature overload**): the attention model already learns geometry-aware atom weighting implicitly via RBF-biased cross-attention, so explicit geometry injection partially duplicates this signal. The dip is most severe for DistanceESPN (`norm_curv` r=0.720 vs base r=0.776, −0.056).

**Why `both` over `multi` alone:** `attention_multi` has the highest Pearson r in this run (0.787 vs 0.766 for `both`), but the gap is within noise for a 20-protein test set. Query geometry features were retained provisionally. *(Superseded — see QQ Rounds Ablation below.)*

---

## Model Selection — QQ Rounds & Geometry Feature Reversal (historical)

**Decision (historical):** QQ rounds kept (qq=2). Geometry features (surface normals + curvature) **dropped**.

**Notebook:** `notebooks/decisions/13_query_rounds_sweep.ipynb` (early content)

> **Superseded** — this reflects an early pass through notebook 13, run before the Sweep A–F restructuring (note the r=0.783 baseline and gradient-accumulation config, both retired — see "Current Core Training Configuration" below). The notebook was later expanded into the full Sweep D QQ-rounds sweep; see "Model Selection — QQ Rounds Sweep D" below for the current, adopted numbers. Kept here for the historical rationale on *why* QQ rounds and geometry features were flagged as coupled in the first place.

**QQ rounds are essential:** Removing QQ rounds drops Pearson r from 0.783 → 0.667 (−0.116) — the largest single-component degradation across all ablations at the time. The kNN=8 distance cutoff means each QQ pass reaches only immediate surface neighbours (~few Å). Without multi-hop iteration, the model cannot capture long-range surface continuity: inter-residue charge patterns, surface-scale charge asymmetry, and smooth ESP gradients all require information to diffuse across multiple hops.

**Geometry features dropped:** The QQ ablation isolates the geometry feature effect without lateral propagation — removing normals + curvature when qq=0 *improves* r from 0.667 to 0.748 (+0.081). Without a surface propagation pathway, explicit geometry injection appeared disruptive rather than informative. **This specific "features hurt" claim was re-tested and reframed — see Sweep C below.**

---

## Model Selection — QQ Rounds Sweep D (full sweep, post-restructuring)

**Status:** Complete, decision recorded.

**Notebook:** `notebooks/decisions/13_query_rounds_sweep.ipynb` (current/full content)

**Swept:** qq ∈ {0, 4(baseline), 8, 12, 16, 24, 32} for AttentionESPN; {0, 4, 8, 12, 16, 24} for DistanceESPN, holding AA/AQ at the 4/4/4 baseline.

**Test Pearson r / RMSE:**
| qq | Attention r / RMSE | Distance r / RMSE |
|----|---------------------|---------------------|
| 0  | 0.8382 / 2.935 | 0.8432 / 2.918 |
| 4  | 0.8898 / 2.660 | 0.8964 / 2.573 |
| 8  | 0.9032 / 2.417 | 0.9055 / 2.402 |
| 12 | 0.9072 / 2.308 | 0.9104 / 2.272 |
| 16 | 0.9073 / 2.252 | 0.9108 / 2.190 |
| 24 | 0.9078 / 2.136 | 0.9102 / 2.149 |
| 32 | 0.9067 / 2.108 | — (not run) |

**Decision:** The 0→12 climb is a 100%-of-bins win at every step for both architectures — no exceptions across the ESP range. Past qq=12, Pearson r plateaus/gets noisy (Attention dips slightly at 24) while RMSE keeps inching down. Per the notebook's own accuracy-only read, winners are Attention qq=16 (val r=0.9032) and Distance qq=24 (val r=0.9073), though margins beyond qq=12 are within run-to-run noise. QQ round count was **not** chained forward into the AA/AQ round sweep (Sweep E) — each axis was perturbed independently off the shared 4/4/4 baseline, not compounded. See "Full-Dataset Champion Configuration" below for the round counts actually adopted in the models used by every post-training analysis notebook.

---

## Model Selection — Message-Passing Aggregation Sweep B

**Status:** Data complete; decision section in the notebook is an unfilled template (literal "?" placeholders) — never formally written up, though the data unambiguously favors one option and that option is already the hardcoded default (see "Current Core Training Configuration").

**Notebook:** `notebooks/decisions/08_message_aggregation_sweep.ipynb`

**Swept:** `mean`, `sum`, `max`, `multi` (concat of mean+sum+max) for the bond/radial/QQ `MessageLayer` stages, both architectures. (Attention's AQ stage is always cross-attention regardless of this setting.)

**Validation Pearson r:** Attention — multi 0.8915 > sum 0.8882 > max 0.8740 > mean 0.8735. Distance — multi 0.8953 > sum 0.8895 > mean 0.8821 > max 0.8567. `multi` wins for both architectures and is the config already adopted everywhere else in the codebase.

---

## Model Selection — Chemistry Layer Ablation Sweep F

**Status:** Data complete; decision section in the notebook is an unfilled template. First attempt (pre-2026-08) never finished cleanly — `distance_chem_stripped` stalled at epoch 98/120 and other runs had an `agg='sum'` vs `'multi'` config bug; redone in full against the standardized 4/4/4 multi-agg baseline.

**Notebook:** `notebooks/decisions/10_chemistry_layer_ablations.ipynb`

**Swept (6-rung ladder × 2 architectures):** `baseline` (full chemistry) → `no_residue` → `no_radial` → `no_bond` (true spatial kNN) → `spatial_only` (positions + element type only) → `chem_stripped` (no atom-atom message passing at all).

**Validation Pearson r:** Attention — no_residue 0.9029 ≈ baseline 0.9016 > no_radial 0.8891 > no_bond 0.8786 > spatial_only 0.8565 > chem_stripped 0.8255. Distance — baseline 0.9046 > no_residue 0.9032 > no_radial 0.8993 > no_bond 0.8797 > chem_stripped 0.8339 > spatial_only 0.8176.

**Unwritten but data-supported read:** Residue-identity embedding looks nearly redundant (dropping it costs ~nothing, occasionally helps). Bond topology and radial kNN each cost real Pearson r when removed. Chemistry-stripped and spatial-only rungs drop substantially (~0.07–0.09 r), confirming atom-level chemistry is load-bearing overall even though residue identity specifically is not. **Note:** the per-bin delta cells (§3) still reference a `phase_d` baseline checkpoint inconsistent with the `phase_h` baseline used in the main config table — likely a leftover from the pre-redo version, not corrected in the 2026-08 rerun.

---

## Model Selection — Structure Rounds (AA/AQ) Sweep E — in progress

**Status:** Explicitly self-flagged as partial by the notebook. As of last update: 3/8 planned runs done (Attention aa8, aa12, aq8); Attention aq12 was still training; all 4 Distance runs were pending. No decision has been written — the notebook's own final cell states the decision is "to be filled in once all 8 runs complete."

**Notebook:** `notebooks/decisions/14_structure_rounds_sweep.ipynb`

**Swept so far:** AA (bond/radial) rounds ∈ {8, 12} and AQ rounds ∈ {8, 12}, each independently vs. the 4/4/4 baseline, holding QQ at 4.

**Test Pearson r / RMSE so far:** AA sweep — Attention baseline 0.8898/2.660, aa8 0.9022/2.416, aa12 0.9003/2.375; Distance baseline 0.8964/2.573, aa8 0.9035/2.425, aa12 0.9034/2.349. AQ sweep — Attention baseline 0.8898/2.660, aq8 0.8923/2.571 (aq12 pending); Distance baseline 0.8964/2.573, aq8 0.8963/2.565, aq12 0.8960/2.560 (essentially flat).

**Early read (not yet a decision):** AA rounds are a real second lever, worth ~+0.01–0.014 r at aa8, with diminishing/mixed returns at aa12. AQ rounds look close to redundant/saturated even at the 4-round baseline. The conditional-combination step (testing AA+QQ or AQ+QQ together) is blocked pending these results. Despite the sweep being unfinished, the full-dataset champion models already adopted asymmetric AA rounds per architecture (Attention aa=4, Distance aa=8) — see "Full-Dataset Champion Configuration" below — consistent with this early read but not confirmed by a closed-out sweep.

---

## Model Validation — EMA Weight Averaging

**Status:** Complete, decision recorded.

**Notebook:** `notebooks/decisions/09_ema_justification.ipynb`

**Motivation:** EMA (`ema_decay=0.999`) was hardcoded on during the Sweep A–F restructuring, but `metrics.csv` only ever logged EMA-swapped validation metrics — nothing demonstrated the benefit directly. `Trainer` was extended to also log raw (non-EMA) val metrics per epoch.

**Test:** Two `phase_h` runs (`attention_444_multi_200ep`, `distance_444_multi_200ep`, the standardized 4/4/4 multi-agg baseline) trained for 200 epochs (vs. the usual 120) to observe raw-vs-EMA divergence and any late-horizon degradation.

**Result:** EMA overtakes raw val loss early (Attention epoch 3, Distance epoch 25) and stays ahead almost throughout, though raw briefly regains the lead near epoch 196 for both. Best EMA val_loss: Attention 0.5183 (epoch 124), Distance 0.5169 (epoch 110). Both models degrade mildly past their best epoch through epoch 200.

**Decision:** EMA is kept — the margin is real but modest, not dramatic. Since checkpoint selection already uses minimum EMA val_loss (not final epoch), late-stage degradation costs nothing in practice. Also confirms the standard 120-epoch training budget isn't leaving performance on the table.

---

## Model Validation — Query Feature Ablation, Re-confirmed (Sweep C)

**Status:** Complete, decision recorded. Supersedes the "geometry features clearly hurt" framing above.

**Notebook:** `notebooks/decisions/11_query_features_ablation.ipynb`

**Motivation:** The original "features hurt" finding (see historical QQ Rounds section above) predated the current EMA + protein-weighted loss standard and wasn't guaranteed to still hold. Re-run from scratch under the current 4/4/4, `agg=multi` baseline.

**Result:** Validation Pearson r — Attention off 0.8915 vs. on 0.8876 (Δ −0.0024 on test); Distance off 0.8953 vs. on 0.8956 (Δ −0.0004 on test, and "on" actually wins RMSE for Distance: 2.561 vs 2.573). Spatial-error metrics (Moran's I, ESP-error Spearman r) show negligible on/off differences for both models.

**Decision:** "Features off" is kept as the default, but the margin is narrow enough that the notebook explicitly reframes the conclusion from "features hurt" to **"features are neutral."** The earlier pre-EMA gap (Attention norm_curv r=0.720 vs base r=0.776) has "essentially disappeared" under the new training recipe. Features-off is retained partly on this narrow margin and partly on the separate SE(3)-invariance argument below (features-off models are exactly rotation/translation invariant by construction).

---

## Model Validation — SE(3) Rotation/Translation Robustness

**Status:** Conclusion stands but is explicitly flagged by the notebook itself as needing a re-run ("⚠ Needs re-run") — cited numbers are from legacy `*_v2_feat_off/on` checkpoints, not the current `phase_a`/`phase_c` checkpoints the rest of the notebook points at.

**Notebook:** `notebooks/decisions/12_invariance_analysis.ipynb`

**Method:** Evaluation-only, reusing the 4 Sweep C checkpoints (`{attention,distance} × {features on/off}`). Positions rigidly transformed (translation / rotation / rotation+translation, 5 repeats each), edges recomputed via the real graph builder, predictions compared pre/post-transform on 50 sampled test proteins. A sanity check first confirmed numerically that RBF edge features are exactly invariant (diff = 0.00) and bond edges untouched, while surface normals do change (max diff 1.58).

**Result:** Translation — prediction-prediction correlation = 1.0000 for all 4 models. Rotation / rotation+translation — features-off models = 1.0000 (exact, by construction); features-on: Attention 0.9932 ± 0.0035, Distance 0.9972 ± 0.0018. Mean Δ Pearson r < 0.001 across all conditions.

**Decision (provisional pending re-run):** Non-equivariant architecture is justified. Rotational instability from the surface-normal feature is real but tiny, with negligible accuracy impact. No equivariant architecture (SE(3)-Transformer/Equiformer) is warranted.

---

## Full-Dataset Champion Configuration

**Status:** Adopted — these are the exact checkpoints used as "the model" throughout every post-training analysis notebook (partial charge probe, embedding analysis, AlphaFold-confidence analysis, mesh density study, PDB/AF comparison, seed conformation study, vertex error analysis).

**Notebook:** `notebooks/decisions/15_full_vs_subset_comparison.ipynb`

**Checkpoints:** `attention_aa4_aq2_qq16` (AttentionESPN, 4 bond/radial rounds, 2 AQ rounds, 16 QQ rounds) and `distance_aa8_aq2_qq24` (DistanceESPN, 8 bond/radial rounds, 2 AQ rounds, 24 QQ rounds), both `agg=multi`, both trained on the full ~8,461-protein dataset and evaluated on the shared 848-protein test split.

**Dataset scale finding:** Holding this exact config fixed, full-dataset training beats the original 1,045-protein subset by a consistent margin for both architectures: Attention test r 0.9368 vs 0.9090 (Δ +0.0278), Distance 0.9386 vs 0.9164 (Δ +0.0222) — roughly 9–10% relative RMSE improvement either way. Epoch budgets weren't perfectly matched (full: 75–100 epochs; subset: 120) so the gap is a slight underestimate of pure scale effect if anything. Per-protein ranking correlation across the two test splits was attempted but only 13 proteins overlapped — too few to draw conclusions from. Scale helps by a similar magnitude regardless of architecture; neither model looks closer to saturating than the other.

---

## Model Validation — Attention Head Specialization

**Status:** Data collection substantially complete (§1–9 of 13), but the notebook's own stated "real test" (§11–12, emergent clustering + quantitative ARI/NMI agreement check) was never executed — those cells have no output in the saved notebook — and the final decision table (§13) is an unfilled "?" placeholder.

**Notebook:** `notebooks/decisions/16_attention_head_analysis.ipynb`

**Scope:** Not "which head count wins" (already settled elsewhere: n_heads=4 is the champion — test r 0.9327/0.9368/0.9347 for 2/4/8 heads respectively) but what each head specializes in chemically. Examines per-element/per-residue attention weight distributions, an electronegativity-weighted head-affinity score, and grouping by side-chain class and solvent exposure (from the project's own SES mesh data) across the 3 head-count checkpoints.

**Findings so far (not yet a written decision):** A specialization score (std of per-head element-attention) rises mildly with head count: 0.0230 (2 heads) → 0.0287 (4) → 0.0330 (8). Acidic residues (ASP/GLU) consistently draw elevated attention across all configs. Whether this constitutes genuine emergent "chemistry/physics/dynamics" clustering (the flagship framing in SUMMER_PLAN.md) is exactly what the un-run §11–12 quantitative validation was meant to establish.

---

## Current Core Training Configuration (post-restructuring baseline)

**Decision:** As of the Sweep A–F restructuring, the following are fixed, hardcoded defaults for every run — no longer optional flags, and no longer treated as ablation axes:

- **Message-passing rounds:** 4/4/4 (`n_bond_radial_rounds=4`, `n_aq_rounds=4`, `n_qq_rounds=4`).
- **Aggregation:** `multi` (mean + sum + max concatenation) in every `MessageLayer`.
- **Query geometry features:** off (`query_curvature`, `query_normal` both `false`) — see QQ Rounds & Geometry Feature Reversal above.
- **EMA weight averaging:** always on, `ema_decay=0.999`. Previously an optional `use_ema` flag; now hardcoded in `Trainer.__init__` — every run maintains an EMA shadow of the model weights and evaluates/checkpoints from it.
- **Protein-size-weighted loss:** always on. Previously an optional `protein_weighted` flag gating `inv_size` weighting (see Loss Weighting & Gradient Accumulation above); `ESPLoss.forward()` now unconditionally computes per-graph MSE and averages across graphs in the batch, so every protein contributes equally to the gradient regardless of its query-node count.
- **Mixed precision:** `bf16` autocast for training/val/inference.

**Superseded:** The `10_batching_analysis.ipynb` (`both_batching`) gradient-accumulation finding is retired as an active axis — that notebook was removed in the restructuring, and gradient accumulation is not part of the current sweep plan (`grad_accum_steps` defaults to 1 in every Sweep A run). The project has moved from tuning individual batching/weighting knobs one at a time to this single fixed, standardized recipe. `pearson_weight` (Sweep A) is the only loss-level axis still being actively swept on top of it; Sweeps B–F re-test the remaining architecture axes (aggregation, query features, QQ rounds, AA/AQ rounds, chemistry ablation) against this same fixed baseline.

---

## Known Issue — Sweep A Baseline Reproducibility (Unresolved, Investigation Paused)

**Status:** Paused, not resolved. Do not re-cite this as closed.

Sweep A's winning config (`checkpoints/phase_a/attention_pw05`), cited throughout `notebooks/decisions/07_loss_function_sweep.ipynb` and every later phase_b–e notebook, was originally recorded at test Pearson r = 0.8926, RMSE = 2.5889. A later fresh re-evaluation of the same checkpoint (`scripts/reevaluate_test_timing.py`, run twice, consistent both times) instead gave r = 0.8898, RMSE = 2.660 — a small but real, reproducible, unexplained shift. bf16 precision and run-to-run randomness were ruled out; the top-priority next step (isolating whether a concurrent `trainer.py` edit — always-on EMA + autocast — caused it, by re-running through the pre-diff `evaluate_test`) was never completed. `attention_pw05`'s original `test_metrics.json`/`test_predictions/` were overwritten in place during the investigation and are not recoverable (`checkpoints/` is outside this git repo and untracked).

Full investigation notes, ruled-out hypotheses, and the priority-ordered next-steps list: `EVAL_REPRODUCIBILITY_INVESTIGATION.md` (repo root).

**Do not run `scripts/reevaluate_test_timing.py` on any other checkpoint without first reading that file** — it now defaults to a safe non-destructive output location (`model_eval/reeval_test_timing/<checkpoint_name>/`), but the underlying r/rmse discrepancy is still unexplained.

---

## Training — DataLoader Worker Configuration

**Decision:** `persistent_workers=False`, `pin_memory=False` for the training/val `DataLoader`, regardless of `num_workers`.

**Alternatives considered:** `persistent_workers=True` + `pin_memory=True` (attempted speed optimization — avoids respawning the worker pool every epoch and enables async H2D copy overlap).

**Rationale:** That combination reliably corrupted data in transit from `DataLoader` workers under the real training config (DDP world_size=2, `num_workers=8`) — `atom_type` values went out of the embedding's valid range and crashed `AtomEncoder`'s `self.atom_emb(atom_type)` with a CUDA device-side assert, reproducing identically on both DistanceESPN and AttentionESPN, on batch 1, every time. Individually cached graph files were verified clean (0/6768 training graphs had any out-of-range index), and the crash never reproduced with `num_workers=0` or with `persistent_workers`/`pin_memory` off — isolating the DataLoader worker path as the cause, not the data itself.

This isn't unique to this codebase: PyTorch Geometric's `Collater` has a known shared-memory leak under `num_workers > 0` ([PyG #3396](https://github.com/pyg-team/pytorch_geometric/issues/3396)), and plain PyTorch has a documented history of `pin_memory` mangling custom container objects passed through `collate_fn` ([#67831](https://github.com/pytorch/pytorch/issues/67831), [#32150](https://github.com/pytorch/pytorch/issues/32150), [#70158](https://github.com/pytorch/pytorch/issues/70158)) plus separate crash/hang reports for the `persistent_workers`+`pin_memory` combination specifically ([#48370](https://github.com/pytorch/pytorch/issues/48370), [#47445](https://github.com/pytorch/pytorch/issues/47445), [#24927](https://github.com/pytorch/pytorch/issues/24927)). Root cause not isolated to a single upstream line; the working theory is PyG's shared-memory batching of heterogeneous graphs (multiple node/edge types needing correct IPC reconstruction) is the fragile part, and `persistent_workers` (workers never respawn, so any leak/staleness accumulates all run) plus `pin_memory` (a second background thread re-walking the same object) both amplify it.

**Current setting:** `num_workers=2` (not 8 — see `sweeps/full_dataset_champions.yaml`), workers respawned every epoch (`persistent_workers=False`). Validated with a full clean run (train → val → checkpoint → test eval, 848 test proteins, zero corruption) on the full 8461-protein dataset. Speed cost: ~45 min/epoch at `num_workers=2` on the full dataset — a 120-epoch run is on the order of days, not hours. Not yet re-isolated whether `pin_memory` or `persistent_workers` alone (rather than the combination) is safe; that's a possible avenue to partially recover the speedup if training time becomes a blocker.

---

## Model Validation — Partial Charge Probe

**Status:** Complete, full-dataset re-run (updates an earlier 110-protein-scale result).

**Finding:** Atom embeddings after Stage 1 message passing encode per-atom partial charges from a frozen-backbone MLP probe, evaluated on the full-dataset champions (5.56M atoms, 848 test proteins): **AttentionESPN RMSE = 0.0110, R² = 0.9987; DistanceESPN RMSE = 0.0132, R² = 0.9981.** Attention encodes charge measurably better (16% lower RMSE) despite both being near-ceiling.

**Notebook:** `notebooks/partial_charge_probe.ipynb`

**Significance:** Partial charges are the direct physical source of ESP — the APBS solver computes ESP by treating each atom as a point charge at its PARSE-assigned partial charge. A frozen probe recovering these charges at ~99.8% explained variance confirms the model is not operating as a geometric interpolator; instead, atom representations after the bond and radial message passing rounds encode the charge distribution required for ESP prediction.

**Per-environment accuracy:** Sulfur is hardest for both architectures — rare and chemically diverse (free thiol vs. disulfide). Carbon is second-hardest, with the widest internal charge-class spread. Worst individual classes: HIP-tautomer histidine ring carbons, and disulfide/thiol sulfur. The probe resolves 36 (element, charge-class) groups, including tautomer disambiguation that a name-keyed lookup would miss. The `after_encoder` (pre-message-passing) comparison was not rerun at full scale — flagged in the notebook as a follow-up, not a gap in the `after_mp` claims above.

---

## Baseline Models — Non-GNN Reference Points

Three baselines were built to bound and contextualize the heterogeneous-graph model's performance from different directions: a physics-only lower bound, a dense-grid learned alternative, and a surface-only learned alternative. All three read from the same PQR-derived structure files as the primary pipeline. None of them see partial charges as an *input* — matching the epistemological constraint placed on the main GNN (recover charge-dependent behavior from geometry/identity alone) — except the Coulomb baseline, whose entire premise is to be powered directly by the PARSE partial charges instead of learning anything.

### Vacuum Coulomb Baseline — physics floor

**Approach:** No learning. `V(q) = Σ_i q_i · k / r_iq` — vacuum Coulomb potential summed directly over PARSE partial charges read from the PQR file, evaluated at query positions (distance floor `R_MIN=0.5` Å to avoid singularities at atom centers).

**Code:** `src/baseline_models/coulomb/predictor.py` (`CoulombESP`).

**Rationale:** APBS solves the linearised Poisson–Boltzmann equation, which accounts for solvent screening and the low-to-high dielectric boundary at the molecular surface. Vacuum Coulomb has neither. Because this baseline is handed the *exact* partial charges the GNN never sees, and still omits all solvent/dielectric physics, it isolates how much of the APBS ESP signal is "trivial" charge summation versus genuinely dependent on the solvent-boundary geometry the GNN has to infer implicitly from structure alone. It's a lower reference bound rather than a competitor to beat outright — informative either way: if the GNN barely beats it, the model may be leaning on charge-like shortcuts inferred from geometry; if it beats it by a large margin, that's evidence the model is capturing solvent-boundary effects Coulomb structurally cannot represent.

### 3D CNN Baseline — dense voxel-grid alternative

**Approach:** `src/baseline_models/cnn3d/` voxelizes the PQR structure into a 4-channel element-occupancy grid (C/N/O/S counts per voxel, 1 Å resolution, heavy-atom bounding box + 5 Å padding — hydrogens and rare elements dropped, same as the GNN's epistemological footing). `Vox3DCNN` is a 3D U-Net: encoder 4→32→64→128 channels via stride-2 conv blocks, a bottleneck, and a decoder with trilinear-upsample skip connections back to a 1-channel dense ESP grid. Query-node ESP is read off the dense grid via trilinear `grid_sample` at each query's continuous xyz position.

**Code:** `src/baseline_models/cnn3d/{model,voxelizer,dataset,train,evaluate}.py`.

**Rationale:** Dense 3D CNNs are the classical architecture family for voxelized molecular property prediction and are the most direct non-graph competitor. Deliberately kept on the same epistemological footing as the GNN — element identity and shape only, no partial charges — so any performance gap reflects the inductive-bias difference (translation-equivariant dense convolution over a regular grid vs. sparse heterogeneous message passing) rather than an information advantage. Voxel resolution and encoder depth are the main structural constraints: 1 Å voxels bound the geometric precision, and the 3-level encoder bounds the effective receptive field relative to the GNN's kNN/multi-round message passing.

### Surface DGCNN Baseline — surface-only alternative

**Approach:** `src/baseline_models/surface_dgcnn/` treats the mesh surface as a point cloud — xyz + normals + nearest-atom element/residue one-hot (33D per point, still no partial charges). A static kNN graph on 3D coordinates is pre-computed and cached per protein (not rebuilt every forward pass, unlike the original DGCNN's dynamic feature-space graph). Three stacked `EdgeConv` layers (`h_i = max_{j∈N(i)} MLP([h_i, h_j − h_i])`) produce multi-scale per-point embeddings, which are concatenated and projected; each query node gathers its k nearest precomputed surface embeddings, mean-pools them, and passes the result through a small MLP head.

**Code:** `src/baseline_models/surface_dgcnn/{model,dataset,train,evaluate}.py`.

**Rationale:** Isolates the surface-only signal from the primary model's atom+query heterogeneous graph — no atom graph, no atom-level message passing, just surface geometry and local chemical identity. The static-graph/precomputed-cache design is a deliberate engineering simplification (not a modeling choice) to keep the forward pass free of per-step kNN computation.

**Geometry-only variant dropped:** A geometry-only ablation (xyz+normals, 6D, no chemistry — testing whether surface shape alone carries ESP signal) was tried first and scored Pearson r=0.085 on the test set — essentially no signal (see `notebooks/baseline_comparison.ipynb`). Its checkpoint/eval outputs have been deleted and the `use_atom_features` toggle removed from the code; chemistry features (element+residue identity, not charge) are now always on, so this baseline is surface geometry **+** local chemical identity, not a pure geometry test.

**Scale caveat:** `notebooks/baseline_comparison.ipynb`'s full head-to-head table (DGCNN geom-only r=0.085 → DGCNN chem r=0.669 → CNN3D r=0.803 → Vacuum Coulomb r=0.863 → AttentionESPN r=0.890 → DistanceESPN r=0.897, best at this scale) was run on the older 110-protein test set / 836-protein training split, against the pre-full-dataset checkpoint names (`AttentionESPN 4/4/10`, `DistanceESPN 10/10/10`) — not the current 848-protein test split or the `attention_aa4_aq2_qq16`/`distance_aa8_aq2_qq24` full-dataset champions used everywhere else in `notebooks/`. The notebook itself flags that Distance beating Attention here is "expected to invert on the full ~8,500-protein dataset" — consistent with the full-vs-subset finding above. The geometry-only r=0.085 figure is still the one cited above since it predates and is unaffected by the full-dataset scale-up, but the rest of this baseline hierarchy has not been re-run against the current champions.

---

## Post-Training Analysis Findings

All notebooks in this section use the full-dataset champions (`attention_aa4_aq2_qq16`, `distance_aa8_aq2_qq24`, 848-protein test split) unless noted.

### AlphaFold Confidence vs. Model Error

**Status:** Complete. **Notebook:** `notebooks/alphafold_uncertainty_analysis.ipynb` (5.76M query-vertex rows, both champions, via `scripts/compute_confidence_error_stats.py`).

**Finding:** Pooled Spearman r(pLDDT, |error|) looks meaningful (−0.09 to −0.10) but **collapses to ~0 per-protein** (median r = +0.008/+0.013) — a between-protein confound (protein size), not a real within-protein spatial effect. PAE similarly reverses sign per-protein. Local ESP-field volatility is the one internally-consistent signal (~+0.04–0.06, both pooled and per-protein), though a two-predictor regression still explains <1% of vertex-level error variance. **Conclusion: AlphaFold's own confidence scores do not predict where the model errs at vertex level**, for either architecture. This directly informs (and is superseded by, as the size driver) the error-clustering finding below.

### Error Clustering — Structural/Chemical Correlates of High-Error Proteins

**Status:** Data and statistics complete; §12 "Summary" is an unfilled template ("*Fill in after reviewing the plots and tables above*") — no final prose synthesis was written, despite rigorous quantitative work throughout.

**Notebook:** `notebooks/error_clustering_analysis.ipynb` (848-protein master CSV, 34 features; Spearman correlations, PCA→UMAP/t-SNE, KMeans/DBSCAN/Agglomerative clustering, Kruskal-Wallis + Mann-Whitney tests).

**Finding:** Top error correlates are all size proxies (`num_nodes_total`, `n_heavy_atoms`, r≈0.62); normalizing by `esp_std` weakens but does not eliminate this. KMeans is best at k=2 (silhouette 0.317); DBSCAN disagrees entirely with KMeans/Agglomerative (ARI=0). The size-based 2-cluster split is highly significant (p~1e-50) and **survives size-normalization** — the same cluster is hardest for *both* architectures, i.e., Attention and Distance fail on the same proteins for the same underlying reason (size), not different architecture-specific weaknesses.

### Mesh Query Density — Direct High-Density Prediction vs. Sparse-Predict + RBF Interpolate

**Status:** Complete, with explicit decision.

**Notebook:** `notebooks/mesh_density_analysis.ipynb` (same champions, re-evaluated on graphs rebuilt at 10%/25% query density instead of the standard 5%; paired Wilcoxon, n=848).

**Finding:** Higher query density is strictly worse and strictly more expensive — Attention RMSE +13.8%/+63.5%, Distance RMSE +25.8%/+111.2% at 10%/25% density (p as low as 2e-140; 88–100% of proteins worse). Mechanism: `curvature_sampling()`'s shrinking spacing constraint pulls in out-of-distribution (flatter) query nodes as k grows — the model was never trained on that query-node distribution. Distance degrades ~1.7–1.9× more than Attention. **Decision: validates the existing sparse-predict (5%) + RBF-interpolate pipeline design** — direct dense prediction is not a viable shortcut without retraining on denser query graphs.

### AlphaFold vs. Experimental PDB Structures

**Status:** Complete, with explicit decision. Directly answers the SUMMER_PLAN.md "AlphaFold vs PDB Structure ESP Comparison" item.

**Notebook:** `notebooks/pdb_af_comparison_analysis.ipynb` (40 AF/PDB pairs, pLDDT≥80, resolution≤2.0Å, sequence-aligned Kabsch CA-RMSD — correcting an earlier naive-pairing bug — cross-structure RBF-interpolated ESP comparison, paired Wilcoxon n=40; see `pdb_comparison_dataset` context for the underlying ~41-pair dataset).

**Finding:** Attention shows **no significant degradation** on real PDB structures vs. AF structures (mean Δ RMSE = +0.018, p=0.572); Distance shows real but modest degradation (Δ=+0.152, p=0.023, ~12%). CA-RMSD is small (mean 0.43 Å) and correctly predicts ESP-field agreement, validating the comparison methodology itself. A counterintuitive negative correlation between CA-RMSD and RMSE-gap is flagged as a likely sequence-length confound, unresolved. **Decision: both models transfer reasonably well to experimental structures; Attention is favored for generalization.**

### AlphaFold Seed Conformational Sampling

**Status:** Complete at pilot scale (n=10 proteins); notebook explicitly flags results as not yet definitive at that scale. Answers the SUMMER_PLAN.md "Conformational Sampling via AlphaFold Seeds" item, at pilot scope.

**Notebook:** `notebooks/seed_conformation_analysis.ipynb` (10-protein pilot — 5 best/5 worst by test-split r — 5 ColabFold seeds each = 50 structures, full pipeline rerun, Kabsch-aligned inter-seed CA-RMSD + RBF-interpolated ESP-field agreement). Capped by the local ColabFold environment limit (fails deterministically ≳550–560 residues on the available GPU/jax combo — 7 mitigations tried and ruled out), which constrained the size of the "worst" group.

**Finding:** Worst-group mean inter-seed CA-RMSD = 22.4 Å vs. best-group 1.0 Å (Mann-Whitney p=0.056/0.016). Pooled across n=10, inter-seed structural disagreement correlates strongly with model RMSE (r=0.64–0.73, p<0.05). Models are **consistently** worse on structurally unstable proteins, not erratically worse — RMSE variance doesn't scale with disagreement. **Conclusion (pilot-scale): some model "error" reflects AlphaFold's own structural uncertainty** — a distinct explanation from the size-driven error-clustering finding above, not yet reconciled against it or scaled beyond n=10.

### Vertex-Level Error Breakdown

**Status:** Data and statistics complete; final "Summary" section is the same unfilled template as the error-clustering notebook — no written prose conclusion.

**Notebook:** `notebooks/vertex_error_analysis.ipynb` (848-protein cached vertex-level stats via `scripts/compute_vertex_error_stats.py`; cross-checked against existing RMSE metrics, diff <1e-3). Directly addresses the SUMMER_PLAN.md "Clean Up Information Loss Comparisons" item's three-way error decomposition.

**Finding:** Error is U-shaped across the ESP value range — worst at the extremes ([-15,-8) and [8,15) kT/e, ~3 kT/e) vs. mid-range (~1.1–1.7 kT/e). Direct query-node prediction beats RBF-interpolated full-mesh reconstruction in 98.2–98.5% of proteins (small but consistent gap, ~0.07–0.08 kT/e) — i.e. reconstruction loss is real but small relative to model loss. Moran's I (spatial autocorrelation of error) correlates strongly with whole-mesh RMSE (r=0.82–0.83): high-error proteins have spatially clustered ("hotspot") error, not diffuse error.

### Embedding & Attention Chemistry Analysis

**Status:** Findings stated (in a "Key Findings" block at the top of the notebook rather than a closing summary); §1c (cross-model embedding similarity) produces only a plot with no printed values or written interpretation.

**Notebook:** `notebooks/embedding_analysis.ipynb` — frozen-backbone probing of the Attention champion plus cross-model comparison against the Distance champion: embedding cosine similarity, per-element/per-residue AQ attention weights, cross-model embedding similarity.

**Finding:** Oxygen draws visibly higher and more variable attention than nitrogen despite comparable edge counts (e.g., Head 2: O 0.105±0.196 vs. N 0.008±0.040) — chemistry-driven, not abundance-driven. Same pattern at residue level: charged/polar residues (ARG, LYS, GLU, ASP) show higher-variance attention than hydrophobic ones. This is consistent with, and partially anticipates, the still-unfinished quantitative validation in decision notebook 16 (Attention Head Specialization) above.

