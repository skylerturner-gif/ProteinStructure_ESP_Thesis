# SUMMER_PLAN.md

This file is the AI-agent reference for summer 2025 research direction. Tiers indicate priority order — work within a tier before moving to the next. Impact and effort ratings guide scope decisions when time is limited.

---

## Tier 1 — Foundations
*Must be completed first. Everything downstream depends on these.*

### Master CSV for Protein Info and Metrics — DONE
- **Impact:** High | **Effort:** Low
- A master CSV (848 test proteins, 34+ features) is now the backing data source for `error_clustering_analysis.ipynb`, `vertex_error_analysis.ipynb`, and `alphafold_uncertainty_analysis.ipynb`, covering pLDDT stats, atom/node counts, and downstream model error metrics. Aggregate per-protein metadata JSONs into a single master CSV covering all pipeline stages: pLDDT stats, sequence length, atom count, net charge, surface area, mesh vertex count, RBF baseline Pearson r and RMSE, model prediction metrics, and training split assignment.
- This CSV is the primary tool for distribution analysis, filtering decisions, and dataset curation across all subsequent tasks.
- **Depends on:** Existing metadata JSONs from data_gen_pipeline.

### Increase Dataset Size — DONE
- **Impact:** Very High | **Effort:** Medium
- `scripts/fetch_uniprot_ids.py` was used to expand the dataset; the full ~8,461-protein dataset is trained and evaluated on locally — every post-restructuring sweep (Sweeps A–F) and the full-dataset champion models (`attention_aa4_aq2_qq16`, `distance_aa8_aq2_qq24`) use it, with an 848-protein held-out test split. `notebooks/decisions/15_full_vs_subset_comparison.ipynb` confirms scale itself is worth ~+0.02–0.03 Pearson r over the original 1,045-protein subset, holding architecture config fixed (see THESISPROCESSES.md → "Full-Dataset Champion Configuration").
- **Depends on:** Master CSV, fetch_uniprot_ids.py (done).

### Rerun Tests with Correct Methods — DONE
- **Impact:** Very High | **Effort:** Low
- The Sweep A–F restructuring (EMA always-on, protein-size-weighted loss always-on, `agg=multi`, 4/4/4-round baseline, query geometry features off) is exactly this correction, and it is now the standard every notebook trains and evaluates against — see THESISPROCESSES.md → "Current Core Training Configuration". `inv_size`/protein-weighted loss is hardcoded unconditional (`ESPLoss.forward()`); query geometry features are confirmed off by default and were separately re-tested in Sweep C (`11_query_features_ablation.ipynb`); QQ round counts were swept exhaustively in Sweep D (`13_query_rounds_sweep.ipynb`) rather than merely "confirmed."
- **Depends on:** Nothing — this unblocked everything else.

---

## Tier 2 — Architecture Decisions
*High-value experiments that answer fundamental questions about the model and inform all future design choices.*

### Full-Body Rotation and Translation Test — DONE
- **Impact:** Very High | **Effort:** Low
- Result: `notebooks/decisions/12_invariance_analysis.ipynb` (formerly `09_equivariance_reliance.ipynb`). Non-equivariant architecture is justified — query-features-off models are exactly SE(3)-invariant by construction (verified numerically); query-features-on models show negligible rotational instability (ΔPearson r < 0.001). No equivariant architecture (Tier 5) is warranted.

### Staged Ablation Study (supersedes "Feature Ablation Study") — Sweeps A–F, mostly complete
- **Impact:** High | **Effort:** Medium
- The original single feature-ablation task grew into a full staged ablation covering loss weighting, aggregation, query features, QQ/AA/AQ round counts, and chemistry-layer ablation across both architectures — see `notebooks/decisions/07`–`16` plus `09_ema_justification.ipynb`. Status per axis, against the shared EMA + protein-size-weighted, 4/4/4-round, `agg=multi` baseline:
  - **Sweep A** (loss pearson-weight, `07_loss_function_sweep.ipynb`) — **DONE.** Winner `pearson_weight=0.5` (`attention_pw05`), the baseline every later sweep is measured against. *Note: its cited test metrics have an unresolved small reproducibility gap on re-evaluation — see THESISPROCESSES.md → "Known Issue — Sweep A Baseline Reproducibility."*
  - **EMA justification** (`09_ema_justification.ipynb`) — **DONE.** EMA kept; modest but real benefit, 120-epoch budget confirmed sufficient.
  - **Sweep B** (aggregation, `08_message_aggregation_sweep.ipynb`) — **DONE.** `multi` wins clearly for both architectures on every tracked axis (val r, val RMSE, train/val gap) — already the adopted default.
  - **Sweep C** (query features, `11_query_features_ablation.ipynb`) — **DONE.** Re-tested under the new baseline; conclusion reframed from "features hurt" to "features neutral" — off is kept by a narrow margin plus the SE(3)-invariance argument (`12_invariance_analysis.ipynb`, itself flagged as needing a re-run against current checkpoints).
  - **Sweep D** (QQ rounds, `13_query_rounds_sweep.ipynb`) — **DONE.** Full sweep 0→32; ~100%-of-bins win at every step 0→12, plateauing after. Full-dataset champions use qq=16 (Attention) / qq=24 (Distance).
  - **Sweep E** (AA/AQ rounds, `14_structure_rounds_sweep.ipynb`) — **DONE**, 8/8 runs complete. AA rounds are a real second lever (`aa8` wins clearly, `aa12` mixed/diminishing); AQ rounds are near-redundant/saturated at baseline. Open note: the full-dataset Attention champion kept aa=4 (baseline) rather than carrying forward this sweep's aa8 winner — Distance's champion did adopt aa=8 — reason not recorded.
  - **Sweep F** (chemistry ablation, `10_chemistry_layer_ablations.ipynb`) — **DONE.** Residue embedding is nearly redundant; bond/radial/chemistry-stripped rungs cost real accuracy (chemistry-stripped and spatial-only worst by far, ~0.05–0.09 r). Also fixed a real bug found during this pass: §3's per-bin delta cells referenced an undefined variable and would `NameError` on re-run.
- **Remaining work:** none — all of Sweeps A–F now have written decisions.
- **Depends on:** Correct-method reruns (done), larger dataset (done).

### pLDDT Correlation with Model Error — DONE (result: no correlation)
- **Impact:** High | **Effort:** Medium
- `notebooks/alphafold_uncertainty_analysis.ipynb` built exactly this correlation (5.76M query-vertex rows, both champion architectures). **Result: pooled Spearman r(pLDDT, |error|) looks meaningful (−0.09 to −0.10) but collapses to ~0 per-protein (median r≈+0.01) — a between-protein size confound, not a real within-protein effect.** PAE similarly reverses sign per-protein. Local ESP-field volatility is a real but tiny contributor (~+0.04–0.06). A two-predictor regression explains <1% of vertex-level error variance. AlphaFold's own confidence does not predict where the model errs.
- **Follow-on, also DONE:** `notebooks/error_clustering_analysis.ipynb` then asked what *does* correlate with protein-level error (AF confidence excluded) — answer: protein size, overwhelmingly (r≈0.62 with heavy-atom count), and this survives size-normalization. The same cluster of proteins is hardest for both architectures. Its own quantitative work is complete but the notebook's final "Summary" section was never written.
- **Depends on:** Master CSV (done), pLDDT per-residue data (done), larger dataset (done).

### Feature Analysis Across Both Architectures — DONE
- **Impact:** Very High | **Effort:** Medium
- `notebooks/embedding_analysis.ipynb` and `notebooks/partial_charge_probe.ipynb` (both full-dataset scale, 848 test proteins) cover this. **Findings:** attention weights are chemistry-driven, not abundance-driven — e.g. oxygen draws visibly higher and more variable attention than nitrogen despite comparable edge counts, and charged/polar residues (ARG, LYS, GLU, ASP) show higher-variance attention than hydrophobic ones. The partial-charge probe shows both architectures recover PARSE partial charges from frozen embeddings near-perfectly (Attention R²=0.9987, Distance R²=0.9981), with Attention measurably better (16% lower RMSE) despite both being near-ceiling.
- Cross-model embedding similarity (§1c of `embedding_analysis.ipynb`) produced a plot but no written interpretation — the one loose end.
- **Depends on:** Larger dataset (done), trained models on same splits (done).

### Attention Heads → Chemistry, Physics, Dynamics — DONE (the flagship result)
- **Impact:** Flagship | **Effort:** Medium
- `notebooks/decisions/16_attention_head_analysis.ipynb`, §1–13 all complete (this line was stale — §11–12's "real test" has in fact been run). Attention organizes measurably around **charge sign** specifically, peaking at n_heads=4: the quantitative §11–12 clustering-from-raw-attention-alone test recovers charge sign strongly at n_heads=4 (ARI 0.468→**0.813**→0.506 for 2/4/8 heads, a striking non-monotonic peak), recovers the 5-way chemistry classification only weakly everywhere (ARI 0.12–0.26), and does not recover solvent exposure at all (ARI≈−0.089) despite exposure modulating attention *magnitude*. Retrained on slim graphs at a uniform 75-epoch budget: test r 0.9335/0.9344/0.9356 for 2/4/8 heads — heads8 now edges out heads4 by +0.0012 (within this project's noise band), reopening but not overturning the earlier "n_heads=4 champion" framing on raw accuracy; the internal-structure finding (charge-sign specialization peaking at n_heads=4) is separate from and not explained by that accuracy ranking. See THESISPROCESSES.md → "Model Validation — Attention Head Specialization" for full detail.
- **Depends on:** Feature analysis (done), probe_charges results (done).

---

## Tier 3 — Publication Credibility
*Required for the work to be publishable. Build these in parallel with Tier 2 where possible.*

### Non-GNN Baselines — DONE
- **Impact:** Very High | **Effort:** High
- Three non-GNN reference points now exist for the write-up, each isolating a different comparison axis against the heterogeneous-graph model — full architecture and rationale in THESISPROCESSES.md → "Baseline Models — Non-GNN Reference Points":
  - **Vacuum Coulomb** (`src/baseline_models/coulomb/`, `legacy/checkpoints/baseline/coulomb`) — no learning; analytical physics floor from PARSE partial charges alone, with no solvent-screening/dielectric-boundary physics.
  - **3D CNN** (`src/baseline_models/cnn3d/`, `legacy/checkpoints/baseline/cnn3d`) — dense voxel-grid 3D U-Net over an element-occupancy grid (C/N/O/S, no partial charges), queried via trilinear `grid_sample`.
  - **Surface DGCNN** (`src/baseline_models/surface_dgcnn/`, `legacy/checkpoints/baseline/surface_dgcnn_chem`) — EdgeConv point-cloud model on the mesh surface only; base variant is geometry-only (xyz+normals), `_chem` variant adds nearest-atom element/residue identity (still no partial charges).

### Clean Up Information Loss Comparisons — DONE
- **Impact:** Medium | **Effort:** Low
- `notebooks/vertex_error_analysis.ipynb` separates model loss from RBF reconstruction loss directly: direct query-node prediction beats RBF-interpolated full-mesh reconstruction in 97.4% (attention) / 98.3% (distance) of proteins (small but consistent gap, ~0.07–0.08 kT/e, ~95% of total error is genuine model error) — reconstruction loss is real but small relative to model loss. It also found error is U-shaped across the ESP range (worst at extremes) and that spatial error clustering (Moran's I) correlates strongly with whole-mesh RMSE (Spearman r=0.831–0.832). `notebooks/mesh_density_analysis.ipynb` complements this by showing higher query density (10%/25%) makes predictions strictly *worse*, validating the current 5%-sparse-predict + interpolate design rather than treating density as a free knob. `vertex_error_analysis.ipynb`'s final prose summary is now written (2026-09-22); `mesh_density_analysis.ipynb`'s summary status wasn't re-checked in this pass — THESISPROCESSES.md's own "Mesh Query Density" section describes it as complete with an explicit decision, which may mean this line was already stale before today.
- **Depends on:** Master CSV (done).

### Protein Analysis Tools
- **Impact:** Medium | **Effort:** Low
- Add analysis utilities for: residue polarity distribution, atom/residue type counts, estimated protein volume (from mesh), hydrophobic/hydrophilic surface fraction. These metrics inform dataset curation and appear in dataset characterization sections of a paper.
- Add to master CSV generation.
- **Depends on:** Master CSV infrastructure.

### AlphaFold vs PDB Structure ESP Comparison — DONE
- **Impact:** High | **Effort:** High
- `notebooks/pdb_af_comparison_analysis.ipynb` ran this on 40 sequence-aligned AF/PDB pairs (pLDDT≥80, resolution≤2.0Å). **Result:** Attention shows no significant degradation on real PDB structures (Δ RMSE=+0.018, p=0.572); Distance shows real but modest degradation (Δ=+0.152, p=0.023, ~12%). CA-RMSD is small (mean 0.43 Å) and correctly predicts ESP-field agreement, validating the methodology. A counterintuitive negative correlation between CA-RMSD and RMSE-gap is flagged as an unresolved likely sequence-length confound. **Conclusion: both models transfer to experimental structures; Attention is favored for generalization** — direct publication value as planned.
- **Depends on:** pLDDT correlation analysis (done — see above; this notebook's own framing deliberately treats AF/PDB structural transfer as separate from the pLDDT-error null result), access to PDB structures for test proteins (done — see `pdb_comparison_dataset`).

---

## Tier 4 — Interesting / Moderate Priority
*Valuable experiments but not on the critical path. Pick these up when Tier 2 and 3 are underway.*

### Force Field Analysis (PARSE vs Other FF)
- **Impact:** Medium | **Effort:** High
- Currently using PARSE force field at pH 7.0. Different force fields (CHARMM, AMBER, GROMOS) produce different partial charges and atomic radii, leading to different meshes and ESP fields. The model implicitly learns PARSE-specific representations.
- Run a subset of the dataset through alternative force fields, compare mesh and ESP differences, and test whether the current model generalizes across FF types.
- **Depends on:** Larger dataset, master CSV.

### Global Node
- **Impact:** Medium | **Effort:** Medium
- Add a single global node to the heterogeneous graph that aggregates messages from all atom and query nodes, then broadcasts back. This creates a low-cost mechanism for long-range information flow without full pairwise edges. Test whether global features (net charge, protein size) improve predictions on charged or large proteins.
- Build and validate single global node before attempting multi-global-node variant.
- **Depends on:** Staged ablation study results (Sweeps A–F).

### Conformational Sampling via AlphaFold Seeds — DONE at pilot scale
- **Impact:** Medium | **Effort:** Very High
- `notebooks/seed_conformation_analysis.ipynb` ran this at pilot scale: 10 proteins (5 best/5 worst by test-split r), 5 ColabFold seeds each = 50 structures, full pipeline rerun, Kabsch-aligned inter-seed CA-RMSD + RBF-interpolated ESP-field agreement. Scale was capped by a hard local environment limit (ColabFold fails deterministically ≳550–560 residues on the available GPU/jax combo — 7 mitigations tried and ruled out; see project memory `colabfold_environment_limit`). **Result:** worst-group mean inter-seed CA-RMSD = 22.4 Å vs. best-group 1.0 Å (p=0.056/0.016); inter-seed structural disagreement correlates strongly with model RMSE (r=0.64–0.73, p<0.05) — models are consistently (not erratically) worse on structurally unstable proteins. Notebook explicitly flags this as pilot-scale, not definitive — scaling past n=10 is blocked by the same ColabFold residue-length ceiling.
- Since the pLDDT-correlation analysis (above) came back null, this notebook's finding is now the primary evidence that some model "error" reflects genuine AlphaFold structural uncertainty rather than model failure — worth reconciling with the error-clustering notebook's competing size-driven explanation in the eventual write-up.
- **Depends on:** pLDDT correlation analysis (done).

### Graph Size and Limit Testing
- **Impact:** Medium | **Effort:** Medium
- Test model performance on fully connected small and medium proteins (remove sparsification). Profile memory and runtime. Test chunked/partitioned graphs for large proteins and determine whether discontinuous graph boundaries create artifacts in predicted ESP.
- **Depends on:** Optimization infrastructure.

### Train Small → Test Large / Train Large → Test Small
- **Impact:** Medium | **Effort:** Medium
- Two clean ablations: (1) train only on proteins ≤300 residues, evaluate on larger proteins — does the model learn local geometry that transfers? (2) reverse. These tests characterize the inductive biases of the architecture and whether learned features generalize across length scales.
- **Depends on:** Larger dataset with good length distribution.

### Half-Precision Training
- **Impact:** Low | **Effort:** Low
- Switch to `torch.float16` or `bfloat16` during training. Track memory reduction and any precision loss in Pearson r / RMSE. Quick win for fitting larger batches or larger models in the same VRAM budget.
- **Depends on:** Stable training setup.

---

## Tier 5 — Stretch / Long-Term
*Only pursue if time permits or if specific Tier 2 results make them necessary.*

### True Equivariance (SE(3)-Transformer or Equiformer)
- **Impact:** High | **Effort:** Very High
- If the rotation/translation test (Tier 2) shows significant performance degradation under coordinate transformation, implement a truly equivariant architecture (SE(3)-Transformer or Equiformer) and compare directly with AttentionESPN. Only worth the engineering cost if the non-equivariant model is demonstrably coordinate-sensitive.
- **Depends on:** Rotation/translation test result.

### Multi-Global Nodes
- **Impact:** Medium | **Effort:** Very High
- Extend the single global node to a chain: first global node aggregates all atom messages (input to query stage), second global node aggregates all query messages (input to QQ stage). Hypothesis: staged global aggregation with directional edges enables long-range charge interaction modeling. Only build after single global node is validated.
- **Depends on:** Global node (Tier 4).

### Cross-FF Training
- **Impact:** Low–Medium | **Effort:** Very High
- Attempt to train a single model across multiple force fields simultaneously, using a global node or embedding flag to condition on FF type. Almost certainly won't generalize perfectly, but tests the hypothesis and is due diligence.
- **Depends on:** Force field analysis (Tier 4).

### Teacher-Student Transfer (Pocket Embeddings → Sparse Model)
- **Impact:** High | **Effort:** Very High
- Train a detailed teacher model on small, fully-connected proteins to learn high-fidelity local geometry embeddings. Use knowledge distillation to transfer those embeddings into the sparse large-protein model. This is a potential second paper; do not pursue until Tier 2 and 3 are complete.
- **Depends on:** Graph size limit testing, stable training setup.

### MaSIF Comparison
- **Impact:** Medium | **Effort:** Very High
- Compare predicted ESP values with MaSIF surface fingerprints for geometry and function analysis. Could strengthen the hypothesis that ESP, geometry, and protein function are jointly encoded on the molecular surface. Useful for a broader framing but not required for the core results.
- **Depends on:** Core results complete, MaSIF environment setup.

---

## Quick Reference: Priority Matrix

| Task | Tier | Impact | Effort |
|---|---|---|---|
| Master CSV — DONE | 1 | High | Low |
| Dataset scale-up — DONE | 1 | Very High | Medium |
| Correct-method reruns — DONE | 1 | Very High | Low |
| Rotation/translation test — DONE (re-run on current checkpoints 2026-09-22) | 2 | Very High | Low |
| Staged ablation (Sweeps A–F) — all six DONE with written decisions | 2 | High | Medium |
| pLDDT–error correlation — DONE (result: null; size drives error instead) | 2 | High | Medium |
| Embedding analysis (2 architectures) — DONE | 2 | Very High | Medium |
| Attention heads → chemistry — DONE (§11–12 complete: charge-sign specialization peaks at n_heads=4) | 2 | Flagship | Medium |
| Non-GNN baselines — DONE (scale caveat: not re-run on full dataset) | 3 | Very High | High |
| Info-loss comparison cleanup — DONE | 3 | Medium | Low |
| Protein analysis tools | 3 | Medium | Low |
| AF vs PDB ESP comparison — DONE | 3 | High | High |
| Force field analysis | 4 | Medium | High |
| Global node | 4 | Medium | Medium |
| Conformational sampling — DONE at pilot scale (n=10) | 4 | Medium | Very High |
| Limit testing | 4 | Medium | Medium |
| Size generalization (train/test) | 4 | Medium | Medium |
| Half-precision | 4 | Low | Low |
| True equivariance | 5 | High | Very High |
| Multi-global nodes | 5 | Medium | Very High |
| Cross-FF training | 5 | Low | Very High |
| Teacher-student transfer | 5 | High | Very High |
| MaSIF comparison | 5 | Medium | Very High |
