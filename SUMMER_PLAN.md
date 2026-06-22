# SUMMER_PLAN.md

This file is the AI-agent reference for summer 2025 research direction. Tiers indicate priority order — work within a tier before moving to the next. Impact and effort ratings guide scope decisions when time is limited.

---

## Tier 1 — Foundations
*Must be completed first. Everything downstream depends on these.*

### Master CSV for Protein Info and Metrics
- **Impact:** High | **Effort:** Low
- Aggregate per-protein metadata JSONs into a single master CSV covering all pipeline stages: pLDDT stats, sequence length, atom count, net charge, surface area, mesh vertex count, RBF baseline Pearson r and RMSE, model prediction metrics (once available), and training split assignment.
- This CSV becomes the primary tool for distribution analysis, filtering decisions, and dataset curation across all subsequent tasks.
- **Depends on:** Existing metadata JSONs from data_gen_pipeline.

### Increase Dataset Size
- **Impact:** Very High | **Effort:** Medium
- Run `scripts/fetch_uniprot_ids.py` to generate `data/candidate_ids.txt` (~320k candidates), then feed batches through `data_gen_pipeline.py` and `model_pipeline.py`. May require zip/unzip tooling to manage disk space on external data root.
- Target: several thousand processed proteins with complete pipeline output (structure → ESP → graph) before training at scale.
- Distribution analysis (length, charge, pLDDT, surface area) using master CSV should guide which proteins to prioritize.
- **Depends on:** Master CSV, fetch_uniprot_ids.py (done).

### Rerun Tests with Correct Methods
- **Impact:** Very High | **Effort:** Low
- Current results may use incorrect configurations (wrong `inv_size`, query geometry features incorrectly enabled). Rerun all model evaluations with correct hyperparameters before drawing any conclusions or building on existing results.
- Specifically: verify `inv_size` normalization, confirm query geometry features (curvature, normal) are disabled unless ablation study, confirm QQ round counts match intended configuration.
- **Depends on:** Nothing — this unblocks everything else.

### Error Distribution by ESP Value
- **Impact:** High | **Effort:** Low
- Plot histogram of prediction error colored by ESP magnitude and net charge. Identify whether the model struggles with extreme positive/negative ESP values, deep pockets, or charge-adjacent surface features.
- This diagnostic directly informs which proteins or surface regions are hardest and should be weighted differently in loss or included in targeted ablations.
- **Depends on:** Rerun with correct methods.

---

## Tier 2 — Architecture Decisions
*High-value experiments that answer fundamental questions about the model and inform all future design choices.*

### Full-Body Rotation and Translation Test
- **Impact:** Very High | **Effort:** Low
- Since the model is not truly equivariant, test whether prediction quality is stable under random SO(3) rotations and translations of the input coordinates. If performance is stable: strong argument against the complexity cost of full equivariance. If performance degrades: implement coordinate augmentation during training (random rotations) and compare; this result also motivates investment in Tier 5 equivariant architectures.
- **Depends on:** Correct-method reruns.

### Feature Ablation Study
- **Impact:** High | **Effort:** Medium
- Systematically remove individual input features — bond type embedding, residue embedding, bond edges, radial edges, query geometry features (curvature, normal) — and measure impact on Pearson r and RMSE. Identifies which features are load-bearing vs redundant.
- Early results suggest query geometry features may be less important than expected; ablation will confirm this and guide architecture simplification or feature engineering.
- **Depends on:** Correct-method reruns, larger dataset.

### pLDDT Correlation with Model Error
- **Impact:** High | **Effort:** Medium
- Build a correlation matrix between per-vertex model prediction error and the AlphaFold pLDDT score of nearby residues. Requires mapping residue-level pLDDT (stored in metadata) to query node positions via nearest-residue assignment.
- This is a novel insight: if model error is elevated in low-confidence AF regions, it suggests the ESP field itself is less predictable there — not just a model failure.
- **Depends on:** Master CSV, pLDDT per-residue data (already in metadata), larger dataset.

### Feature Analysis Across All Three Architectures
- **Impact:** Very High | **Effort:** Medium
- Run `scripts/analyze_embeddings.py` on AttentionESPN, DistanceESPN, and any third variant. Early results show the two architectures learn qualitatively different internal representations despite similar predictive performance — quantify this with embedding cosine similarity, attention weight statistics, and cross-model representation comparison.
- This is a strong standalone finding and supports the broader argument that the problem has multiple valid inductive biases.
- **Depends on:** Larger dataset, trained models on same splits.

### Attention Heads → Chemistry, Physics, Dynamics
- **Impact:** Flagship | **Effort:** Medium
- If attention heads in AttentionESPN can be shown to preferentially weight atoms by chemical environment (charge type, hydrogen-bond donor/acceptor, hydrophobic vs polar), this connects learned representations to known biochemistry. Combine with `scripts/probe_charges.py` partial charge recovery results.
- This is the most publishable result on the roadmap. A figure showing attention head specialization tied to PARSE partial charge categories would be the centerpiece of a paper.
- **Depends on:** Feature analysis, probe_charges results.

---

## Tier 3 — Publication Credibility
*Required for the work to be publishable. Build these in parallel with Tier 2 where possible.*

### 3D CNN Baseline
- **Impact:** Very High | **Effort:** High
- Implement a trusted volumetric baseline — a 3D CNN that predicts ESP on a voxel grid or on surface vertices using local volumetric neighborhoods. Reviewers will require a non-GNN baseline to justify the graph approach.
- The baseline does not need to match the GNN; it needs to be clearly described and fairly evaluated on the same proteins with the same metrics.
- **Depends on:** Larger dataset, correct-method reruns.

### Clean Up Information Loss Comparisons
- **Impact:** Medium | **Effort:** Low
- The current evaluation conflates three distinct information loss sources: (1) GT grid → interpolated surface (sampling loss), (2) interpolated surface → model prediction from sparse query nodes (model loss), and (3) model query node predictions → reconstructed full surface (RBF reconstruction loss). Separate these clearly with distinct metrics and figures.
- **Depends on:** Master CSV.

### Protein Analysis Tools
- **Impact:** Medium | **Effort:** Low
- Add analysis utilities for: residue polarity distribution, atom/residue type counts, estimated protein volume (from mesh), hydrophobic/hydrophilic surface fraction. These metrics inform dataset curation and appear in dataset characterization sections of a paper.
- Add to master CSV generation.
- **Depends on:** Master CSV infrastructure.

### AlphaFold vs PDB Structure ESP Comparison
- **Impact:** High | **Effort:** High
- For proteins with known PDB structures, run the full APBS pipeline on both AF-predicted and experimental PDB structures. Compare ground-truth ESP fields and model predictions. If AF structures produce systematically different (or more unstable) ESP values in low-pLDDT regions, this validates the pLDDT correlation analysis and has direct publication value.
- Plot delta between AF-derived and PDB-derived APBS ESP for the same protein.
- **Depends on:** pLDDT correlation analysis, access to PDB structures for test proteins.

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
- **Depends on:** Feature ablation study results.

### Conformational Sampling via AlphaFold Seeds
- **Impact:** Medium | **Effort:** Very High
- Generate 5–10 slightly different structural conformations of the same protein using different AlphaFold random seeds, then run both APBS and the model on each. This tests whether ESP instability in low-pLDDT regions correlates with structural variation — a potential mechanistic explanation for the pLDDT–error correlation.
- **Depends on:** pLDDT correlation analysis.

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
| Master CSV | 1 | High | Low |
| Dataset scale-up | 1 | Very High | Medium |
| Correct-method reruns | 1 | Very High | Low |
| Error distribution analysis | 1 | High | Low |
| Rotation/translation test | 2 | Very High | Low |
| Feature ablation | 2 | High | Medium |
| pLDDT–error correlation | 2 | High | Medium |
| Embedding analysis (3 architectures) | 2 | Very High | Medium |
| Attention heads → chemistry | 2 | Flagship | Medium |
| 3D CNN baseline | 3 | Very High | High |
| Info-loss comparison cleanup | 3 | Medium | Low |
| Protein analysis tools | 3 | Medium | Low |
| AF vs PDB ESP comparison | 3 | High | High |
| Force field analysis | 4 | Medium | High |
| Global node | 4 | Medium | Medium |
| Conformational sampling | 4 | Medium | Very High |
| Limit testing | 4 | Medium | Medium |
| Size generalization (train/test) | 4 | Medium | Medium |
| Half-precision | 4 | Low | Low |
| True equivariance | 5 | High | Very High |
| Multi-global nodes | 5 | Medium | Very High |
| Cross-FF training | 5 | Low | Very High |
| Teacher-student transfer | 5 | High | Very High |
| MaSIF comparison | 5 | Medium | Very High |
