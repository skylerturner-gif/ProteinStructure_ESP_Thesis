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

**Decision-to-be-Made:** Use base pdb file or add H with pdb2pqr/custom function.
**Problem:** Without the Hydrogens in the structure, the esp values sampled are too close to singularities creating large esp gradients. Are H's necessary for structural data and if so how do I tackle this?
**Options:** 
- Use base pdb with larger probes/offsets. This would keep the process fast and simple, but there are still singularities and much of the structure begins to smooth over.
- Use pdb2pqr to add Hydrogens. This would give the most accurate mesh to the actual protiens, but would require using another software for the final program and introduces licensing issues. pdb2pqr also adds other unnecessary steps, like charge and radius which I won't need in the final program. Need to explore offsets with Hydrogens added.
- Use custom Hydrogen placing function. This would allow for potentially a faster method as it would not need to add charge or radius. This is nontrivial and would take considerable time away from the core of the research.

### ESP Sampling — Vertices

**Decision-to-be-Made:** Use all vertices, evenly distributed subsample, or curvature weighted subsample.
**Problem:** The VRAM needed for the model scales by N^2, where N is the number of nodes. By using the vertices as query nodes, the number of query points explodes by protein size. How many query points can I scale back on and maintain an expressive and accurate representation of the esp? 
**Options:** 
- Use all vertices. This is the most intensive as some large proteins have 214,957 vertices and 7,057 heavy atoms, resulting in ~225,000 nodes with H.
- Use evenly distributed subsample. This would be the easiest and very nicely scalable by surface area or number of atoms, but may miss valuable insights like pockets or binding sites.
- Use curvature weighted subsample. This would require some tricky math to calculate the curvature of the meshes and set some kind of boundary of what is "curved," but would capture relevant features..

### ESP Sampling — Normal Offset

**Decision-to-be-Made:** Sample ESP at the vertices or points offset nÅ outward along vertex normals.
**Problem:** The surface vertex sits at the SES boundary, which can fall at a numerical discontinuity in the DX grid. Can I sample directly with proper logic checks or do I sample slightly farther way to ensure values are meaningful?
**Options:** 
- Sample at the surface exactly (0.0 Å). With proper checks this should be fine and the most accurate to the protein.
- A small outward offset samples the solvent-accessible region where ESP is physically meaningful for molecular interactions. Would need to try several values to find the proper amount and this could vary by protein size and shape

---

### ESP Sampling — Methods

**Decision-to-be-Made:** Use nearest-neighbour interpolation from the DX grid, Laplacian reconstruction from a subsampled set of constrained vertices, or Trilinear interpolation.
**Problem:** The .dx file is a voxel grid of values that normally don't line up with the SES mesh and vertices. How do I properly map esp values to my mesh without discontinuities or over smoothing?
**Options:** 
- Nearest-neighbour gives a direct but grid-resolution-limited mapping, this can cause singularities or discontinuities to snap to the vertex. 
- Laplacian reconstruction provides a smooth, mesh-intrinsic field that may be better suited as an EGNN training target, but is computationally heavy.
- Trilinear Interpolation is the middle ground and could solve both problems or just add to them.

---

### Protein Filtering — Sequence Length

**No global Min or Max size set yet.** Short peptides (< ~50 residues) may produce degenerate meshes, and large peptides (> ~500 residues) could blow VRAM and storage capabilities; this will be evaluated as data accumulates.

---

### Future Decisions

- EGNN architecture depth, hidden dimension, number of equivariant layers
- Attention based or distance based message passing
- Cross domain or fully connected query to atom node message passing
- multihead or multipass approach to lower VRAM requirements and smooth continuous fields
- Loss function (MSE on ESP values vs. correlation-based loss)
- Train/val/test split strategy (random vs. organism-stratified vs. sequence-identity clustered)
- Whether to normalise ESP values per-protein or globally across the dataset
- Whether to include sequence encoding as a node feature