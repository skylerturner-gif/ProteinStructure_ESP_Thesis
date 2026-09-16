# Protein & Amino Acid Reference

Background reference on the 20 standard amino acids, for use alongside the ESP/GNN
pipeline. Scope is deliberately narrowed to match what this project actually sees:

- **20 standard proteinogenic residues only** — the AlphaFold structures used here are
  predicted directly from a UniProt sequence, so no non-standard residues (e.g.
  selenocysteine), ligands, waters, or HETATM records appear. This matches
  `RESIDUE_VOCAB` in [`src/data/graph_builder.py`](src/data/graph_builder.py) exactly
  (20 entries + an `unknown` fallback that in practice is never populated).
- **Single-chain monomers, no metal ions.** Metal-coordinating behavior (His/Cys/Asp/Glu
  as ligands to Zn²⁺, Fe²⁺/Fe³⁺, etc.) is mentioned only in passing below — it is out of
  scope for this dataset, which contains no cofactors.

---

## 1. Names & abbreviations

| Name | 3-letter | 1-letter | Name | 3-letter | 1-letter |
|---|---|---|---|---|---|
| Alanine | Ala | A | Leucine | Leu | L |
| Arginine | Arg | R | Lysine | Lys | K |
| Asparagine | Asn | N | Methionine | Met | M |
| Aspartate (aspartic acid) | Asp | D | Phenylalanine | Phe | F |
| Cysteine | Cys | C | Proline | Pro | P |
| Glutamate (glutamic acid) | Glu | E | Serine | Ser | S |
| Glutamine | Gln | Q | Threonine | Thr | T |
| Glycine | Gly | G | Tryptophan | Trp | W |
| Histidine | His | H | Tyrosine | Tyr | Y |
| Isoleucine | Ile | I | Valine | Val | V |

---

## 2. General composition and typical function

Every residue shares the same backbone (amine N–Cα–carbonyl C=O); what's listed below
is the side chain (R-group) hung off Cα, and what that side chain is typically doing in
a folded protein.

| Residue | Side chain | Typical function |
|---|---|---|
| **Gly** | H (no side chain — achiral) | Maximum backbone flexibility; enables tight turns and conformations no other residue can reach; common in loops and at active sites needing close packing. |
| **Ala** | –CH₃ | Small, chemically inert methyl group; common helix "filler" with minimal steric or electronic influence. |
| **Val** | –CH(CH₃)₂ (β-branched) | Hydrophobic core packing; β-branching favors β-sheet strands. |
| **Leu** | –CH₂CH(CH₃)₂ | Hydrophobic core packing; abundant in coiled-coils (e.g. leucine zippers). |
| **Ile** | –CH(CH₃)CH₂CH₃ (β-branched, chiral) | Hydrophobic core packing, especially β-sheets. |
| **Pro** | Ring closes back onto the backbone N (pyrrolidine) | No backbone N–H donor; fixed φ (~ -60°) makes it conformationally rigid — breaks α-helices/β-sheets, common at turns and helix caps. |
| **Met** | –CH₂CH₂–S–CH₃ (thioether) | Hydrophobic packing; the universal translation-start residue; oxidation-sensitive. |
| **Phe** | –CH₂–C₆H₅ (benzyl, aromatic) | Hydrophobic core packing, π-stacking. |
| **Trp** | Indole ring (bicyclic aromatic + N–H) | Largest side chain; hydrophobic core and membrane-interface packing, π-stacking, occasional H-bond donor via indole N–H; intrinsic UV fluorescence. |
| **Tyr** | Phenol ring (aromatic + –OH) | Amphipathic: π-stacking like Phe/Trp, plus H-bonding via –OH; common kinase phosphorylation site; frequent at protein–protein interfaces. |
| **Ser** | –CH₂OH | H-bonding; classic nucleophile in serine-protease catalytic triads; phosphorylation site. |
| **Thr** | –CH(OH)CH₃ (β-branched) | H-bonding; phosphorylation and O-glycosylation site. |
| **Cys** | –CH₂SH (thiol) | Forms disulfide bonds that cross-link and stabilize tertiary/quaternary structure; nucleophile in cysteine proteases; classic metal ligand (out of scope here). |
| **Asn** | –CH₂–CONH₂ (carboxamide) | H-bonding; N-glycosylation sequon (Asn-X-Ser/Thr); frequent at surface turns. |
| **Gln** | –CH₂CH₂–CONH₂ | H-bonding; nitrogen-transport/metabolism donor; frequent at solvent-exposed surface. |
| **Asp** | –CH₂–COO⁻ (pKa ≈ 3.9) | Negatively charged at physiological pH; salt bridges, general acid/base catalysis, classic metal ligand (out of scope here). |
| **Glu** | –CH₂CH₂–COO⁻ (pKa ≈ 4.1) | Negatively charged; salt bridges, catalysis; strong α-helix former on the surface. |
| **Lys** | –(CH₂)₄–NH₃⁺ (pKa ≈ 10.5) | Positively charged; salt bridges, nucleic-acid binding; frequent PTM site (ubiquitination, acetylation, methylation). |
| **Arg** | Guanidinium group (pKa ≈ 12.5) | Strongest, most delocalized positive charge and most H-bond donors of any side chain; nucleic-acid and phosphate binding. |
| **His** | Imidazole ring (pKa ≈ 6.0) | Near-neutral pKa means its charge state depends on local environment — classic general acid/base catalyst (e.g. protease catalytic triads); classic metal ligand (out of scope here). |

---

## 3. Current classifications / groupings

The textbook five-way split, by side-chain chemistry at physiological pH:

- **Aliphatic / nonpolar:** Gly, Ala, Val, Leu, Ile, Pro, Met
- **Aromatic:** Phe, Trp, Tyr
- **Polar, uncharged:** Ser, Thr, Cys, Asn, Gln
- **Acidic (negative):** Asp, Glu
- **Basic (positive):** Lys, Arg, His

This single scheme undersells how residues actually get grouped in practice — different
properties matter for different questions. IMGT's amino-acid aide-mémoire (a
still-maintained immunogenetics reference) keeps several classification axes side by
side rather than picking one:

| Axis | Classes |
|---|---|
| **Polarity** (2-way) | Polar: R, N, D, Q, E, H, K, S, T, Y · Nonpolar: A, C, G, I, L, M, F, P, W, V |
| **Charge** (3-way) | Positive: R, H, K · Negative: D, E · Uncharged: everything else |
| **Hydropathy** (3-way) | Hydrophobic: A, C, I, L, M, F, W, V · Neutral: G, H, P, S, T, Y · Hydrophilic: R, N, D, Q, E, K |
| **Chemical group** (7-way) | Aliphatic: A, G, I, L, P, V · Aromatic: F, W, Y · Sulfur: C, M · Hydroxyl: S, T · Basic: R, H, K · Acidic: D, E · Amide: N, Q |
| **H-bonding role** | Donor: R, K, W · Acceptor: D, E · Both: N, Q, H, S, T, Y · Neither: A, C, G, I, L, M, F, P, V |
| **Volume** (5-way, Å³) | Very small: A, G, S · Small: N, D, C, P, T · Medium: Q, E, H, V · Large: R, I, L, K, M · Very large: F, W, Y |

**Residues that don't sit cleanly in any one bucket:**
- **Gly** has no side chain to classify — it's polarity-neutral by default and usually
  discussed separately as a flexibility/turn residue rather than by side-chain chemistry.
- **Pro** is sometimes called an "imino acid" rather than an amino acid, since its side
  chain is covalently closed back onto the backbone nitrogen — this is what gives it its
  rigidity, not a chemical property of an R-group.
- **Cys** is grouped nonpolar by strict hydropathy scales (its –SH is only weakly polar),
  but its disulfide-bond and metal-ligand reactivity make "special/reactive" a more
  useful bucket in practice.
- **His** sits on the boundary of basic and neutral because its imidazole pKa (~6.0) is
  close to physiological pH — whether it's charged depends on the local electrostatic
  environment, which is exactly the kind of thing this project's ESP model is trying to
  learn to predict around.

---

## 4. Inside vs. outside — measured from this dataset

Rather than relying on generic textbook exposure tables, this section measures burial
directly from the dataset used in this project via
[`scripts/analyze_residue_exposure.py`](scripts/analyze_residue_exposure.py).

**Method:** each protein's MSMS SES mesh already partitions the solvent-excluded
surface into vertices. Every vertex is assigned to its geometrically nearest heavy atom
(same nearest-atom assignment used in `scripts/survey_mesh_atom_overlap.py`), and its
share of surface area is approximated as `ses_area / n_verts` (MSMS vertices are
near-uniformly spaced at a fixed density, so this is a reasonable per-vertex weight
without re-deriving triangle areas from mesh faces). Per-atom areas are summed to
per-residue-instance areas, then divided by each residue type's theoretical maximum ASA
(Tien et al. 2013, Gly-X-Gly scale) to get a **relative solvent accessibility (RSA)**
per residue instance — the standard way to compare burial across residue types of very
different sizes. A residue instance is called **buried** if RSA < 25% and **exposed**
otherwise, the conventional two-state threshold (Rost & Sander, 1994).

Run over the **full dataset — all 8,461 proteins, ~3.49M residue instances**, zero
parse errors:

```
python scripts/analyze_residue_exposure.py --all --workers 8
```

| Residue | Instances | Mean area (Å²) | Mean RSA | Median RSA | % buried | % exposed |
|---|---:|---:|---:|---:|---:|---:|
| ARG | 194,806 | 108.70 | 0.397 | 0.422 | 12.9% | 87.1% |
| LYS | 217,808 | 88.94 | 0.377 | 0.390 | 15.6% | 84.5% |
| GLN | 159,849 | 79.65 | 0.354 | 0.376 | 16.9% | 83.2% |
| PRO | 194,011 | 55.97 | 0.352 | 0.408 | 23.3% | 76.7% |
| ASN | 156,097 | 68.07 | 0.349 | 0.375 | 20.6% | 79.4% |
| HIS | 82,923 | 76.83 | 0.343 | 0.364 | 27.9% | 72.1% |
| SER | 285,973 | 51.12 | 0.330 | 0.368 | 26.2% | 73.8% |
| GLU | 247,582 | 71.97 | 0.323 | 0.338 | 21.9% | 78.1% |
| THR | 183,577 | 53.47 | 0.311 | 0.336 | 33.1% | 66.9% |
| ASP | 188,934 | 59.20 | 0.307 | 0.321 | 29.5% | 70.5% |
| GLY | 220,454 | 31.58 | 0.304 | 0.338 | 34.3% | 65.7% |
| TYR | 98,057 | 77.07 | 0.293 | 0.281 | 43.7% | 56.3% |
| TRP | 37,705 | 80.15 | 0.281 | 0.249 | 50.1% | 49.9% |
| ALA | 254,051 | 35.70 | 0.277 | 0.313 | 41.4% | 58.6% |
| MET | 77,338 | 61.79 | 0.276 | 0.280 | 46.3% | 53.7% |
| PHE | 124,222 | 64.67 | 0.270 | 0.244 | 50.8% | 49.2% |
| LEU | 328,651 | 44.40 | 0.221 | 0.193 | 57.9% | 42.1% |
| VAL | 208,228 | 37.48 | 0.215 | 0.189 | 58.4% | 41.6% |
| CYS | 54,916 | 34.59 | 0.207 | 0.173 | 62.5% | 37.5% |
| ILE | 171,891 | 38.90 | 0.197 | 0.156 | 63.6% | 36.4% |

*(Sorted by mean RSA, most-exposed first. Full CSV: `/home/student/thesis/outputs/residue_exposure.csv`.)*

**Reading it:** the ranking lines up almost exactly with the classic hydropathy
classification above — charged and polar residues (Arg, Lys, Gln, Asn, Glu, Asp, His,
Ser) cluster at the exposed end, and the nonpolar aliphatic core-packers (Ile, Val,
Leu, Cys, Met) cluster at the buried end. A few notable departures from a naive
"charged = outside, hydrophobic = inside" story:

- **Pro** and **Gly** are both far more exposed than their (weak) hydrophobicity would
  suggest — consistent with their role as turn/loop residues rather than core packers;
  turns are, structurally, on the surface almost by definition.
- **Trp** and **Phe** sit close to 50/50 despite being aromatic/hydrophobic — large
  aromatic rings are common at membrane and protein–protein interfaces, not just buried
  cores, which pulls their average up relative to purely aliphatic residues like Ile/Val.
- **Cys** is the most buried polar-by-formula residue, consistent with disulfide bonds
  typically forming between residues held together in a folded (i.e. buried-adjacent)
  core rather than at floppy surface loops.

---

## Sources

- [20 Amino Acids: Structure, Classification & Abbreviation](https://www.proteinstructures.com/20-common-amino-acids/)
- [IMGT classes of the 20 common amino acids](https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html)
- Tien, M.Z., Meyer, A.G., Sydykova, D.K., Spielman, S.J., Wilke, C.O. (2013). "Maximum
  Allowed Solvent Accessibilites of Residues in Proteins." *PLOS ONE* 8(11): e80635.
  [PMC3836772](https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/)
- Rost, B., Sander, C. (1994). "Conservation and prediction of solvent accessibility in
  protein families." *Proteins* 20(3): 216–226. (Source of the 25% buried/exposed
  two-state RSA threshold used in section 4.)
- Row-level exposure data: computed directly from this project's dataset via
  `scripts/analyze_residue_exposure.py`, not an external source.
