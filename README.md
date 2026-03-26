# Protein Structure ESP Prediction Thesis

This project investigates whether geometric deep learning models can learn geometry-conditioned approximations of protein electrostatic potential (ESP) fields directly from AlphaFold-predicted protein structures.  

It is organized to support reproducible pipelines for **ground-truth electrostatics computation**, **graph-based dataset preparation**, and **model training/evaluation**.

---

## Project Structure
```
ProteinStructure_ESP_Thesis/
├── src/
│ ├── acquisition/ # download AlphaFold proteins
│ ├── electrostatics/ # wrappers for PDB2PQR & APBS
│ ├── surface/ # SES mesh generation & ESP mapping
│ ├── data/ # dataset, graph building, transforms
│ ├── models/ # EGNN architecture and losses
│ ├── analysis/ # metrics & visualization
│ └── utils/ # general helpers (logging, I/O, configs)
├── scripts/ # executable pipelines (download, preprocess, train, evaluate)
├── data/ # raw, processed, and split data
├── outputs/ # figures, metrics, logs
├── tests/ # unit tests for core modules
├── .gitignore
└── README.md
```


---

## Key Pipelines

### 1. Ground-Truth Electrostatics
- Download protein structures (AlphaFold or PDB)  
- Assign charges/radii via PDB2PQR  
- Compute ESP with APBS  
- Generate surface meshes and snap ESP values to vertices  
- Store processed data for ML

### 2. Machine Learning Pipeline
- Load processed protein graphs (vertices + ESP)  
- Build node and edge representations for EGNN  
- Apply optional transforms (normalization, masking)  
- Train and evaluate EGNN to predict ESP  
- Analyze predictions vs APBS and AlphaFold confidence metrics

---

## Design Principles

- **`src/`**: reusable, modular logic  
- **`scripts/`**: entry points that orchestrate workflows  
- **`utils/`**: shared helper functions (I/O, logging, configs)  
- **`analysis/`**: metrics, evaluation, visualization, experiment summaries  
- **Dataset splits**: handled at the data loading stage to ensure reproducibility  
- **Graph construction**: done before model consumption, edges fixed per protein  

---

## Next Steps (Project Stage 1)
- Finalize folder structure and data logging  
- Implement graph building for SES vertices + ESP  
- Configure dataset, transforms, and dataloaders  
- Set up logging and evaluation scaffolding  

---

> This README can grow later with:
> - Installation instructions  
> - Example scripts  
> - Hyperparameters & training configs  
> - Detailed visualization methods