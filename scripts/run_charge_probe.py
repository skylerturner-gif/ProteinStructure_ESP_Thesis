"""
scripts/run_charge_probe.py

Train (or load a cached) partial-charge probe for each of the two
full-dataset champion checkpoints, evaluate on the test split, and cache
compact, notebook-ready artifacts under /home/student/thesis/outputs/charge_probes/ so
notebooks/partial_charge_probe.ipynb only ever loads results rather than
re-running training/inference itself.

Per model:
  1. Train (seeded, reproducible) or load a cached probe -- see
     src.analysis.charge_probe.train_probe, which precomputes frozen-
     backbone embeddings once and reuses them across epochs.
  2. Evaluate on the full test split (src.analysis.charge_probe.evaluate_probe).
  3. Collect per-atom (element, true charge, predicted charge) data once,
     used to build the PARSE-grounded per-charge-class accuracy table and a
     subsampled scatter-plot dataset (full atom-level data is millions of
     rows -- too large and unnecessary to cache in full for plotting).

Outputs per model, under /home/student/thesis/outputs/charge_probes/:
    {model}_probe_{layer}.pt        -- trained probe + training metadata
    {model}_eval_summary.json       -- evaluate_probe's global + per_protein dict
    {model}_charge_classes.csv      -- per (element, charge-class) accuracy table
    {model}_scatter_sample.npz      -- subsampled true/pred/element arrays for plotting

Run in the `pyg_env` conda environment.

Usage:
    conda activate pyg_env
    python scripts/run_charge_probe.py
    python scripts/run_charge_probe.py --limit-test 50   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.charge_probe import (
    ChargeProbe,
    evaluate_probe,
    extract_atom_embeddings,
    load_parse_charge_reference,
    read_pqr_atoms,
    train_probe,
)
from src.analysis.embedding_analysis import ELEMENT_NAMES, _load_graph, load_model_frozen
from src.data.dataset import load_split_manifest
from src.utils.config import get_config, get_data_root
from src.utils.helpers import get_pipeline_logger
from src.utils.paths import ProteinPaths

CKPT_ROOT = Path("/home/student/thesis/checkpoints/full_dataset")
PROBE_MODELS = {
    "attention": CKPT_ROOT / "attention_aa4_aq2_qq16",
    "distance":  CKPT_ROOT / "distance_aa8_aq2_qq24",
}

DEFAULT_LAYER          = "after_mp"
DEFAULT_EPOCHS         = 30
DEFAULT_LR             = 1e-3
DEFAULT_SEED           = 42
DEFAULT_N_TRAIN_SAMPLE = 500
DEFAULT_SCATTER_SAMPLE = 100_000


def _collect_atom_data(
    probe: ChargeProbe, backbone, protein_ids: list[str], data_root: Path,
    layer: str, device: torch.device, log,
) -> pd.DataFrame:
    """Per-atom (protein_id, atom_name, res_name, element, true/pred charge)
    for every atom in protein_ids, one pass."""
    probe_gpu = probe.to(device)
    probe_gpu.eval()

    rows = []
    n_done = 0
    with torch.no_grad():
        for pid in protein_ids:
            paths = ProteinPaths(pid, data_root)
            if not paths.pqr_path.exists():
                continue
            charges, atom_names, res_names = read_pqr_atoms(paths.pqr_path)
            data   = _load_graph(pid, data_root)
            h_atom = extract_atom_embeddings(backbone, data, layer=layer, device=device)
            if h_atom.shape[0] != len(charges):
                continue
            pred = probe_gpu(h_atom.to(device)).cpu().numpy()
            elem = data["atom"].atom_type.cpu().numpy()
            for aname, rname, el, true_q, pred_q in zip(atom_names, res_names, elem, charges, pred):
                rows.append((pid, aname, rname, int(el), float(true_q), float(pred_q)))

            n_done += 1
            if n_done % 50 == 0 or n_done == len(protein_ids):
                print(f"\r    atom-data collection: {n_done}/{len(protein_ids)}", end="", flush=True)
    print()
    log.info("Collected atom data for %d/%d proteins (%d atoms)", n_done, len(protein_ids), len(rows))

    return pd.DataFrame(
        rows, columns=["protein_id", "atom_name", "res_name", "element", "true_charge", "pred_charge"]
    )


def _element_name(idx: int) -> str:
    return ELEMENT_NAMES[idx] if 0 <= idx < len(ELEMENT_NAMES) else "unknown"


def _build_charge_class_table(df: pd.DataFrame, charge_to_pairs: dict[float, list[str]]) -> pd.DataFrame:
    d = df.copy()
    d["charge_class"] = d["true_charge"].round(3)
    d["element_name"] = d["element"].map(_element_name)

    def _label(charge_class: float, element_name: str) -> str:
        matches = sorted(charge_to_pairs.get(charge_class, []))
        example = ", ".join(matches[:3]) + ("..." if len(matches) > 3 else "")
        return f"{element_name} q={charge_class:+.3f} ({example})" if example else f"{element_name} q={charge_class:+.3f}"

    rows = []
    for (elem_name, q), g in d.groupby(["element_name", "charge_class"]):
        t_arr, p_arr = g["true_charge"].to_numpy(), g["pred_charge"].to_numpy()
        n = len(g)
        rmse = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))
        mae  = float(np.mean(np.abs(p_arr - t_arr)))
        r = (float(np.corrcoef(t_arr, p_arr)[0, 1])
             if n > 1 and t_arr.std() > 0 and p_arr.std() > 0 else float("nan"))
        rows.append({
            "Label": _label(q, elem_name), "Element": elem_name, "Charge class": q,
            "N atoms": n, "Mean charge": float(t_arr.mean()), "Charge std": float(t_arr.std()),
            "RMSE": rmse, "MAE": mae, "Pearson r": r,
        })
    return pd.DataFrame(rows).sort_values(["Element", "Charge class"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/load + evaluate the partial-charge probe for both "
                     "full-dataset champions, caching compact notebook-ready artifacts."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/student/thesis/outputs/charge_probes"))
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER, choices=["after_encoder", "after_mp"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-train-sample", type=int, default=DEFAULT_N_TRAIN_SAMPLE)
    parser.add_argument("--scatter-sample", type=int, default=DEFAULT_SCATTER_SAMPLE)
    parser.add_argument("--limit-test", type=int, default=None,
                         help="Only evaluate on the first N test proteins (smoke test).")
    parser.add_argument("--force", action="store_true",
                         help="Retrain even if a cached probe already exists.")
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    log = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PARSE charge reference...")
    parse_reference = load_parse_charge_reference()
    charge_to_pairs: dict[float, list[str]] = defaultdict(list)
    for (resname, atomname), charge in parse_reference.items():
        charge_to_pairs[round(charge, 3)].append(f"{resname}.{atomname}")
    print(f"  {len(parse_reference)} PARSE entries, {len(charge_to_pairs)} distinct charge values")

    train_ids, val_ids, test_ids = load_split_manifest(data_root)
    if args.limit_test is not None:
        test_ids = test_ids[: args.limit_test]
    sampled_train_ids = random.Random(args.seed).sample(train_ids, min(args.n_train_sample, len(train_ids)))
    print(f"Train pool: {len(train_ids)} -> seeded sample of {len(sampled_train_ids)} (seed={args.seed})")
    print(f"Test split: {len(test_ids)} proteins")

    for name, ckpt_dir in PROBE_MODELS.items():
        print(f"\n=== {name} ({ckpt_dir.name}) ===")
        if not ckpt_dir.exists():
            print(f"  checkpoint not found: {ckpt_dir} -- skipping")
            continue

        backbone, ckpt = load_model_frozen(ckpt_dir, device)
        hidden_dim = ckpt["model_config"]["hidden_dim"]
        print(f"  backbone={ckpt['model_name']}  hidden_dim={hidden_dim}")

        probe_path = args.output_dir / f"{name}_probe_{args.layer}.pt"
        if probe_path.exists() and not args.force:
            saved = torch.load(probe_path, weights_only=False)
            probe = ChargeProbe(saved["hidden_dim"])
            probe.load_state_dict(saved["state_dict"])
            print(f"  loaded cached probe -> {probe_path} "
                  f"(trained on {len(saved['train_protein_ids'])} proteins, seed={saved['seed']})")
        else:
            print(f"  training on {len(sampled_train_ids)} proteins for {args.epochs} epochs...")
            torch.manual_seed(args.seed)
            probe = ChargeProbe(hidden_dim)
            probe = train_probe(
                probe, backbone, sampled_train_ids, data_root,
                layer=args.layer, device=device, epochs=args.epochs, lr=args.lr,
            )
            torch.save({
                "state_dict": probe.state_dict(), "hidden_dim": hidden_dim,
                "layer": args.layer, "seed": args.seed, "n_train_sample": args.n_train_sample,
                "train_protein_ids": sampled_train_ids,
            }, probe_path)
            print(f"  saved -> {probe_path}")

        print(f"  evaluating on {len(test_ids)} test proteins...")
        results = evaluate_probe(probe, backbone, test_ids, data_root, layer=args.layer, device=device)
        g = results["global"]
        print(f"  RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  Mean R²={g['mean_r2']:.4f}  "
              f"proteins={g['n_proteins']}  atoms={g['n_atoms']:,}")
        with open(args.output_dir / f"{name}_eval_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"  collecting per-atom data...")
        atom_df = _collect_atom_data(probe, backbone, test_ids, data_root, args.layer, device, log)

        charge_class_df = _build_charge_class_table(atom_df, charge_to_pairs)
        assert charge_class_df["N atoms"].sum() == len(atom_df), "atom count mismatch after grouping"
        charge_class_df.to_csv(args.output_dir / f"{name}_charge_classes.csv", index=False)
        print(f"  {len(charge_class_df)} (element, charge-class) groups -> "
              f"{name}_charge_classes.csv")

        n_scatter = min(args.scatter_sample, len(atom_df))
        sample_df = atom_df.sample(n=n_scatter, random_state=args.seed)
        np.savez_compressed(
            args.output_dir / f"{name}_scatter_sample.npz",
            true_charge=sample_df["true_charge"].to_numpy(dtype=np.float32),
            pred_charge=sample_df["pred_charge"].to_numpy(dtype=np.float32),
            element=sample_df["element"].to_numpy(dtype=np.int64),
        )
        print(f"  scatter sample ({n_scatter:,} of {len(atom_df):,} atoms) -> "
              f"{name}_scatter_sample.npz")

    log.info("run_charge_probe complete")
    print("\nDone.")


if __name__ == "__main__":
    main()
