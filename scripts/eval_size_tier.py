"""
scripts/eval_size_tier.py

Size-generalization pilot, eval step: evaluate a checkpoint trained only on
large proteins (>=600 aa, scripts/build_size_tier_dataset.py) against the
medium (300-599 aa) or small (<300 aa) tier it never saw during training.
No retraining -- graphs for every tier already exist in the standard
full_protein_dataset/, so this reads straight from there.

Mirrors scripts/eval_mesh_density.py's checkpoint-loading pattern (rebuild
the exact architecture from the checkpoint's own saved model_config/
feature_spec, not live config.yaml) and its non-destructive output
convention: writes to a separate output directory, never touches the
source checkpoint's own test_metrics.json/test_predictions/.

Usage:
    python scripts/eval_size_tier.py \\
        /home/student/thesis/checkpoints/attention_size_large600 \\
        --id-file /home/student/thesis/outputs/size_tier_medium_ids.txt \\
        --tier-label medium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import ProteinGraphDataset
from src.data.transform import NormalizeESP
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.training.trainer import evaluate_test
from src.training.loss import ESPLoss

MAIN_DATA_ROOT = Path("/home/student/thesis/full_protein_dataset")


def _build_model(model_name: str, model_config: dict, feature_spec: dict, device: torch.device):
    common = dict(
        hidden_dim             = model_config["hidden_dim"],
        n_rbf                  = model_config["n_rbf"],
        n_bond_radial_rounds   = model_config["n_bond_radial_rounds"],
        n_aq_rounds             = model_config["n_aq_rounds"],
        n_qq_rounds             = model_config["n_qq_rounds"],
        agg                     = model_config["agg"],
        use_element_embedding   = model_config.get("use_element_embedding", True),
        use_residue_embedding   = model_config.get("use_residue_embedding", True),
        use_bond_edges          = model_config.get("use_bond_edges", True),
        use_radial_edges        = model_config.get("use_radial_edges", True),
        has_curvature           = feature_spec.get("query_curvature", False),
        has_normal              = feature_spec.get("query_normal", False),
    )
    if model_name == "distance":
        model = DistanceESPN(**common)
    else:
        model = AttentionESPN(**common, n_heads=model_config.get("n_heads", 4))
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a large-only-trained checkpoint against a held-out medium/small "
                     "size tier it never saw during training. No retraining."
    )
    parser.add_argument("checkpoint_dir", type=Path,
                        help="Checkpoint directory containing best_model.pt.")
    parser.add_argument("--data-root", type=Path, default=MAIN_DATA_ROOT,
                        help="Default: the standard full_protein_dataset/ -- every tier's "
                             "graphs already live there.")
    parser.add_argument("--id-file", type=Path, required=True,
                        help="Text file with the eval-tier protein IDs (one per line), e.g. "
                             "outputs/size_tier_medium_ids.txt or size_tier_small_ids.txt.")
    parser.add_argument("--tier-label", type=str, required=True,
                        help="Label for this tier, used in the output path, e.g. 'medium'.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Default: model_eval/size_tier_eval/<checkpoint_name>/<tier_label>/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_path = args.checkpoint_dir / "best_model.pt"
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model_name   = ckpt["model_name"]
    model_config = ckpt["model_config"]
    feature_spec = ckpt["feature_spec"]
    esp_mean     = ckpt["esp_mean"]
    esp_std      = ckpt["esp_std"]

    write_dir = args.output_root or (
        PROJECT_ROOT.parent / "model_eval" / "size_tier_eval"
        / args.checkpoint_dir.name / args.tier_label
    )
    write_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.checkpoint_dir.name} @ {args.tier_label} ===")
    print(f"  data_root: {args.data_root}")
    print(f"  writing to: {write_dir}  (source checkpoint dir untouched)")

    model = _build_model(model_name, model_config, feature_spec, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_ids = [
        line.strip() for line in args.id_file.read_text().splitlines() if line.strip()
    ]
    eval_ds = ProteinGraphDataset(eval_ids, args.data_root, rebuild=False)
    eval_ds.transform = NormalizeESP(esp_mean, esp_std)

    loss_fn = ESPLoss(pearson_weight=0.5)  # only used for the loss field, not selection
    extra_state = {"esp_mean": esp_mean, "esp_std": esp_std}

    results = evaluate_test(
        model, loss_fn, eval_ds, device, extra_state,
        checkpoint_dir  = write_dir,
        predictions_dir = write_dir / "test_predictions",
    )

    g = results["global"]
    print(f"  r={g['pearson_r']:.4f}  rmse={g['rmse']:.4f}  mae={g['mae']:.4f}  "
          f"n_proteins={g['n_proteins']}")


if __name__ == "__main__":
    main()
