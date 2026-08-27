"""
scripts/eval_mesh_density.py

Evaluate an already-trained checkpoint against test-set graphs built at a
DIFFERENT query-vertex density than the one it was trained on (e.g. 10%/25%
instead of the standard 5%) — no retraining. Tests whether the model, asked
to predict directly at higher density, beats today's sparse-predict-then-
RBF-interpolate approach (scripts/analyze_model.py --reconstruction).

Prerequisites (run first):
    1. scripts/resample_query_density.py  — regenerate query_idx at the new
       density for the test-set protein IDs, into an isolated data root.
    2. scripts/rebuild_graphs_for_ids.py --data-root <isolated root>
       --id-file <same test-id file>  — build graphs at the new density.

Mirrors scripts/reevaluate_test_timing.py's checkpoint-loading pattern
(rebuild the exact architecture from the checkpoint's own saved
model_config/feature_spec, not live config.yaml) and its non-destructive
output convention: writes to a separate output directory, never touches
the source checkpoint's test_metrics.json/test_predictions/ (see
EVAL_REPRODUCIBILITY_INVESTIGATION.md for why that matters).

Usage:
    python scripts/eval_mesh_density.py \\
        checkpoints/full_dataset/attention_aa4_aq2_qq16 \\
        --density-root /home/student/thesis/full_protein_dataset_density10 \\
        --id-file /tmp/test_ids.txt \\
        --density-label density_10
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
        description="Evaluate a trained checkpoint against test graphs built at a different "
                     "query-vertex density. No retraining."
    )
    parser.add_argument("checkpoint_dir", type=Path,
                        help="Checkpoint directory containing best_model.pt.")
    parser.add_argument("--density-root", type=Path, required=True,
                        help="Isolated data root with graphs already rebuilt at the new "
                             "density (see scripts/resample_query_density.py + "
                             "scripts/rebuild_graphs_for_ids.py).")
    parser.add_argument("--id-file", type=Path, required=True,
                        help="Text file with the test-set protein IDs (one per line) — "
                             "the same list used to build --density-root.")
    parser.add_argument("--density-label", type=str, required=True,
                        help="Label for this density, used in the output path, "
                             "e.g. 'density_10'.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Default: model_eval/mesh_density_eval/<checkpoint_name>/"
                             "<density_label>/")
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
        PROJECT_ROOT.parent / "model_eval" / "mesh_density_eval"
        / args.checkpoint_dir.name / args.density_label
    )
    write_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.checkpoint_dir.name} @ {args.density_label} ===")
    print(f"  density_root: {args.density_root}")
    print(f"  writing to:   {write_dir}  (source checkpoint dir untouched)")

    model = _build_model(model_name, model_config, feature_spec, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ids = [
        line.strip() for line in args.id_file.read_text().splitlines() if line.strip()
    ]
    test_ds = ProteinGraphDataset(test_ids, args.density_root, rebuild=False)
    test_ds.transform = NormalizeESP(esp_mean, esp_std)

    loss_fn = ESPLoss(pearson_weight=0.5)  # only used for the loss field, not selection
    extra_state = {"esp_mean": esp_mean, "esp_std": esp_std}

    results = evaluate_test(
        model, loss_fn, test_ds, device, extra_state,
        checkpoint_dir  = write_dir,
        predictions_dir = write_dir / "test_predictions",
        bf16            = True,
    )

    g = results["global"]
    print(f"  r={g['pearson_r']:.4f}  rmse={g['rmse']:.4f}  mae={g['mae']:.4f}  "
          f"n_proteins={g['n_proteins']}")


if __name__ == "__main__":
    main()
