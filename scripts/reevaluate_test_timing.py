"""
scripts/reevaluate_test_timing.py

Re-run test-set evaluation on one or more already-trained checkpoints to
capture per-protein inference time (added to evaluate_test in
src/training/trainer.py — existing checkpoints predate this and have no
inference_time_s field in their test_metrics.json).

Does not retrain anything. Rebuilds the exact architecture from the
checkpoint's own saved model_config/feature_spec (not from config.yaml,
which may have since changed), reloads the test split, and re-runs
evaluate_test.

By default, results are written to a SEPARATE output directory (--output-dir,
default model_eval/reeval_test_timing/<ckpt_name>/) rather than into the
source checkpoint directory. best_model.pt is only ever read, never
touched. This is deliberate: an earlier run of this script overwrote
checkpoints/phase_a/attention_pw05/test_metrics.json + test_predictions/ in
place, the fresh numbers didn't match the previously-recorded/cited values,
and the original files could not be recovered (checkpoints/ lives outside
the git repo, so there was no version history to fall back on). See
EVAL_REPRODUCIBILITY_INVESTIGATION.md for the full incident writeup.

Pass --in-place to restore the old overwrite-the-checkpoint-dir behavior —
only do this once the discrepancy in that doc is understood, and back up
test_metrics.json / test_predictions/ yourself first.

Usage:
    python scripts/reevaluate_test_timing.py checkpoints/phase_a/attention_pw05
    python scripts/reevaluate_test_timing.py checkpoints/phase_a/* checkpoints/phase_d/*
    python scripts/reevaluate_test_timing.py checkpoints/phase_a/attention_pw05 \\
        --output-dir /tmp/reeval_scratch
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.dataset import ProteinGraphDataset, load_split_manifest
from src.data.transform import NormalizeESP
from src.models.attention_espn import AttentionESPN
from src.models.distance_espn import DistanceESPN
from src.training.trainer import evaluate_test
from src.training.loss import ESPLoss
from src.utils.config import get_data_root


def set_seed(seed: int) -> None:
    """Mirror pipelines/07_train.py's seeding so this script's RNG state is
    reproducible and consistent with the training entry point, even though
    the current eval path (fixed weights, no dropout, no augmentation
    transform) doesn't consume randomness today — see investigation doc,
    this was flagged as unset and worth closing regardless."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def reevaluate(
    ckpt_dir: Path,
    device: torch.device,
    data_root: Path,
    write_dir: Path,
    in_place: bool,
) -> None:
    best_path = ckpt_dir / "best_model.pt"
    if not best_path.exists():
        print(f"  SKIP {ckpt_dir} — no best_model.pt")
        return

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model_name   = ckpt["model_name"]
    model_config = ckpt["model_config"]
    feature_spec = ckpt["feature_spec"]
    esp_mean     = ckpt["esp_mean"]
    esp_std      = ckpt["esp_std"]

    print(f"\n=== {ckpt_dir.name} ({model_name}) ===")
    if in_place:
        print(f"  WARNING: --in-place set — overwriting {ckpt_dir} directly")
    else:
        print(f"  Writing results to: {write_dir}  (source checkpoint dir untouched)")
    write_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(model_name, model_config, feature_spec, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_ids = load_split_manifest(data_root)
    test_ds = ProteinGraphDataset(test_ids, data_root, rebuild=False)
    test_ds.transform = NormalizeESP(esp_mean, esp_std)

    loss_fn = ESPLoss(pearson_weight=0.5)  # only used for the loss field, not selection
    extra_state = {"esp_mean": esp_mean, "esp_std": esp_std}

    t0 = time.perf_counter()
    results = evaluate_test(
        model, loss_fn, test_ds, device, extra_state,
        checkpoint_dir  = write_dir,
        predictions_dir = write_dir / "test_predictions",
        bf16            = True,
    )
    wall = time.perf_counter() - t0

    g = results["global"]
    print(
        f"  r={g['pearson_r']:.4f}  rmse={g['rmse']:.4f}  "
        f"mean_inference={g['mean_inference_time_s']*1000:.2f}ms/protein  "
        f"total_inference={g['total_inference_time_s']:.2f}s  "
        f"(wall incl. model load: {wall:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dirs", nargs="+", type=Path,
                        help="One or more checkpoint directories (each containing best_model.pt)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write test_metrics.json / test_predictions/ into "
                             "(one subdirectory per checkpoint, named after the checkpoint "
                             "dir). Default: <data_root>/../model_eval/reeval_test_timing/ "
                             "(matches the data_root.parent / 'model_eval' convention used "
                             "elsewhere, e.g. src/baseline_models/*/evaluate.py, "
                             "pipelines/run_sweep.py). Ignored if --in-place is set.")
    parser.add_argument("--in-place", action="store_true", default=False,
                        help="DANGEROUS: overwrite test_metrics.json + test_predictions/ "
                             "inside the source checkpoint dir, like this script used to "
                             "do unconditionally. Back up those files yourself first — see "
                             "EVAL_REPRODUCIBILITY_INVESTIGATION.md for why this "
                             "default was changed.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (random/numpy/torch), matching pipelines/07_train.py's "
                             "default. The current eval path is deterministic given fixed "
                             "weights, but this keeps behavior consistent and defensive "
                             "against future stochastic additions (e.g. transform.py's "
                             "RandomRotation).")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.yaml — required for checkpoints "
                             "trained on an isolated graph root (e.g. true_radial graphs), "
                             "since the checkpoint itself doesn't record which data_root it "
                             "was trained against.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or get_data_root()
    output_dir = args.output_dir or (get_data_root().parent / "model_eval" / "reeval_test_timing")
    print(f"Device: {device}  |  seed: {args.seed}  |  "
          f"{len(args.checkpoint_dirs)} checkpoint(s) to re-evaluate")

    for ckpt_dir in args.checkpoint_dirs:
        write_dir = ckpt_dir if args.in_place else output_dir / ckpt_dir.name
        reevaluate(ckpt_dir, device, data_root, write_dir, args.in_place)


if __name__ == "__main__":
    main()
