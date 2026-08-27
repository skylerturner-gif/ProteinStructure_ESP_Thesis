"""
scripts/dump_checkpoint_config.py

Prints the actual model_config/feature_spec/training-provenance recorded
inside one or more checkpoints — the checkpoint is the only durable record
of what a run actually used (pipeline.log only logs a one-line summary;
full config is otherwise only ever printed to stdout). Also the tool that
caught the agg='sum' vs 'multi' mismatch across phase_e/f.

Usage:
    # One checkpoint, full detail
    python scripts/dump_checkpoint_config.py checkpoints/phase_a/attention_pw05

    # Many checkpoints (glob), one row per checkpoint
    python scripts/dump_checkpoint_config.py checkpoints/phase_*/*/

    # Flag anything that doesn't match an expected value
    python scripts/dump_checkpoint_config.py checkpoints/phase_*/*/ --check agg=multi
    python scripts/dump_checkpoint_config.py checkpoints/phase_*/*/ --check agg=multi --check use_element_embedding=True
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Fields pulled from model_config for the compact table view.
_TABLE_FIELDS = [
    "n_bond_radial_rounds", "n_aq_rounds", "n_qq_rounds",
    "agg", "use_residue_embedding", "use_bond_edges",
    "use_radial_edges", "use_element_embedding",
]


def _load(ckpt_dir: Path) -> dict | None:
    path = ckpt_dir / "best_model.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _parse_check(spec: str) -> tuple[str, str]:
    key, _, val = spec.partition("=")
    return key.strip(), val.strip()


def _coerce(val: str):
    if val in ("True", "true"):
        return True
    if val in ("False", "false"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dirs", nargs="+", type=Path,
                        help="One or more checkpoint directories (each containing best_model.pt)")
    parser.add_argument("--check", action="append", default=[],
                        help="key=value to flag mismatches for, e.g. --check agg=multi. "
                             "Repeatable. Checks model_config first, then feature_spec.")
    parser.add_argument("--full", action="store_true",
                        help="Print full detail per checkpoint instead of one compact table.")
    args = parser.parse_args()

    checks = [_parse_check(c) for c in args.check]

    rows = []
    missing = []
    for d in args.checkpoint_dirs:
        ckpt = _load(d)
        if ckpt is None:
            missing.append(d)
            continue
        rows.append((d, ckpt))

    if missing:
        print(f"SKIPPED (no best_model.pt): {[str(d) for d in missing]}")
        print()

    if args.full:
        for d, ckpt in rows:
            print(f"=== {d} ===")
            print(f"  model_name:   {ckpt.get('model_name')}")
            print(f"  epoch:        {ckpt.get('epoch')}")
            print(f"  val_loss:     {ckpt.get('val_loss')}")
            print(f"  val_pearson_r:{ckpt.get('val_pearson_r')}")
            print(f"  esp_mean/std: {ckpt.get('esp_mean')} / {ckpt.get('esp_std')}")
            print(f"  model_config: {ckpt.get('model_config')}")
            print(f"  feature_spec: {ckpt.get('feature_spec')}")
            print()
    else:
        name_w = max((len(str(d)) for d, _ in rows), default=10) + 2
        header = f"{'checkpoint':<{name_w}}" + "".join(f"{f:>14s}" for f in _TABLE_FIELDS)
        print(header)
        print("-" * len(header))
        for d, ckpt in rows:
            mc = ckpt.get("model_config", {})
            row = f"{str(d):<{name_w}}"
            for f in _TABLE_FIELDS:
                row += f"{str(mc.get(f, '-')):>14s}"
            print(row)

    if checks:
        print()
        print(f"--- check: {', '.join(f'{k}={v}' for k, v in checks)} ---")
        n_mismatch = 0
        for d, ckpt in rows:
            mc = ckpt.get("model_config", {})
            fs = ckpt.get("feature_spec", {})
            for key, expected_str in checks:
                expected = _coerce(expected_str)
                actual = mc.get(key, fs.get(key, "<missing>"))
                if actual != expected:
                    print(f"  MISMATCH  {d}: {key}={actual!r}  (expected {expected!r})")
                    n_mismatch += 1
        if n_mismatch == 0:
            print(f"  all {len(rows)} checkpoint(s) match.")
        else:
            print(f"  {n_mismatch} mismatch(es) found across {len(rows)} checkpoint(s).")


if __name__ == "__main__":
    main()
