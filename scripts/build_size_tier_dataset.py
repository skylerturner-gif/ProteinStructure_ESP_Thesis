"""
scripts/build_size_tier_dataset.py

Size-generalization pilot ("Phase 3b" in the project roadmap): train only on
large proteins, evaluate on medium and small proteins held entirely out of
training. Cutoffs (user-chosen, exploratory -- getting the size definitions
right matters more than squeezing out performance): small <300 aa,
medium 300-599 aa, large >=600 aa.

No new graphs are built -- every protein in the 8,461-protein master dataset
already has a cached graph.pt at the standard 5% query density. This script
only partitions existing protein IDs by size and (for the large tier, which
needs its own train/val/test split for pipelines/07_train.py) creates a
lightweight symlink-farm data_root pointing back at the real per-protein
directories, so nothing is copied or rebuilt.

Output:
  - /home/student/thesis/full_protein_dataset_size_large600/<protein_id> ->
    symlink to full_protein_dataset/<protein_id>, one per large-tier protein,
    plus a split_manifest.json (80/10/10 stratified split, seed=42 -- same
    convention as every other split in this project) written by the existing
    write_split_manifest() so pipelines/07_train.py works against this
    data_root exactly like any other.
  - outputs/size_tier_large_ids.txt   (>=600 aa, the training population)
  - outputs/size_tier_medium_ids.txt  (300-599 aa, eval only)
  - outputs/size_tier_small_ids.txt   (<300 aa, eval only)
    Medium/small need no symlink farm -- eval reads graphs directly from
    full_protein_dataset/ via --data-root + --id-file, same pattern as
    scripts/eval_mesh_density.py.

Usage:
    conda activate pyg_env
    python scripts/build_size_tier_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProteinGraphDataset, write_split_manifest

MASTER_CSV = Path("/home/student/thesis/outputs/protein_master_dataset.csv")
MAIN_DATA_ROOT = Path("/home/student/thesis/full_protein_dataset")
LARGE_DATA_ROOT = Path("/home/student/thesis/full_protein_dataset_size_large600")

SMALL_MAX = 300   # <300 aa = small
LARGE_MIN = 600   # >=600 aa = large; [300, 600) = medium

OUT_DIR = Path("/home/student/thesis/outputs")


def main() -> None:
    df = pd.read_csv(MASTER_CSV)
    print(f"Master dataset: {len(df)} proteins")

    small_ids  = df.loc[df["sequence_length"] < SMALL_MAX, "protein_id"].tolist()
    medium_ids = df.loc[(df["sequence_length"] >= SMALL_MAX) & (df["sequence_length"] < LARGE_MIN),
                         "protein_id"].tolist()
    large_ids  = df.loc[df["sequence_length"] >= LARGE_MIN, "protein_id"].tolist()
    print(f"small (<{SMALL_MAX} aa):  {len(small_ids)}")
    print(f"medium ({SMALL_MAX}-{LARGE_MIN-1} aa): {len(medium_ids)}")
    print(f"large (>={LARGE_MIN} aa):  {len(large_ids)}")
    assert len(small_ids) + len(medium_ids) + len(large_ids) == len(df)

    # ── Medium/small: plain ID lists, no symlink farm needed (eval reads
    # straight from the real data_root) ─────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "size_tier_small_ids.txt").write_text("\n".join(small_ids) + "\n")
    (OUT_DIR / "size_tier_medium_ids.txt").write_text("\n".join(medium_ids) + "\n")
    (OUT_DIR / "size_tier_large_ids.txt").write_text("\n".join(large_ids) + "\n")
    print(f"Wrote size_tier_{{small,medium,large}}_ids.txt -> {OUT_DIR}")

    # ── Large: symlink farm + its own split manifest for training ──────────
    if LARGE_DATA_ROOT.exists():
        existing = {p.name for p in LARGE_DATA_ROOT.iterdir() if p.is_symlink()}
        missing  = set(large_ids) - existing
        stale    = existing - set(large_ids)
        if not missing and not stale:
            print(f"{LARGE_DATA_ROOT} already has all {len(large_ids)} large-tier symlinks, none stale -- skipping farm build")
        else:
            print(f"{LARGE_DATA_ROOT} exists but is out of date: {len(missing)} missing, {len(stale)} stale -- fixing")
            for pid in stale:
                (LARGE_DATA_ROOT / pid).unlink()
            for pid in missing:
                (LARGE_DATA_ROOT / pid).symlink_to(MAIN_DATA_ROOT / pid)
    else:
        LARGE_DATA_ROOT.mkdir(parents=True)
        for pid in large_ids:
            (LARGE_DATA_ROOT / pid).symlink_to(MAIN_DATA_ROOT / pid)
        print(f"Created {LARGE_DATA_ROOT} with {len(large_ids)} symlinks -> {MAIN_DATA_ROOT}")

    manifest_path = LARGE_DATA_ROOT / "split_manifest.json"
    if manifest_path.exists():
        print(f"{manifest_path} already exists -- not overwriting (write_split_manifest is a no-op here anyway)")
    large_ds = ProteinGraphDataset(large_ids, LARGE_DATA_ROOT)
    manifest_path = write_split_manifest(large_ds, train=0.8, val=0.1, seed=42)
    print(f"Split manifest written/confirmed -> {manifest_path}")


if __name__ == "__main__":
    main()
