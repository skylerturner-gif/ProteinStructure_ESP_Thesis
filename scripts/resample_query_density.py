"""
scripts/resample_query_density.py

Regenerate query_idx at a different sample_frac for an explicit list of
protein IDs, reusing each protein's already-computed esp_verts (no APBS/DX
re-interpolation, no MSMS re-run) via curvature_sampling. Writes to an
isolated data root — structure/electrostatics/mesh/metadata are symlinked
back to the source root (unchanged at any density); only esp/ is freshly
written with the new query_idx. The source root's own esp/graph files are
never touched.

Used for the mesh-density evaluation: does asking an already-trained model
to predict directly at 10%/25% query density (instead of today's 5%) beat
sparse-predict-then-RBF-interpolate? See scripts/eval_mesh_density.py for
the inference step that follows this.

Usage:
    conda activate pyg_env
    python scripts/resample_query_density.py \\
        --id-file /tmp/test_ids.txt \\
        --source-root /home/student/thesis/full_protein_dataset \\
        --dest-root /home/student/thesis/full_protein_dataset_density10 \\
        --sample-frac 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surface.esp_mapping import curvature_sampling
from src.utils.helpers import get_pipeline_logger, notify, timer
from src.utils.config import get_config
from src.utils.paths import ProteinPaths


def _resample_one(
    protein_id: str, source_root: Path, dest_root: Path, sample_frac: float, log
) -> str:
    src = ProteinPaths(protein_id, source_root)
    dst = ProteinPaths(protein_id, dest_root)

    if not src.esp_path.exists() or not src.mesh_path.exists():
        log.error("[%s] Missing esp/mesh in source root", protein_id)
        return "fail"

    dst.protein_dir.mkdir(parents=True, exist_ok=True)

    # Symlink everything that doesn't change with query density.
    for src_dir, dst_dir in [
        (src.structure_dir,      dst.structure_dir),
        (src.electrostatics_dir, dst.electrostatics_dir),
        (src.mesh_dir,           dst.mesh_dir),
    ]:
        if not dst_dir.exists():
            dst_dir.symlink_to(src_dir)
    if not dst.metadata_path.exists() and src.metadata_path.exists():
        dst.metadata_path.symlink_to(src.metadata_path)

    dst.esp_dir.mkdir(parents=True, exist_ok=True)
    dst.logs_dir.mkdir(parents=True, exist_ok=True)
    dst.graph_dir.mkdir(parents=True, exist_ok=True)

    try:
        with timer() as t:
            esp_data  = np.load(src.esp_path)
            mesh_data = np.load(src.mesh_path)
            verts, faces = esp_data["verts"], esp_data["faces"]
            ses_area = float(mesh_data["ses_area"])

            n_query   = max(1, int(len(verts) * sample_frac))
            query_idx = curvature_sampling(verts, faces, n_query, ses_area)

            np.savez_compressed(
                dst.esp_path,
                verts=verts, faces=faces,
                esp_verts=esp_data["esp_verts"], esp_faces=esp_data["esp_faces"],
                query_idx=query_idx,
            )
        log.info(
            "[%s] Resampled query_idx: %d verts -> %d query (frac=%.2f)  (%.2f s)",
            protein_id, len(verts), len(query_idx), sample_frac, t.seconds,
        )
        return "ok"
    except Exception as e:
        log.error("[%s] Resample failed: %s", protein_id, e)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate query_idx at a new sample_frac for an explicit protein-ID list, "
                     "reusing already-computed esp_verts (no APBS/DX work)."
    )
    parser.add_argument("--id-file", type=Path, required=True,
                         help="Text file with one protein ID per line.")
    parser.add_argument("--source-root", type=Path, required=True,
                         help="Existing data root with esp/mesh already computed.")
    parser.add_argument("--dest-root", type=Path, required=True,
                         help="Isolated output root (created if missing).")
    parser.add_argument("--sample-frac", type=float, required=True,
                         help="New fraction of mesh vertices to select as query nodes "
                              "(e.g. 0.10, 0.25).")
    args = parser.parse_args()

    log = get_pipeline_logger(Path(get_config()["paths"]["log_file"]))

    protein_ids = [
        line.strip() for line in args.id_file.read_text().splitlines() if line.strip()
    ]
    print(f"Resampling {len(protein_ids):,} proteins at sample_frac={args.sample_frac} "
          f"-> {args.dest_root}")

    n_ok = n_fail = 0
    for protein_id in protein_ids:
        status = _resample_one(protein_id, args.source_root, args.dest_root, args.sample_frac, log)
        if status == "ok":
            n_ok += 1
            notify(protein_id, "complete", "query density resample")
        else:
            n_fail += 1
            notify(protein_id, "failed", "query density resample")

    print(f"Done — ok: {n_ok}  failed: {n_fail}")
    log.info("Query density resample (frac=%.2f) complete — ok: %d  failed: %d",
              args.sample_frac, n_ok, n_fail)


if __name__ == "__main__":
    main()
