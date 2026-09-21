"""
src/data/rbf.py

Gaussian RBF edge-distance encoding — the single source of truth for the basis,
shared by the graph builder (which stores raw distances) and the models (which
expand them at forward time).

Why the split
-------------
The encoding is a pure function of one scalar per edge but is ``n_rbf`` floats
wide, so caching it on disk costs ~16x the bytes for zero extra information.
Measured on a 600+ aa protein: ``edge_attr`` was 33.4 MB of a 42.4 MB graph
file (78.8%).  Training on the large-protein tier is disk-bound — the GPUs sat
at ~3% utilisation with ``/proc/pressure/io`` reporting ~80% full stall — so
those bytes were the dominant cost of an epoch.

Graphs therefore store ``edge_dist`` (float32, one per edge, plus a separate
``bond_order`` column for bond edges) and ``materialize_edge_attr`` rebuilds
``edge_attr`` on-device inside the model's forward.  This shrinks a graph file
~3.9x (42.4 MB -> ~11 MB) and cuts host RAM and DataLoader IPC by the same
factor.  It does NOT reduce VRAM: the expanded tensor is identical to what used
to be read from disk, it is just produced on the GPU instead of shipped there.

Legacy graphs that already carry a baked ``edge_attr`` are passed through
untouched, so old caches keep working without a rebuild.

Column order for bond edges is ``[bond_order | rbf(dist)]``, matching what
graph_builder wrote before the split — do not reorder, trained checkpoints
depend on it.

Public API
----------
  RBF_RANGES                                    # relation -> (d_min, d_max) Å
  rbf_expand(dist, n_rbf, d_min, d_max)         -> Tensor (E, n_rbf)
  materialize_edge_attr(data, n_rbf)            -> HeteroData  (in place)
  is_slim(data)                                 -> bool
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


# Per-relation RBF windows in Å.  These were chosen per edge type in the
# original graph_builder and are baked into every trained checkpoint — the
# model learns against this basis, so changing a window invalidates existing
# weights.  Keyed by relation name only (the 'true_radial' graph variant reuses
# the 'radial' relation and therefore its window).
RBF_RANGES: dict[str, tuple[float, float]] = {
    "bond":   (0.9, 1.8),
    "radial": (1.8, 8.0),
    "aq":     (0.0, 12.0),
    "qq":     (0.0, 8.0),
}


def rbf_expand(dist: Tensor, n_rbf: int, d_min: float, d_max: float) -> Tensor:
    """
    Expand a (E,) tensor of distances into (E, n_rbf) Gaussian RBF features.

    Numerically equivalent to graph_builder._rbf_encode (the numpy version kept
    for the non-slim build path): same centre spacing, same sigma, same
    unnormalised Gaussian.  Always returns float32 — the cached ``edge_attr``
    it replaces was float32, and downstream ``torch.cat`` with node features
    expects that dtype.

    Args:
        dist:   (E,) edge distances in Å.  Any float dtype; cast to float32.
        n_rbf:  number of basis functions.
        d_min:  first centre (Å).
        d_max:  last centre (Å).

    Returns:
        (E, n_rbf) float32 tensor on the same device as ``dist``.
    """
    centers = torch.linspace(d_min, d_max, n_rbf, device=dist.device, dtype=torch.float32)
    sigma   = (d_max - d_min) / max(n_rbf - 1, 1)
    d       = dist.to(torch.float32).unsqueeze(-1)
    return torch.exp(-((d - centers) ** 2) / (sigma ** 2))


def materialize_edge_attr(data: HeteroData, n_rbf: int) -> HeteroData:
    """
    Ensure every edge store on *data* has an ``edge_attr``, expanding from the
    stored ``edge_dist`` where it is missing.  Mutates and returns *data*.

    Idempotent and legacy-safe:
      - a store that already has ``edge_attr`` is left alone, so graphs built
        before the slim format (and any store already expanded earlier in this
        forward pass) cost nothing;
      - a store with neither ``edge_attr`` nor ``edge_dist`` raises, rather than
        silently message-passing on missing geometry.

    Called at the top of DistanceESPN.forward / AttentionESPN.forward, so every
    consumer — trainer, sweeps, eval scripts, notebooks, probes — gets it
    without needing to know the storage format.

    Args:
        data:   batched or single HeteroData.
        n_rbf:  basis width the model was built for.  Because expansion now
                happens at forward time, a slim graph can be read at any n_rbf
                without rebuilding the cache.

    Returns:
        The same HeteroData instance, with ``edge_attr`` on every edge store.
    """
    for edge_type in data.edge_types:
        store = data[edge_type]
        if store.get("edge_attr", None) is not None:
            continue

        relation = edge_type[1]
        dist     = store.get("edge_dist", None)
        if dist is None:
            raise RuntimeError(
                f"Edge store {edge_type} has neither 'edge_attr' nor 'edge_dist'. "
                "The graph cache is malformed — rebuild it with "
                "pipelines/06_build_graphs.py, or convert an older cache with "
                "scripts/slim_graph_edge_attr.py."
            )
        if relation not in RBF_RANGES:
            raise KeyError(
                f"No RBF window registered for relation {relation!r} "
                f"(known: {sorted(RBF_RANGES)}). Add it to RBF_RANGES in "
                "src/data/rbf.py."
            )

        d_min, d_max = RBF_RANGES[relation]
        attr = rbf_expand(dist, n_rbf, d_min, d_max)

        # Bond edges carry a leading bond_order column: [bond_order | rbf].
        bond_order = store.get("bond_order", None)
        if bond_order is not None:
            attr = torch.cat([bond_order.to(attr.dtype).unsqueeze(-1), attr], dim=-1)

        store.edge_attr = attr

    return data


def is_slim(data: HeteroData) -> bool:
    """
    True if *data* stores raw distances rather than baked RBF features.

    Reads the ``edge_attr_slim`` flag written by graph_builder when present,
    otherwise infers it from the stores (useful for graphs converted in place
    by scripts/slim_graph_edge_attr.py).
    """
    flag = getattr(data, "edge_attr_slim", None)
    if flag is not None:
        return bool(flag)
    return any(
        data[et].get("edge_attr", None) is None and data[et].get("edge_dist", None) is not None
        for et in data.edge_types
    )
