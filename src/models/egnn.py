"""
src/models/egnn.py

Shared building blocks for ESP graph neural network models.

Imported by distance_espn.py and attention_espn.py — not intended to be
used directly in training scripts.

Exports
-------
  _mlp(dims)              — MLP factory
  AtomEncoder             — discrete atom attributes → hidden_dim vector
  QueryEncoder            — optional surface geometry features → hidden_dim
  MessageLayer            — one MP step with configurable aggregation
  _AtomMP                 — Stage 1: bond → radial passes on atom nodes
  _QueryRefine            — Stage 3: QQ passes on query nodes
  N_ELEMENT_TYPES         — vocabulary size for atom type embedding
  N_RESIDUE_TYPES         — vocabulary size for residue type embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from src.data.graph_builder import N_ELEMENT_TYPES, N_RESIDUE_TYPES

__all__ = [
    "_mlp",
    "AtomEncoder",
    "QueryEncoder",
    "MessageLayer",
    "_AtomMP",
    "_QueryRefine",
    "N_ELEMENT_TYPES",
    "N_RESIDUE_TYPES",
    "AGG_MODES",
]

# Valid MessageLayer aggregation modes.
#   mean, sum, max — single reduction
#   multi          — concatenate mean + sum + max
AGG_MODES = ("mean", "sum", "max", "multi")


# ── MLP factory ───────────────────────────────────────────────────────────────

def _mlp(dims: list[int], act: type = nn.SiLU) -> nn.Sequential:
    """Build a fully-connected MLP with SiLU activations between hidden layers."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


# ── Input encoding ────────────────────────────────────────────────────────────

class AtomEncoder(nn.Module):
    """
    Map discrete atom attributes to a continuous hidden vector.

    Inputs (from HeteroData['atom']):
      atom_type    — int64 element index  (0..N_ELEMENT_TYPES-1)   — always used
      residue_type — int64 residue index  (0..N_RESIDUE_TYPES-1)   — ablatable
      bond_count   — int64 number of covalent bonds                — ablatable

    Each active input is independently embedded, concatenated, then projected
    to hidden_dim via a two-layer MLP.

    Args:
        hidden_dim:            output dimensionality
        use_residue_embedding: include the residue-type embedding (default True)
        use_bond_count:        include the bond-count projection (default True)
    """

    def __init__(
        self,
        hidden_dim: int,
        use_residue_embedding: bool = True,
        use_bond_count:        bool = True,
    ) -> None:
        super().__init__()
        self.use_residue_embedding = use_residue_embedding
        self.use_bond_count        = use_bond_count

        emb = hidden_dim // 3
        n_active = 1 + int(use_residue_embedding) + int(use_bond_count)

        self.atom_emb  = nn.Embedding(N_ELEMENT_TYPES, emb)
        self.res_emb   = nn.Embedding(N_RESIDUE_TYPES, emb) if use_residue_embedding else None
        self.bond_proj = nn.Linear(1, emb)                  if use_bond_count        else None
        self.proj      = _mlp([emb * n_active, hidden_dim, hidden_dim])

    def forward(self, data: HeteroData) -> Tensor:
        parts = [self.atom_emb(data["atom"].atom_type)]
        if self.use_residue_embedding:
            parts.append(self.res_emb(data["atom"].residue_type))
        if self.use_bond_count:
            parts.append(self.bond_proj(data["atom"].bond_count.float().unsqueeze(-1)))
        return self.proj(torch.cat(parts, dim=-1))


# ── Surface geometry encoder ──────────────────────────────────────────────────

class QueryEncoder(nn.Module):
    """
    Projects optional surface geometry features into the query node embedding.

    Activated when query_curvature and/or query_normal are enabled in the
    features: block of config.yaml.  The projected output is added to the
    zero-initialised query embedding before Stage 2 message passing.

    Args:
        hidden_dim:    output dimensionality (must match model hidden_dim)
        has_curvature: True if data["query"].curvature is present
        has_normal:    True if data["query"].normal is present
    """

    def __init__(self, hidden_dim: int, has_curvature: bool, has_normal: bool) -> None:
        super().__init__()
        in_dim    = int(has_curvature) + (3 if has_normal else 0)
        self.proj = _mlp([in_dim, hidden_dim, hidden_dim])

    def forward(self, data: HeteroData) -> Tensor:
        parts: list[Tensor] = []
        if hasattr(data["query"], "curvature"):
            parts.append(data["query"].curvature.log1p().unsqueeze(-1))
        if hasattr(data["query"], "normal"):
            parts.append(data["query"].normal)
        return self.proj(torch.cat(parts, dim=-1))


# ── Message passing layer ─────────────────────────────────────────────────────

class MessageLayer(nn.Module):
    """
    One message-passing step for a single edge type.

    Handles both same-type edges (bond, radial, qq) and bipartite edges (aq).

    Message  : MLP(cat(h_src, h_dst, edge_attr))  →  hidden_dim
    Aggregate: configurable — "mean", "sum", "max", or "multi" (mean+sum+max
               concatenated)
    Update   : h_dst = LayerNorm(h_dst + MLP(cat(h_dst, aggregations)))

    Args:
        hidden_dim:    node feature dimensionality
        edge_feat_dim: number of edge features
        agg:           one of AGG_MODES ("mean", "sum", "max", "multi").
                      "multi" concatenates all three reductions (update MLP
                      input = hidden_dim × 4 instead of × 2).
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, agg: str = "mean") -> None:
        super().__init__()
        if agg not in AGG_MODES:
            raise ValueError(f"agg={agg!r} must be one of {AGG_MODES}")
        self.agg        = agg
        update_in       = hidden_dim * 4 if agg == "multi" else hidden_dim * 2
        self.msg_mlp    = _mlp([hidden_dim * 2 + edge_feat_dim, hidden_dim, hidden_dim])
        self.update_mlp = _mlp([update_in, hidden_dim, hidden_dim])
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_src: Tensor,
        h_dst: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        n_dst: int,
    ) -> Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        msgs = self.msg_mlp(
            torch.cat([h_src[src_idx], h_dst[dst_idx], edge_attr], dim=-1)
        )
        if self.agg == "multi":
            agg_mean = scatter(msgs, dst_idx, dim=0, dim_size=n_dst, reduce="mean")
            agg_sum  = scatter(msgs, dst_idx, dim=0, dim_size=n_dst, reduce="sum")
            agg_max  = scatter(msgs, dst_idx, dim=0, dim_size=n_dst, reduce="max")
            combined = torch.cat([h_dst, agg_mean, agg_sum, agg_max], dim=-1)
        else:
            agg      = scatter(msgs, dst_idx, dim=0, dim_size=n_dst, reduce=self.agg)
            combined = torch.cat([h_dst, agg], dim=-1)
        delta = self.update_mlp(combined)
        return self.norm(h_dst + delta)


# ── Stage modules ─────────────────────────────────────────────────────────────

class _AtomMP(nn.Module):
    """
    Stage 1: interleaved bond → radial passes, repeated n_rounds times.
    Weights are shared across rounds (one layer instance per edge type).

    Args:
        use_bond_edges:   run the bond-edge message pass each round (default True)
        use_radial_edges: run the radial-edge message pass each round (default True)
                          If both are False, Stage 1 is the identity — h_atom
                          passes through unchanged from AtomEncoder.
    """

    def __init__(
        self,
        hidden_dim: int, n_rbf: int, n_rounds: int, agg: str = "mean",
        use_bond_edges:   bool = True,
        use_radial_edges: bool = True,
    ) -> None:
        super().__init__()
        self.n_rounds     = n_rounds
        self.bond_layer   = MessageLayer(hidden_dim, n_rbf + 1, agg=agg) if use_bond_edges   else None
        self.radial_layer = MessageLayer(hidden_dim, n_rbf,     agg=agg) if use_radial_edges else None

    def forward(self, h_atom: Tensor, data: HeteroData) -> Tensor:
        n = h_atom.shape[0]
        for _ in range(self.n_rounds):
            if self.bond_layer is not None:
                bond   = data["atom", "bond",   "atom"]
                h_atom = self.bond_layer(h_atom, h_atom, bond.edge_index, bond.edge_attr, n)
            if self.radial_layer is not None:
                radial = data["atom", "radial", "atom"]
                h_atom = self.radial_layer(h_atom, h_atom, radial.edge_index, radial.edge_attr, n)
        return h_atom


class _QueryRefine(nn.Module):
    """
    Stage 3: QQ passes repeated n_rounds times.
    Weights are shared across rounds (one layer instance).
    """

    def __init__(self, hidden_dim: int, n_rbf: int, n_rounds: int, agg: str = "mean") -> None:
        super().__init__()
        self.n_rounds = n_rounds
        self.qq_layer = MessageLayer(hidden_dim, n_rbf, agg=agg)

    def forward(self, h_query: Tensor, data: HeteroData) -> Tensor:
        qq = data["query", "qq", "query"]
        n  = h_query.shape[0]
        for _ in range(self.n_rounds):
            h_query = self.qq_layer(h_query, h_query, qq.edge_index, qq.edge_attr, n)
        return h_query
