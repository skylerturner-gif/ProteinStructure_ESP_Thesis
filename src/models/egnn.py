"""
src/models/egnn.py

Shared building blocks for ESP graph neural network models.

Imported by distance_espn.py and attention_espn.py — not intended to be
used directly in training scripts.

Exports
-------
  _mlp(dims)              — MLP factory
  AtomEncoder             — discrete atom attributes → hidden_dim vector
  MessageLayer            — one MP step (same-type or bipartite edges)
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
    "MessageLayer",
    "_AtomMP",
    "_QueryRefine",
    "N_ELEMENT_TYPES",
    "N_RESIDUE_TYPES",
]


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
      atom_type    — int64 element index  (0..N_ELEMENT_TYPES-1)
      residue_type — int64 residue index  (0..N_RESIDUE_TYPES-1)
      bond_count   — int64 number of covalent bonds

    Each input is independently embedded, concatenated, then projected to
    hidden_dim via a two-layer MLP.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        emb = hidden_dim // 3
        self.atom_emb  = nn.Embedding(N_ELEMENT_TYPES, emb)
        self.res_emb   = nn.Embedding(N_RESIDUE_TYPES, emb)
        self.bond_proj = nn.Linear(1, emb)
        self.proj      = _mlp([emb * 3, hidden_dim, hidden_dim])

    def forward(self, data: HeteroData) -> Tensor:
        e = self.atom_emb(data["atom"].atom_type)
        r = self.res_emb(data["atom"].residue_type)
        b = self.bond_proj(data["atom"].bond_count.float().unsqueeze(-1))
        return self.proj(torch.cat([e, r, b], dim=-1))


# ── Message passing layer ─────────────────────────────────────────────────────

class MessageLayer(nn.Module):
    """
    One message-passing step for a single edge type.

    Handles both same-type edges (bond, radial, qq) and bipartite edges (aq).

    Message  : MLP(cat(h_src, h_dst, edge_attr))  →  hidden_dim
    Aggregate: mean over source neighbors per destination node
    Update   : h_dst = LayerNorm(h_dst + MLP(cat(h_dst, agg)))
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int) -> None:
        super().__init__()
        self.msg_mlp    = _mlp([hidden_dim * 2 + edge_feat_dim, hidden_dim, hidden_dim])
        self.update_mlp = _mlp([hidden_dim * 2, hidden_dim, hidden_dim])
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
        msgs  = self.msg_mlp(
            torch.cat([h_src[src_idx], h_dst[dst_idx], edge_attr], dim=-1)
        )
        agg   = scatter(msgs, dst_idx, dim=0, dim_size=n_dst, reduce="mean")
        delta = self.update_mlp(torch.cat([h_dst, agg], dim=-1))
        return self.norm(h_dst + delta)


# ── Stage modules ─────────────────────────────────────────────────────────────

class _AtomMP(nn.Module):
    """Stage 1: interleaved bond → radial passes, repeated n_rounds times."""

    def __init__(self, hidden_dim: int, n_rbf: int, n_rounds: int) -> None:
        super().__init__()
        self.cov_layers  = nn.ModuleList(
            [MessageLayer(hidden_dim, n_rbf + 1) for _ in range(n_rounds)]
        )
        self.supp_layers = nn.ModuleList(
            [MessageLayer(hidden_dim, n_rbf) for _ in range(n_rounds)]
        )

    def forward(self, h_atom: Tensor, data: HeteroData) -> Tensor:
        bond  = data["atom", "bond",  "atom"]
        radial = data["atom", "radial", "atom"]
        n    = h_atom.shape[0]
        for cov_l, supp_l in zip(self.cov_layers, self.supp_layers):
            h_atom = cov_l( h_atom, h_atom, bond.edge_index,  bond.edge_attr,  n)
            h_atom = supp_l(h_atom, h_atom, radial.edge_index, radial.edge_attr, n)
        return h_atom


class _QueryRefine(nn.Module):
    """Stage 3: QQ passes repeated n_rounds times."""

    def __init__(self, hidden_dim: int, n_rbf: int, n_rounds: int) -> None:
        super().__init__()
        self.qq_layers = nn.ModuleList(
            [MessageLayer(hidden_dim, n_rbf) for _ in range(n_rounds)]
        )

    def forward(self, h_query: Tensor, data: HeteroData) -> Tensor:
        qq = data["query", "qq", "query"]
        n  = h_query.shape[0]
        for qq_l in self.qq_layers:
            h_query = qq_l(h_query, h_query, qq.edge_index, qq.edge_attr, n)
        return h_query
