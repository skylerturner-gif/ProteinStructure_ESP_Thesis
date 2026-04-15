"""
src/models/attention_espn.py

Attention-based ESP prediction model.

Identical to DistanceESPN for Stages 1 and 3. Stage 2 replaces mean
aggregation with multi-head cross-attention: each query node attends over
its knn_aq atom neighbors, with a per-head RBF geometry bias.

  Stage 1 — Atom encoder    : bond → radial,  2 rounds
  Stage 2 — Query encoder   : AQ,           3 rounds  (cross-attention)
  Stage 3 — Query refinement: QQ,           2 rounds

All stages trained jointly end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as pyg_softmax

from src.models.egnn import (
    AtomEncoder,
    QueryEncoder,
    _AtomMP,
    _QueryRefine,
    _mlp,
)

__all__ = ["AttentionESPN"]


class AQAttentionLayer(nn.Module):
    """
    Multi-head cross-attention from atom nodes to query nodes along AQ edges.

    Each query attends over its knn_aq atom neighbors. Attention scores are
    scaled dot-product of projected query and atom features plus a per-head
    geometry bias from the RBF distance features.

    Score_h(q,a) = (Q_h(h_q) · K_h(h_a)) / √d_h  +  rbf_bias_h(edge_attr)
    α_h          = softmax over source atoms per query, per head
    msg          = Σ_a α_h(q,a) · V_h(h_a)   concatenated across heads
    h_query      = LayerNorm(h_query + MLP(cat(h_query, msg)))
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
            )
        self.n_heads = n_heads
        self.d_head  = hidden_dim // n_heads

        self.Q        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rbf_bias = nn.Linear(edge_feat_dim, n_heads, bias=False)
        self.update_mlp = _mlp([hidden_dim * 2, hidden_dim, hidden_dim])
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_atom: Tensor,
        h_query: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        n_query: int,
    ) -> Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        E, H, d = src_idx.shape[0], self.n_heads, self.d_head

        Q = self.Q(h_query[dst_idx]).view(E, H, d)
        K = self.K(h_atom[src_idx]).view(E, H, d)
        V = self.V(h_atom[src_idx]).view(E, H, d)

        scores = (Q * K).sum(-1) / (d ** 0.5) + self.rbf_bias(edge_attr)

        alpha = torch.zeros_like(scores)
        for h in range(H):
            alpha[:, h] = pyg_softmax(scores[:, h], dst_idx, num_nodes=n_query)

        msgs  = (alpha.unsqueeze(-1) * V).reshape(E, H * d)
        agg   = scatter(msgs, dst_idx, dim=0, dim_size=n_query, reduce="sum")
        delta = self.update_mlp(torch.cat([h_query, agg], dim=-1))
        return self.norm(h_query + delta)


class AttentionESPN(nn.Module):
    """
    ESP prediction model with multi-head cross-attention on AQ edges.

    Args:
        hidden_dim:           node feature dimensionality (default 128)
        n_rbf:                RBF basis functions per edge (default 16)
        n_heads:              attention heads in AQ layers (default 4)
        n_bond_radial_rounds: Stage 1 rounds (default 2)
        n_aq_rounds:          Stage 2 rounds (default 3)
        n_qq_rounds:          Stage 3 rounds (default 2)
        multi_agg:            use mean+sum+max aggregation in bond/radial/qq
                              MessageLayers (default False)
        has_curvature:        query nodes carry a curvature scalar (default False)
        has_normal:           query nodes carry a surface normal 3-vector (default False)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_rbf:      int = 16,
        n_heads:    int = 4,
        n_bond_radial_rounds: int = 2,
        n_aq_rounds:          int = 3,
        n_qq_rounds:          int = 2,
        multi_agg:            bool = False,
        has_curvature:        bool = False,
        has_normal:           bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.n_aq_rounds = n_aq_rounds

        self.atom_encoder  = AtomEncoder(hidden_dim)
        self.atom_mp       = _AtomMP(hidden_dim, n_rbf, n_bond_radial_rounds, multi_agg=multi_agg)
        self.aq_layer      = AQAttentionLayer(hidden_dim, n_rbf, n_heads)
        self.query_refine  = _QueryRefine(hidden_dim, n_rbf, n_qq_rounds, multi_agg=multi_agg)
        self.output_head   = _mlp([hidden_dim, hidden_dim // 2, 1])
        self.query_encoder = (
            QueryEncoder(hidden_dim, has_curvature, has_normal)
            if (has_curvature or has_normal) else None
        )

    def forward(self, data: HeteroData) -> Tensor:
        # Stage 1 — atom encoder
        h_atom = self.atom_encoder(data)
        h_atom = self.atom_mp(h_atom, data)

        # Stage 2 — atom→query (cross-attention, shared weights)
        n_query = data["query"].pos.shape[0]
        h_query = torch.zeros(n_query, self.hidden_dim, device=h_atom.device)
        if self.query_encoder is not None:
            h_query = h_query + self.query_encoder(data)
        aq = data["atom", "aq", "query"]
        for _ in range(self.n_aq_rounds):
            h_query = self.aq_layer(h_atom, h_query, aq.edge_index, aq.edge_attr, n_query)

        # Stage 3 — query refinement
        h_query = self.query_refine(h_query, data)

        return self.output_head(h_query).squeeze(-1)
