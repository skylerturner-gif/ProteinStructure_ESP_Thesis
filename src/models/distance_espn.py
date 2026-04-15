"""
src/models/distance_espn.py

Fully distance-based ESP prediction model.

All edge types use mean-aggregation message passing. Geometry is encoded
purely via Gaussian RBF features on each edge — no attention, no coordinate
updates.

  Stage 1 — Atom encoder    : bond → radial,  2 rounds
  Stage 2 — Query encoder   : AQ,           3 rounds  (mean aggregation)
  Stage 3 — Query refinement: QQ,           2 rounds

All stages trained jointly end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from src.models.egnn import (
    AtomEncoder,
    QueryEncoder,
    MessageLayer,
    _AtomMP,
    _QueryRefine,
    _mlp,
)

__all__ = ["DistanceESPN"]


class DistanceESPN(nn.Module):
    """
    Fully distance-based ESP prediction model.

    Args:
        hidden_dim:           node feature dimensionality (default 128)
        n_rbf:                RBF basis functions per edge (default 16)
        n_bond_radial_rounds: Stage 1 rounds (default 2)
        n_aq_rounds:          Stage 2 rounds (default 3)
        n_qq_rounds:          Stage 3 rounds (default 2)
        multi_agg:            use mean+sum+max aggregation in all MessageLayers
                              (default False)
        has_curvature:        query nodes carry a curvature scalar (default False)
        has_normal:           query nodes carry a surface normal 3-vector (default False)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_rbf:      int = 16,
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
        self.aq_layer      = MessageLayer(hidden_dim, n_rbf, multi_agg=multi_agg)
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

        # Stage 2 — atom→query (mean aggregation, shared weights)
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
