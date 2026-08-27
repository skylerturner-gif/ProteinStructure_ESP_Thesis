"""Surface DGCNN baseline model (static kNN, no torch_cluster at inference).

Architecture:
  1. Three stacked EdgeConv layers using a pre-computed static kNN graph on xyz coords.
     Static graph = kNN built once on 3D coordinates, cached to disk by the dataset.
     Input features: xyz + normals + nearest-atom element/residue one-hot (33D per point).
  2. Multi-scale concat → per-point projection.
  3. For each query node: gather k nearest pre-computed surface embeddings → mean → MLP.

The query→surface mapping is also precomputed and cached in the dataset, so the
forward pass contains no kNN computation at all — just indexed gathers and MLPs.

A geometry-only variant (no atom chemistry, testing whether surface shape
alone can predict ESP) was tried and dropped — it scored Pearson r=0.085 on
the test set, essentially no signal. Chemistry features (element+residue
identity, still no partial charges) are always on now.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class _EdgeConv(MessagePassing):
    """EdgeConv with max aggregation: h_i = max_{j∈N(i)} MLP([h_i, h_j - h_i])."""

    def __init__(self, nn_module: nn.Module) -> None:
        super().__init__(aggr="max")
        self.nn = nn_module

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))


class SurfaceDGCNN(nn.Module):
    """DGCNN on protein surface → ESP prediction at query nodes.

    Args:
        in_features: feature dimension per surface point — 33 = xyz+normals (6)
                     + element one-hot (6) + residue one-hot (21). The
                     geometry-only variant (6, no chemistry) was dropped after
                     scoring r=0.085 on the test set; see SurfaceDataset.
        hidden_dim:  channel width
        n_query_k:   surface points aggregated per query (must match dataset's n_qk)
    """

    def __init__(self, in_features: int = 33, hidden_dim: int = 128, n_query_k: int = 8) -> None:
        super().__init__()
        self.n_query_k = n_query_k

        # EdgeConv input: concat(x_i, x_j - x_i) = 2 * in_features
        self.conv1 = _EdgeConv(_mlp(in_features * 2, hidden_dim, hidden_dim))
        self.conv2 = _EdgeConv(_mlp(hidden_dim * 2, hidden_dim, hidden_dim))
        self.conv3 = _EdgeConv(_mlp(hidden_dim * 2, hidden_dim, hidden_dim))

        self.point_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.query_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        point_cloud: Tensor,   # (S, 6)       — xyz + normals
        edge_index: Tensor,    # (2, S*k)     — static kNN graph (pre-computed)
        query_surf: Tensor,    # (Q, n_query_k) — k nearest surface indices per query
    ) -> Tensor:
        """Return ESP predictions at Q query nodes, shape (Q,)."""
        x1 = self.conv1(point_cloud, edge_index)           # (S, H)
        x2 = self.conv2(x1, edge_index)                    # (S, H)
        x3 = self.conv3(x2, edge_index)                    # (S, H)
        per_point = self.point_proj(torch.cat([x1, x2, x3], dim=-1))  # (S, H)

        # Gather k nearest surface embeddings per query → mean pool
        # query_surf: (Q, k) → per_point[query_surf]: (Q, k, H)
        q_embed = per_point[query_surf].mean(dim=1)  # (Q, H)
        return self.query_head(q_embed).squeeze(-1)  # (Q,)
