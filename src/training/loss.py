"""
src/training/loss.py

Loss functions for ESP surface prediction.

ESPLoss combines two terms:
  1. MSE  — penalises absolute prediction error in kT/e
  2. Pearson — penalises mis-ranking of the spatial ESP pattern (1 − r)

The Pearson term is computed per-protein (not globally across the batch) so
that the model is rewarded for correctly predicting the pattern on each
individual surface, independent of between-protein scale differences.

Usage
-----
    loss_fn = ESPLoss(pearson_weight=0.1)
    loss = loss_fn(pred, data['query'].y, data['query'].batch)

Standalone helpers
------------------
    pearson_r(pred, target)  — Pearson r for a 1-D pair of tensors
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pearson_r(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Pearson correlation coefficient between two 1-D tensors.

    Args:
        pred:   predicted values  (N,)
        target: ground-truth ESP  (N,)
        eps:    small constant to avoid division by zero

    Returns:
        Scalar tensor in [-1, 1].
    """
    pred_c   = pred   - pred.mean()
    target_c = target - target.mean()
    num      = (pred_c * target_c).sum()
    denom    = pred_c.norm() * target_c.norm()
    return num / (denom + eps)


class ESPLoss(nn.Module):
    """
    Combined MSE + per-graph Pearson loss for ESP surface prediction.

    Loss = MSE(pred, target)  +  pearson_weight × mean_g(1 − r_g)

    where r_g is the Pearson correlation for graph g in the batch.
    MSE drives accurate absolute values; the Pearson term rewards correct
    spatial patterning regardless of global scale.

    Args:
        pearson_weight: weight for the Pearson term (default 0.1).
                        Set to 0.0 to use pure MSE.
    """

    def __init__(self, pearson_weight: float = 0.1) -> None:
        super().__init__()
        self.pearson_weight = pearson_weight

    def forward(self, pred: Tensor, target: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            pred:   predicted ESP values  (N_query_total,)
            target: ground-truth ESP       (N_query_total,)
            batch:  graph index per query  (N_query_total,)  — from data['query'].batch

        Returns:
            Scalar loss tensor.
        """
        mse = F.mse_loss(pred, target)

        if self.pearson_weight == 0.0:
            return mse

        n_graphs = int(batch.max().item()) + 1
        pearson_losses = torch.zeros(n_graphs, device=pred.device)
        for g in range(n_graphs):
            mask = batch == g
            r = pearson_r(pred[mask], target[mask])
            pearson_losses[g] = 1.0 - r

        return mse + self.pearson_weight * pearson_losses.mean()

    def __repr__(self) -> str:
        return f"ESPLoss(pearson_weight={self.pearson_weight})"
