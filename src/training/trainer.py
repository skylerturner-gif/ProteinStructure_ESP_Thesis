"""
src/training/trainer.py

Training loop, validation, checkpointing, and metric tracking.

Usage
-----
    trainer = Trainer(
        model       = model,
        optimizer   = optimizer,
        scheduler   = scheduler,   # ReduceLROnPlateau
        loss_fn     = ESPLoss(pearson_weight=0.1),
        device      = torch.device("cuda"),
        checkpoint_dir = Path("checkpoints/run_01"),
    )
    trainer.fit(train_loader, val_loader, n_epochs=100)

Checkpoints
-----------
Two files are written to checkpoint_dir:
  best_model.pt   — saved whenever val_loss improves
  latest_model.pt — overwritten every epoch

Each checkpoint contains:
  epoch, model_state, optimizer_state, scheduler_state,
  val_loss, val_rmse, val_pearson_r,
  plus any extra fields passed via extra_state (e.g. esp_mean, esp_std).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.training.loss import ESPLoss, pearson_r
from src.utils.helpers import get_logger

log = get_logger(__name__)


class Trainer:
    """
    Manages the train/val loop, metric tracking, and checkpointing.

    Args:
        model:          the model to train (DistanceESPN or AttentionESPN)
        optimizer:      PyTorch optimizer (e.g. AdamW)
        scheduler:      LR scheduler — must accept scheduler.step(val_loss)
                        (i.e. ReduceLROnPlateau)
        loss_fn:        ESPLoss instance
        device:         torch.device for training
        checkpoint_dir: directory where checkpoints are saved
        clip_grad_norm: max gradient norm for clipping (default 1.0)
        extra_state:    additional fields saved in every checkpoint
                        (e.g. {'esp_mean': 0.12, 'esp_std': 3.4})
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn: ESPLoss,
        device: torch.device,
        checkpoint_dir: Path,
        clip_grad_norm: float = 1.0,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.loss_fn        = loss_fn
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.clip_grad_norm = clip_grad_norm
        self.extra_state    = extra_state or {}

        self.best_val_loss  = float("inf")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Single epoch loops ────────────────────────────────────────────────────

    def train_epoch(self, loader) -> dict[str, float]:
        """Run one training epoch. Returns {'loss': float}."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            pred   = self.model(data)
            target = data["query"].y
            batch  = data["query"].batch

            loss = self.loss_fn(pred, target, batch)
            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return {"loss": total_loss / max(n_batches, 1)}

    def val_epoch(self, loader) -> dict[str, float]:
        """
        Run one validation epoch.
        Returns {'loss': float, 'rmse': float, 'pearson_r': float}.
        """
        self.model.eval()
        total_loss   = 0.0
        total_sq_err = 0.0
        total_n      = 0
        pearson_vals: list[float] = []

        with torch.no_grad():
            for data in loader:
                data   = data.to(self.device)
                pred   = self.model(data)
                target = data["query"].y
                batch  = data["query"].batch

                loss = self.loss_fn(pred, target, batch)
                total_loss   += loss.item()
                total_sq_err += ((pred - target) ** 2).sum().item()
                total_n      += target.shape[0]

                n_graphs = int(batch.max().item()) + 1
                for g in range(n_graphs):
                    mask = batch == g
                    r = pearson_r(pred[mask], target[mask])
                    pearson_vals.append(r.item())

        n_batches   = max(len(pearson_vals) // max(1, int(batch.max().item()) + 1), 1)
        mean_loss   = total_loss / max(len(loader), 1)
        rmse        = (total_sq_err / max(total_n, 1)) ** 0.5
        mean_pearson = sum(pearson_vals) / max(len(pearson_vals), 1)

        return {"loss": mean_loss, "rmse": rmse, "pearson_r": mean_pearson}

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, n_epochs: int) -> None:
        """
        Train for n_epochs, printing a one-line summary each epoch and
        saving checkpoints whenever val_loss improves.
        """
        print(
            f"\n{'Epoch':>6}  {'Train loss':>11}  {'Val loss':>10}  "
            f"{'RMSE':>8}  {'Pearson r':>10}  {'LR':>10}"
        )
        print("-" * 65)

        for epoch in range(1, n_epochs + 1):
            t0          = time.perf_counter()
            train_m     = self.train_epoch(train_loader)
            val_m       = self.val_epoch(val_loader)
            elapsed     = time.perf_counter() - t0

            current_lr  = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_m["loss"])

            is_best = val_m["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m["loss"]

            self._save_checkpoint("latest_model.pt", epoch, val_m)
            if is_best:
                self._save_checkpoint("best_model.pt", epoch, val_m)

            marker = " *" if is_best else ""
            print(
                f"{epoch:>6}  {train_m['loss']:>11.4f}  {val_m['loss']:>10.4f}  "
                f"{val_m['rmse']:>8.4f}  {val_m['pearson_r']:>10.4f}  "
                f"{current_lr:>10.2e}{marker}"
            )
            log.info(
                "epoch=%d  train_loss=%.4f  val_loss=%.4f  "
                "rmse=%.4f  pearson_r=%.4f  lr=%.2e  t=%.1fs",
                epoch, train_m["loss"], val_m["loss"],
                val_m["rmse"], val_m["pearson_r"], current_lr, elapsed,
            )

        print(f"\nDone. Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self, filename: str, epoch: int, val_metrics: dict
    ) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "val_loss":        val_metrics["loss"],
                "val_rmse":        val_metrics["rmse"],
                "val_pearson_r":   val_metrics["pearson_r"],
                **self.extra_state,
            },
            path,
        )
        log.info("Saved checkpoint → %s", path)

    @staticmethod
    def load_checkpoint(
        path: Path,
        model,
        optimizer=None,
        scheduler=None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint and restore model (and optionally optimizer/scheduler).

        Returns the full checkpoint dict so the caller can access esp_mean,
        esp_std, epoch, and val metrics.
        """
        ckpt = torch.load(path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        log.info(
            "Loaded checkpoint from %s  (epoch=%d, val_loss=%.4f)",
            path, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")),
        )
        return ckpt
