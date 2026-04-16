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

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGLoader

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
        clip_grad_norm:        float = 1.0,
        grad_accum_steps:      int   = 1,
        early_stopping_patience: int = 0,
        extra_state: dict[str, Any] | None = None,
        rank: int = 0,
    ) -> None:
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.loss_fn        = loss_fn
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.clip_grad_norm          = clip_grad_norm
        self.grad_accum_steps        = max(1, grad_accum_steps)
        self.early_stopping_patience = max(0, early_stopping_patience)
        self.extra_state             = extra_state or {}
        self.rank                    = rank
        self.is_main                 = (rank == 0)

        self.best_val_loss  = float("inf")
        self._epochs_no_improve = 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list] = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "val_rmse": [], "val_pearson_r": [], "lr": [],
        }

    # ── Single epoch loops ────────────────────────────────────────────────────

    def train_epoch(self, loader) -> dict[str, float]:
        """Run one training epoch. Returns {'loss': float}."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        self.optimizer.zero_grad()
        for step_idx, data in enumerate(loader):
            data   = data.to(self.device)
            pred   = self.model(data)
            target = data["query"].y
            batch  = data["query"].batch

            # Scale loss so accumulated gradients match a single-step update.
            loss = self.loss_fn(pred, target, batch) / self.grad_accum_steps
            loss.backward()
            total_loss += loss.item() * self.grad_accum_steps   # track unscaled
            n_batches  += 1

            at_boundary = (step_idx + 1) % self.grad_accum_steps == 0
            is_last     = (step_idx + 1) == len(loader)
            if at_boundary or is_last:
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

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

        # All-reduce across ranks so every rank uses the same global metrics.
        # This keeps ReduceLROnPlateau and is_best in sync across all ranks.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed as dist
            t = torch.tensor(
                [
                    total_loss,
                    total_sq_err,
                    float(total_n),
                    float(sum(pearson_vals)),
                    float(len(pearson_vals)),
                    float(len(loader)),
                ],
                device=self.device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mean_loss    = t[0].item() / max(t[5].item(), 1)
            rmse         = (t[1].item() / max(t[2].item(), 1)) ** 0.5
            mean_pearson = t[3].item() / max(t[4].item(), 1)
        else:
            mean_loss    = total_loss   / max(len(loader), 1)
            rmse         = (total_sq_err / max(total_n, 1)) ** 0.5
            mean_pearson = sum(pearson_vals) / max(len(pearson_vals), 1)

        return {"loss": mean_loss, "rmse": rmse, "pearson_r": mean_pearson}

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, n_epochs: int) -> None:
        """
        Train for n_epochs, printing a one-line summary each epoch and
        saving checkpoints whenever val_loss improves.
        """
        if self.is_main:
            print(
                f"\n{'Epoch':>6}  {'Train loss':>11}  {'Val loss':>10}  "
                f"{'RMSE':>8}  {'Pearson r':>10}  {'LR':>10}"
            )
            print("-" * 65)

        for epoch in range(1, n_epochs + 1):
            # Let samplers (DynamicBatchSampler, DistributedSampler, etc.)
            # update their internal epoch counter for a fresh shuffle.
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            t0          = time.perf_counter()
            train_m     = self.train_epoch(train_loader)
            val_m       = self.val_epoch(val_loader)
            elapsed     = time.perf_counter() - t0

            current_lr  = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_m["loss"])
            else:
                self.scheduler.step()

            is_best = val_m["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m["loss"]
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1

            if self.is_main:
                self._save_checkpoint("latest_model.pt", epoch, val_m)
                if is_best:
                    self._save_checkpoint("best_model.pt", epoch, val_m)

                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_m["loss"])
                self.history["val_loss"].append(val_m["loss"])
                self.history["val_rmse"].append(val_m["rmse"])
                self.history["val_pearson_r"].append(val_m["pearson_r"])
                self.history["lr"].append(current_lr)
                self._save_metrics_csv()

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

            if (
                self.early_stopping_patience > 0
                and self._epochs_no_improve >= self.early_stopping_patience
            ):
                if self.is_main:
                    print(
                        f"\n[early stop] No improvement for "
                        f"{self.early_stopping_patience} epochs — stopping at epoch {epoch}."
                    )
                break

        if self.is_main:
            print(f"\nDone. Best val loss: {self.best_val_loss:.4f}")
            print(f"Checkpoints saved to: {self.checkpoint_dir}")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self, filename: str, epoch: int, val_metrics: dict
    ) -> None:
        path = self.checkpoint_dir / filename
        raw = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     raw.state_dict(),
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

    def _save_metrics_csv(self) -> None:
        """Write the full epoch history to metrics.csv (overwritten each epoch)."""
        path = self.checkpoint_dir / "metrics.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history.keys()))
            writer.writeheader()
            n = len(self.history["epoch"])
            for i in range(n):
                writer.writerow({k: self.history[k][i] for k in self.history})

    # ── Test evaluation ───────────────────────────────────────────────────────

    def evaluate_test(
        self,
        test_ds,
        *,
        predictions_dir: Path | None = None,
    ) -> dict:
        """Thin wrapper — delegates to the module-level :func:`evaluate_test`."""
        return evaluate_test(
            self.model, self.loss_fn, test_ds, self.device, self.extra_state,
            checkpoint_dir  = self.checkpoint_dir,
            predictions_dir = predictions_dir,
        )

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
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
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


# ── Standalone test evaluation ─────────────────────────────────────────────────

def evaluate_test(
    model,
    loss_fn: ESPLoss,
    test_ds,
    device: torch.device,
    extra_state: dict[str, Any],
    *,
    checkpoint_dir: Path,
    predictions_dir: Path | None = None,
) -> dict:
    """
    Run inference on a test dataset, compute per-protein metrics, and
    optionally save per-protein prediction files.

    This is a module-level function so it can be called from a standalone
    inference script (e.g. scripts/09_evaluate_model.py) without constructing
    a full Trainer instance.  The :class:`Trainer` method delegates here.

    Iterates one protein at a time (batch_size=1) so that protein_id can be
    tracked directly via test_ds.protein_ids.  ESP values are un-normalized
    before metric computation using esp_mean/esp_std from extra_state.

    Args:
        model:           trained model in eval state after calling this function
        loss_fn:         ESPLoss instance
        test_ds:         ProteinGraphDataset for the test split (transform
                         should already be set to NormalizeESP)
        device:          torch.device to run inference on
        extra_state:     dict containing at least esp_mean and esp_std
                         (as stored in checkpoints via Trainer.extra_state)
        checkpoint_dir:  directory where test_metrics.json is written
        predictions_dir: if provided, save <protein_id>_pred.npz here for
                         every protein — each file contains query_pos,
                         pred_esp, and true_esp in kT/e

    Returns:
        dict with keys:
            "global":      {loss, rmse, mae, pearson_r, n_proteins}
            "per_protein": {protein_id: {rmse, mae, pearson_r, n_query_nodes}}

    Also writes:
        <checkpoint_dir>/test_metrics.json
    """
    if predictions_dir is not None:
        predictions_dir = Path(predictions_dir)
        if predictions_dir.exists():
            for f in predictions_dir.glob("*_pred.npz"):
                f.unlink()
        predictions_dir.mkdir(parents=True, exist_ok=True)

    esp_mean = float(extra_state.get("esp_mean", 0.0))
    esp_std  = float(extra_state.get("esp_std",  1.0))

    loader = PyGLoader(test_ds, batch_size=1, shuffle=False)
    model.eval()

    per_protein: dict[str, dict] = {}
    total_sq_err  = 0.0
    total_abs_err = 0.0
    total_n       = 0
    total_loss    = 0.0
    pearson_vals: list[float] = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            protein_id = test_ds.protein_ids[i]
            data       = data.to(device)

            pred   = model(data)
            target = data["query"].y
            batch  = data["query"].batch

            loss         = loss_fn(pred, target, batch)
            total_loss  += loss.item()

            # Un-normalize for interpretable kT/e metrics
            pred_raw   = pred   * esp_std + esp_mean
            target_raw = target * esp_std + esp_mean

            n       = target_raw.shape[0]
            sq_err  = ((pred_raw - target_raw) ** 2).sum().item()
            abs_err = (pred_raw - target_raw).abs().sum().item()
            r       = pearson_r(pred_raw, target_raw).item()

            total_sq_err  += sq_err
            total_abs_err += abs_err
            total_n       += n
            pearson_vals.append(r)

            per_protein[protein_id] = {
                "rmse":          float((sq_err / n) ** 0.5),
                "mae":           float(abs_err / n),
                "pearson_r":     float(r),
                "n_query_nodes": int(n),
            }

            if predictions_dir is not None:
                np.savez_compressed(
                    predictions_dir / f"{protein_id}_pred.npz",
                    query_pos = data["query"].pos.cpu().numpy(),
                    pred_esp  = pred_raw.cpu().numpy(),
                    true_esp  = target_raw.cpu().numpy(),
                )

    n_proteins = len(per_protein)
    global_metrics = {
        "loss":       float(total_loss / max(len(loader), 1)),
        "rmse":       float((total_sq_err  / max(total_n, 1)) ** 0.5),
        "mae":        float(total_abs_err  / max(total_n, 1)),
        "pearson_r":  float(sum(pearson_vals) / max(len(pearson_vals), 1)),
        "n_proteins": n_proteins,
    }

    results = {"global": global_metrics, "per_protein": per_protein}

    metrics_path = Path(checkpoint_dir) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Test metrics saved → %s", metrics_path)

    return results
