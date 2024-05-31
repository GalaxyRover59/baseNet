"""Utils for training myNet models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

if TYPE_CHECKING:
    import dgl
    import numpy as np
    from torch.optim import Optimizer


class myNetLightningModuleMixin:
    """Mix-in class implementing common functions for training."""

    def training_step(self, batch: tuple, batch_idx: int):
        """Training step.

        Args:
            batch: Data batch
            batch_idx: Batch index

        Returns:
            Total loss
        """
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"train_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )

        return results["Total_Loss"]

    def on_train_epoch_end(self):
        """Step scheduler every epoch."""
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch: tuple, batch_idx: int):
        """Validation step.

        Args:
            batch: Data batch
            batch_idx: Batch index

        Returns:
            Total loss
        """
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"val_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )
        return results["Total_Loss"]

    def test_step(self, batch: tuple, batch_idx: int):
        """Test step.

        Args:
            batch: Data batch
            batch_idx: Batch index

        Returns:
            Test result
        """
        torch.set_grad_enabled(True)
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"test_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )
        return results

    def configure_optimizers(self):
        """Configure optimizers."""
        if self.optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                eps=1e-8,
            )
        else:
            optimizer = self.optimizer
        if self.scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay_steps,
                eta_min=self.lr * self.decay_alpha,
            )
        else:
            scheduler = self.scheduler
        return [optimizer, ], [scheduler, ]

    def on_test_model_eval(self, *args, **kwargs):
        """Executed on model testing.

        Args:
            *args: Pass-through
            **kwargs: Pass-through
        """
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step.

        Args:
            batch: Data batch
            batch_idx: Batch index
            dataloader_idx: Data loader index

        Returns:
            Prediction
        """
        torch.set_grad_enabled(True)
        return self.step(batch)


class ModelLightningModule(myNetLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MLP models."""

    def __init__(
            self,
            model,
            include_line_graph: bool = False,
            data_mean: float = 0.0,
            data_std: float = 1.0,
            loss: str = "mse_loss",
            optimizer: Optimizer | None = None,
            scheduler=None,
            lr: float = 0.001,
            decay_steps: int = 1000,
            decay_alpha: float = 0.01,
            sync_dist: bool = False,
            **kwargs,
    ):
        """Init ModelLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            include_line_graph: Whether to include line graphs
            data_mean: Average of training data
            data_std: Standard deviation of training data
            loss: Function used for training
            optimizer: Optimizer for training
            scheduler: Scheduler for training
            lr: Learning rate for training
            decay_steps: Number of steps for decaying learning rate
            decay_alpha: Parameter determines the minimum learning rate
            sync_dist: Whether sync logging across all GPU workers or not
            **kwargs: Pass-through to parent init
        """
        super().__init__(**kwargs)

        self.model = model
        self.include_line_graph = include_line_graph
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.data_mean = data_mean
        self.data_std = data_std
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        if loss == "mse_loss":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.save_hyperparameters(ignore=["model"])

    def forward(
            self,
            g: dgl.DGLGraph,
            lat: torch.Tensor | None = None,
            l_g: dgl.DGLGraph | None = None,
            state_attr: torch.Tensor | None = None,
    ):
        """
        Args:
            g: dgl Graph
            lat: Lattice
            l_g: Line graph
            state_attr: State attribute

        Returns:
            Model prediction
        """
        g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
                g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.include_line_graph:
            return self.model(g=g, l_g=l_g, state_attr=state_attr)
        return self.model(g, state_attr=state_attr)

    def step(self, batch: tuple):
        """
        Args:
            batch: Batch of training data

        Returns:
            results, batch_size
        """
        if self.include_line_graph:
            g, lat, l_g, state_attr, labels = batch
            preds = self(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
        else:
            g, lat, state_attr, labels = batch
            preds = self(g=g, lat=lat, state_attr=state_attr)
        results = self.loss_fn(loss=self.loss, preds=preds, labels=labels)  # type: ignore
        batch_size = preds.numel()
        # results['targets'] = labels
        # results['predictions'] = preds
        return results, batch_size

    def loss_fn(self, loss: nn.Module, labels: torch.Tensor, preds: torch.Tensor):
        """
        Args:
            loss: Loss function
            labels: Labels to compute the loss
            preds: Predictions

        Returns:
            {Total_Loss, MAE, RMSE}
        """
        scaled_pred = torch.reshape(preds * self.data_std + self.data_mean, labels.size())
        total_loss = loss(labels, scaled_pred)
        mae = self.mae(labels, scaled_pred)
        rmse = self.rmse(labels, scaled_pred)
        return {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse}
