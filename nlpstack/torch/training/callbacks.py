from __future__ import annotations

import shutil
import tempfile
import typing
from logging import Logger, getLogger
from pathlib import Path
from typing import Any

import torch

if typing.TYPE_CHECKING:
    from nlpstack.torch.training.trainer import Trainer, TrainingState


class StopEarly(Exception):
    """Exception to stop training early."""


class Callback:
    @property
    def logger(self) -> Logger:
        return getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_start(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        resources: dict[str, Any],
    ) -> None:
        pass

    def on_batch(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        batch_inputs: dict[str, Any],
        batch_outputs: dict[str, Any],
        batch_metrics: dict[str, Any],
        is_training: bool,
        resources: dict[str, Any],
    ) -> None:
        pass

    def on_epoch(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        metrics: dict[str, Any],
        resources: dict[str, Any],
    ) -> None:
        pass

    def on_end(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        resources: dict[str, Any],
    ) -> None:
        pass


class EarlyStopping(Callback):
    def __init__(
        self,
        patience: int = 3,
        metric: str = "-valid_loss",
        restore_best: bool = True,
    ) -> None:
        self.patience = patience
        self.metric = metric[1:] if metric.startswith(("+", "-")) else metric
        self.restore_best = restore_best
        self.direction = -1 if metric.startswith("-") else 1
        self.best_metric = -self.direction * float("inf")
        self.best_epoch = 0
        self.counter = 0
        self._work_dir: Path | None = None

    @property
    def work_dir(self) -> Path:
        if self._work_dir is None:
            raise RuntimeError("work_dir is not set")
        return self._work_dir

    def set_work_dir(self) -> None:
        if self._work_dir is not None:
            raise RuntimeError("work_dir is already set")
        self._work_dir = Path(tempfile.TemporaryDirectory().name)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def unset_work_dir(self) -> None:
        if self._work_dir is None:
            raise RuntimeError("work_dir is not set")
        shutil.rmtree(self._work_dir)
        self._work_dir = None

    def save_checkpoint(self, training_state: "TrainingState") -> None:
        torch.save(training_state.state_dict(), self.work_dir / "checkpoint.pt")

    def load_checkpoint(self, training_state: "TrainingState") -> None:
        training_state.load_state_dict(torch.load(self.work_dir / "checkpoint.pt"))

    def on_start(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        resources: dict[str, Any],
    ) -> None:
        del trainer, resources
        self.best_metric = -self.direction * float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.set_work_dir()
        self.save_checkpoint(training_state)

    def on_epoch(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        metrics: dict[str, Any],
        resources: dict[str, Any],
    ) -> None:
        del trainer, resources
        if self.direction * metrics[self.metric] > self.direction * self.best_metric:
            self.best_metric = metrics[self.metric]
            self.best_epoch = training_state.epoch
            self.counter = 0
            self.save_checkpoint(training_state)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                raise StopEarly

    def on_end(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        resources: dict[str, Any],
    ) -> None:
        del trainer, resources
        self.logger.info(f"Best metric {self.metric}={self.best_metric} at epoch {training_state.epoch}")
        if self.restore_best:
            self.logger.info("Restoring best checkpoint...")
            self.load_checkpoint(training_state)
        self.unset_work_dir()


class MlflowCallback(Callback):
    def __init__(self) -> None:
        import mlflow

        self._mlflow = mlflow

    def on_batch(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        batch_inputs: dict[str, Any],
        batch_outputs: dict[str, Any],
        batch_metrics: dict[str, Any],
        is_training: bool,
        resources: dict[str, Any],
    ) -> None:
        if is_training:
            self._mlflow.log_metric("epoch", training_state.epoch, step=training_state.step)
            if "train_loss" in batch_metrics:
                self._mlflow.log_metric("train_loss", batch_metrics["train_loss"], step=training_state.step)

    def on_epoch(
        self,
        trainer: "Trainer",
        training_state: "TrainingState",
        metrics: dict[str, Any],
        resources: dict[str, Any],
    ) -> None:
        for key, value in metrics.items():
            if key in ("train_loss",):
                continue
            self._mlflow.log_metric(key, value, step=training_state.step)
