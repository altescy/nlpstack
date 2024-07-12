import shutil
import tempfile
import typing
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

if typing.TYPE_CHECKING:
    from nlpstack.integrations.torch.model import TorchModelOutput
    from nlpstack.integrations.torch.training.trainer import TorchTrainer, TrainingState


class StopEarly(Exception):
    """Exception to stop training early."""


class Callback:
    """
    A base class for callbacks that can be used with :class:`TorchTrainer`.

    Callbacks are used to customize the training process. They can be used to
    implement early stopping, logging, checkpointing, etc.

    The following methods are called at the following points:

    - `on_start`: Called at the start of training.
    - `on_batch`: Called at the end of each batch.
    - `on_epoch`: Called at the end of each epoch.
    - `on_end`: Called at the end of training.
    """

    @property
    def logger(self) -> Logger:
        """
        The logger for this callback.
        """
        return getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_start(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        resources: Mapping[str, Any],
    ) -> None:
        pass

    def on_batch(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        batch_inputs: Sequence[Mapping[str, Any]],
        batch_outputs: Sequence["TorchModelOutput"],
        batch_metrics: Mapping[str, Any],
        is_training: bool,
        resources: Mapping[str, Any],
    ) -> None:
        pass

    def on_epoch(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        metrics: Mapping[str, Any],
        resources: Mapping[str, Any],
    ) -> None:
        pass

    def on_end(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        resources: Mapping[str, Any],
    ) -> None:
        pass


class EarlyStopping(Callback):
    """
    A callback to stop training early if the validation metric does not improve
    for a certain number of epochs.

    Parameters:
        patience: The number of epochs to wait for the validation metric to
            improve before stopping training. Defaults to `3`.
        metric: The validation metric to use. This should be a key in the
            `metrics` dictionary returned by the model's `forward` method.
            You can also use a key that starts with "+" or "-", in which case
            the metric will be maximized or minimized, respectively. Defaults
            to `"-valid_loss"`.
        restore_best: Whether to restore the model to the state with the best
            validation metric at the end of training. Defaults to `True`.
    """

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
        self._work_dir: Optional[Path] = None

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
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        resources: Mapping[str, Any],
    ) -> None:
        del trainer, resources
        self.best_metric = -self.direction * float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.set_work_dir()
        self.save_checkpoint(training_state)

    def on_epoch(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        metrics: Mapping[str, Any],
        resources: Mapping[str, Any],
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
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        resources: Mapping[str, Any],
    ) -> None:
        del trainer, resources
        self.logger.info(f"Best metric {self.metric}={self.best_metric} at epoch {training_state.epoch}")
        if self.restore_best:
            self.logger.info("Restoring best checkpoint...")
            self.load_checkpoint(training_state)
        self.unset_work_dir()


class MlflowCallback(Callback):
    """
    A callback to log metrics to MLflow.

    Note:
        This callback requires MLflow to be installed. You can install it with
        `pip install mlflow`.

    Parameters:
        metric_prefix: The prefix to add to metric names. This is useful to
            distinguish them from post-training evaluation metrics. Defaults
            to `"o."`, which stands for "online".
    """

    def __init__(self, metric_prefix: str = "o.") -> None:
        import mlflow

        self._mlflow = mlflow
        self._metric_prefix = metric_prefix

    def on_batch(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        batch_inputs: Sequence[Mapping[str, Any]],
        batch_outputs: Sequence["TorchModelOutput"],
        batch_metrics: Mapping[str, Any],
        is_training: bool,
        resources: Mapping[str, Any],
    ) -> None:
        if is_training:
            self._mlflow.log_metric("epoch", training_state.epoch, step=training_state.step)
            if "train_loss" in batch_metrics:
                self._log_metric("train_loss", batch_metrics["train_loss"], step=training_state.step)

    def on_epoch(
        self,
        trainer: "TorchTrainer",
        training_state: "TrainingState",
        metrics: Mapping[str, Any],
        resources: Mapping[str, Any],
    ) -> None:
        from nlpstack.integrations.mlflow.util import flatten_dict_for_mlflow_log

        metrics = flatten_dict_for_mlflow_log(metrics)
        for key, value in metrics.items():
            if key in ("train_loss",):
                continue
            self._log_metric(key, value, step=training_state.step)

    def _log_metric(self, key: str, value: Any, step: int) -> None:
        key = f"{self._metric_prefix}{key}"
        if isinstance(value, (int, float)):
            self._mlflow.log_metric(key, value, step=step)
        else:
            self._log_nonnumerical_metric(key, value, step)

    def _log_nonnumerical_metric(self, key: str, value: Any, step: int) -> None:
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = Path(_tempdir)

            temppath = tempdir / key
            temppath.write_text(repr(value))

            self._mlflow.log_artifact(temppath, f"metrics/step_{step}")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_mlflow")
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        import mlflow

        state["_mlflow"] = mlflow
        self.__dict__.update(state)
