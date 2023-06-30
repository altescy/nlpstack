import dataclasses
import warnings
from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Sequence, TypeVar, Union

import torch

from nlpstack.common import ProgressBar
from nlpstack.data import DataLoader, Instance
from nlpstack.evaluation import EmptyMetric, Metric
from nlpstack.torch.model import TorchModel
from nlpstack.torch.training.callbacks import Callback, StopEarly
from nlpstack.torch.training.optimizers import AdamFactory, LRSchedulerFactory, OptimizerFactory
from nlpstack.torch.util import move_to_device

logger = getLogger(__name__)

Inference = TypeVar("Inference")


@dataclasses.dataclass
class TrainingState:
    """The state of the training loop.

    TrainingState is a dataclass that stores the state of the training loop. It is used by the Trainer
    class to store the current epoch, step, model, optimizer, and learning rate scheduler.

    Args:
        epoch: The current epoch.
        step: The current step.
        model: The model being trained.
        optimizer: The optimizer used for training.
        lrscheduler: The learning rate scheduler used for training. Defaults to None.
    """

    epoch: int
    step: int
    model: TorchModel
    optimizer: torch.optim.Optimizer
    lrscheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lrscheduler": self.lrscheduler.state_dict() if self.lrscheduler else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.lrscheduler is not None:
            self.lrscheduler.load_state_dict(state_dict["lr_scheduler"])


class TorchTrainer:
    """Trainer for PyTorch models.

    TorchTrainer is a class that handles the training loop for PyTorch models. It is responsible for iterating
    over the training data, computing the loss, and updating the model parameters by gradient descent.

    Args:
        max_epochs: The maximum number of epochs to train for. Defaults to 10.
        batch_size: The batch size to use for training. If train_dataloader or valid_dataloader
            is provided, this is ignored. Defaults to 32.
        learning_rate: The learning rate to use for training. If optimizer_factory is provided,
            this is ignored. Defaults to 1e-3.
        train_dataloader: The dataloader to use for training. If provided, batch_size is ignored.
            Defaults to `DataLoader(batch_size, shuffled=True)`.
        valid_dataloader: The dataloader to use for validation. If provided, batch_size is ignored.
            Defaults to `DataLoader(batch_size, shuffled=False)`.
        optimizer_factory: The optimizer factory to use for training. If provided, learning_rate is
            ignored. Defaults to `AdamFactory(lr=learning_rate)`.
        lrscheduler_factory: The learning rate scheduler factory to use for training. Defaults to None.
        callbacks: The callbacks to use for training. Defaults to None.
        devices: The devices to use for training. Defaults to None.
    """

    def __init__(
        self,
        *,
        max_epochs: int = 10,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        train_dataloader: Optional[DataLoader] = None,
        valid_dataloader: Optional[DataLoader] = None,
        optimizer_factory: Optional[OptimizerFactory] = None,
        lrscheduler_factory: Optional[LRSchedulerFactory] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        devices: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    ) -> None:
        """Initializes a new Trainer instance.

        The __init__ method of Trainer takes a number of optional arguments that can be used to configure
        the training loop. Please use callbacks if you need to customize the training loop such as early
        stopping, checkpointing, logging, etc.

        """

        if batch_size is not None and train_dataloader is not None:
            warnings.warn("batch_size is ignored when train_dataloader is provided")
        if batch_size is not None and valid_dataloader is not None:
            warnings.warn("batch_size is ignored when valid_dataloader is provided")
        if learning_rate is not None and optimizer_factory is not None:
            warnings.warn("learning_rate is ignored when optimizer_factory is provided")
        if devices is not None and not isinstance(devices, (int, str)) and len(devices) > 1:
            raise ValueError("Currently, only a single device is supported")

        learning_rate = learning_rate or 1e-3
        available_dataloader = train_dataloader or valid_dataloader
        batch_size = batch_size or (available_dataloader._batch_size if available_dataloader else 32)
        devices = [devices] if isinstance(devices, (int, str)) else devices

        self._train_dataloader = train_dataloader or DataLoader(batch_size=batch_size, shuffle=True)
        self._valid_dataloader = valid_dataloader or DataLoader(batch_size=batch_size, shuffle=False)
        self._optimizer_factory = optimizer_factory or AdamFactory(lr=learning_rate)
        self._lrscheduler_factory = lrscheduler_factory
        self._max_epochs = max_epochs
        self._callbacks = callbacks or []
        self._devices = [torch.device(device) for device in devices] if devices else [torch.device("cpu")]

    def _get_metrics(
        self,
        total_loss: float,
        num_batches: int,
        training_state: TrainingState,
        metric: Metric[Inference],
        reset: bool = False,
        prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

        if metric is not None:
            metrics.update(metric.compute())
            if reset:
                metric.reset()

        if prefix is not None:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        return metrics

    def train(
        self,
        model: TorchModel[Inference],
        train: Sequence[Instance],
        valid: Optional[Sequence[Instance]] = None,
        *,
        metric: Optional[Metric[Inference]] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> TrainingState:
        """Runs the training/validation loop with the given model and training data.

        Args:
            model: The model to train.
            train: The training dataset.
            valid: The validation dataset. Defaults to None.
            resources: Additional resources to pass to callbacks. Defaults to None.

        Returns:
            :obj:`TrainingState`: The final training state.
        """

        if self._devices is not None:
            if len(self._devices) > 1:
                raise ValueError("Currently, only a single device is supported")
            model = model.to(device=self._devices[0])

        if valid is not None and self._valid_dataloader is None:
            raise ValueError("valid_dataloader is required when valid is not None")

        metric = metric or EmptyMetric()
        resources = resources or {}

        device = model.get_device()
        optimizer = self._optimizer_factory.setup(model)
        lrscheduler = self._lrscheduler_factory.setup(optimizer) if self._lrscheduler_factory else None

        state = TrainingState(
            epoch=0,
            step=0,
            model=model,
            optimizer=optimizer,
            lrscheduler=lrscheduler,
        )

        metrics: Dict[str, float] = {}

        total_batches = self._max_epochs * len(self._train_dataloader(train))
        if valid is not None:
            total_batches += self._max_epochs * len(self._valid_dataloader(valid))

        # setup progress bar
        epoch_digits = len(str(self._max_epochs))
        totalbar_desc = "Epoch {epoch:$edig$d}/$meps$"
        totalbar_desc = totalbar_desc.replace("$edig$", str(epoch_digits)).replace("$meps$", str(self._max_epochs))
        trainbar_desc = "Training"
        validbar_desc = "Validating"
        max_desc_len = max(len(totalbar_desc.format(epoch=0)), len(trainbar_desc), len(validbar_desc))
        totalbar_template = "{desc:<$mdl$s}  {percentage:3d}%  {bar}  {elapsed_time}<{remaining_time}"
        batchbar_template = "{desc:<$mdl$s}  {percentage:3d}%  {bar}  {average_iterations}it/s"
        totalbar_template = totalbar_template.replace("$mdl$", str(max_desc_len))
        batchbar_template = batchbar_template.replace("$mdl$", str(max_desc_len))

        for callback in self._callbacks:
            callback.on_start(
                trainer=self,
                training_state=state,
                resources=resources,
            )

        try:
            with ProgressBar[int](total_batches, desc="", template=totalbar_template) as totalbar:
                for epoch in range(1, self._max_epochs + 1):
                    totalbar.set_description(totalbar_desc.format(epoch=epoch))

                    model.train()
                    metric.reset()
                    train_dataloader = self._train_dataloader(train)
                    with ProgressBar(
                        train_dataloader,
                        desc=trainbar_desc,
                        template=batchbar_template,
                        leave=False,
                    ) as batchbar:
                        num_train_batches = 0
                        total_train_loss = 0.0
                        for batch in batchbar:
                            batch = move_to_device(batch, device)

                            output = model(**batch)
                            loss = output.loss

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            state.step += 1
                            num_train_batches += 1
                            total_train_loss += loss.item()

                            metric.update(output.inference)

                            batch_train_metrics = self._get_metrics(
                                training_state=state,
                                total_loss=loss.item(),
                                num_batches=1,
                                metric=metric,
                                reset=False,
                                prefix="train_",
                            )

                            for callback in self._callbacks:
                                callback.on_batch(
                                    trainer=self,
                                    training_state=state,
                                    batch_inputs=batch,
                                    batch_outputs=output,
                                    batch_metrics=batch_train_metrics,
                                    is_training=True,
                                    resources=resources,
                                )

                            batchbar.set_postfix(
                                **{
                                    key: f"{value:.2f}"
                                    for key, value in self._get_metrics(
                                        training_state=state,
                                        total_loss=total_train_loss,
                                        num_batches=num_train_batches,
                                        metric=metric,
                                        reset=False,
                                    ).items()
                                }
                            )

                            totalbar.update()

                    metrics = self._get_metrics(
                        training_state=state,
                        total_loss=total_train_loss,
                        num_batches=num_train_batches,
                        metric=metric,
                        reset=True,
                        prefix="train_",
                    )

                    if valid is not None and self._valid_dataloader is not None:
                        model.eval()
                        metric.reset()
                        valid_dataloader = self._valid_dataloader(valid)
                        with torch.no_grad(), ProgressBar(
                            valid_dataloader,
                            desc=validbar_desc,
                            template=batchbar_template,
                            leave=False,
                        ) as batchbar:
                            batchbar.set_description(validbar_desc)

                            num_valid_batches = 0
                            total_valid_loss = 0.0
                            for batch in batchbar:
                                batch = move_to_device(batch, device)

                                output = model(**batch)
                                loss = output.loss

                                num_valid_batches += 1
                                total_valid_loss += loss.item()

                                metric.update(output.inference)

                                batch_valid_metrics = self._get_metrics(
                                    training_state=state,
                                    total_loss=loss.item(),
                                    num_batches=1,
                                    metric=metric,
                                    reset=False,
                                    prefix="valid_",
                                )

                                for callback in self._callbacks:
                                    callback.on_batch(
                                        trainer=self,
                                        training_state=state,
                                        batch_inputs=batch,
                                        batch_outputs=output,
                                        batch_metrics=batch_valid_metrics,
                                        is_training=False,
                                        resources=resources,
                                    )

                                batchbar.set_postfix(
                                    **{
                                        key: f"{value:.2f}"
                                        for key, value in self._get_metrics(
                                            training_state=state,
                                            total_loss=total_valid_loss,
                                            num_batches=num_valid_batches,
                                            metric=metric,
                                            reset=False,
                                        ).items()
                                    }
                                )

                                totalbar.update()

                        metrics.update(
                            self._get_metrics(
                                training_state=state,
                                total_loss=total_valid_loss,
                                num_batches=num_valid_batches,
                                metric=metric,
                                reset=True,
                                prefix="valid_",
                            )
                        )

                    if lrscheduler is not None:
                        lrscheduler.step()

                    logger.info(
                        f"Epoch {epoch}/{self._max_epochs} - "
                        + " ".join(f"{key}={value:.2f}" for key, value in metrics.items())
                    )

                    state.epoch += 1

                    for callback in self._callbacks:
                        callback.on_epoch(
                            trainer=self,
                            training_state=state,
                            metrics=metrics,
                            resources=resources,
                        )
        except StopEarly:
            logger.info("Training stopped early!")
        except KeyboardInterrupt:
            logger.info("Training interrupted!")

        for callback in self._callbacks:
            callback.on_end(
                trainer=self,
                training_state=state,
                resources=resources,
            )

        return state
