from __future__ import annotations

import dataclasses
from logging import getLogger
from typing import Any, Sequence

import torch

from nlpstack.common import tqdm
from nlpstack.data import DataLoader, Instance
from nlpstack.torch.models import Model
from nlpstack.torch.training.callbacks import Callback, StopEarly
from nlpstack.torch.training.optimizers import AdamFactory, LRSchedulerFactory, OptimizerFactory
from nlpstack.torch.util import move_to_device

logger = getLogger(__name__)


@dataclasses.dataclass
class TrainingState:
    epoch: int
    step: int
    model: Model
    optimizer: torch.optim.Optimizer
    lrscheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lrscheduler": self.lrscheduler.state_dict() if self.lrscheduler else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.lrscheduler is not None:
            self.lrscheduler.load_state_dict(state_dict["lr_scheduler"])


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        lrscheduler_factory: LRSchedulerFactory | None = None,
        max_epochs: int = 10,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._optimizer_factory = optimizer_factory or AdamFactory()
        self._lrscheduler_factory = lrscheduler_factory
        self._max_epochs = max_epochs
        self._callbacks = callbacks or []

    def _get_metrics(
        self,
        training_state: TrainingState,
        total_loss: float,
        num_batches: int,
        reset: bool = False,
        prefix: str | None = None,
    ) -> dict[str, float]:
        metrics = training_state.model.get_metrics(reset=reset)
        metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        if prefix is not None:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        return metrics

    def train(
        self,
        model: Model,
        train: Sequence[Instance],
        valid: Sequence[Instance] | None = None,
        resources: dict[str, Any] | None = None,
    ) -> TrainingState:
        if valid is not None and self._valid_dataloader is None:
            raise ValueError("valid_dataloader is required when valid is not None")

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

        metrics: dict[str, float] = {}

        for callback in self._callbacks:
            callback.on_start(
                trainer=self,
                training_state=state,
                resources=resources,
            )

        try:
            with tqdm(range(1, self._max_epochs + 1), position=0, leave=True) as epochbar:
                for epoch in epochbar:
                    epochbar.set_description(f"Epoch {epoch}")

                    model.train()
                    train_dataloader = self._train_dataloader(train)
                    with tqdm(train_dataloader, position=1, leave=False) as batchbar:
                        batchbar.set_description("Training")

                        num_train_batches = 0
                        total_train_loss = 0.0
                        for batch in batchbar:
                            batch = move_to_device(batch, device)

                            output = model(**batch)
                            loss = output["loss"]

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            state.step += 1
                            num_train_batches += 1
                            total_train_loss += loss.item()

                            batch_train_metrics = self._get_metrics(
                                training_state=state,
                                total_loss=loss.item(),
                                num_batches=1,
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
                                **self._get_metrics(
                                    training_state=state,
                                    total_loss=total_train_loss,
                                    num_batches=num_train_batches,
                                    reset=False,
                                )
                            )

                    metrics = self._get_metrics(
                        training_state=state,
                        total_loss=total_train_loss,
                        num_batches=num_train_batches,
                        reset=True,
                        prefix="train_",
                    )

                    if valid is not None and self._valid_dataloader is not None:
                        model.eval()
                        valid_dataloader = self._valid_dataloader(valid)
                        with torch.no_grad(), tqdm(valid_dataloader, position=1, leave=False) as batchbar:
                            batchbar.set_description("Validating")

                            num_valid_batches = 0
                            total_valid_loss = 0.0
                            for batch in batchbar:
                                batch = move_to_device(batch, device)

                                output = model(**batch)
                                loss = output["loss"]

                                num_valid_batches += 1
                                total_valid_loss += loss.item()

                                batch_valid_metrics = self._get_metrics(
                                    training_state=state,
                                    total_loss=loss.item(),
                                    num_batches=1,
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
                                    **self._get_metrics(
                                        training_state=state,
                                        total_loss=total_valid_loss,
                                        num_batches=num_valid_batches,
                                        reset=False,
                                    )
                                )

                        metrics.update(
                            self._get_metrics(
                                training_state=state,
                                total_loss=total_valid_loss,
                                num_batches=num_valid_batches,
                                reset=True,
                                prefix="valid_",
                            )
                        )

                    if lrscheduler is not None:
                        lrscheduler.step()

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
