from __future__ import annotations

from typing import Sequence

import torch

from nlpstack.common import tqdm
from nlpstack.data import DataLoader, Instance
from nlpstack.torch.models import Model
from nlpstack.torch.training.optimizers import AdamFactory, OptimizerFactory
from nlpstack.torch.util import move_to_device


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        max_epochs: int = 10,
    ) -> None:
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._optimizer_factory = optimizer_factory or AdamFactory()
        self._max_epochs = max_epochs

    def train(
        self,
        model: Model,
        train: Sequence[Instance],
        valid: Sequence[Instance] | None = None,
    ) -> None:
        if valid is not None and self._valid_dataloader is None:
            raise ValueError("valid_dataloader is required when valid is not None")

        optimizer = self._optimizer_factory.setup(model)

        device = model.get_device()

        with tqdm(range(1, self._max_epochs + 1), position=0, leave=True) as epochbar:
            for epoch in epochbar:
                epochbar.set_description(f"Epoch {epoch}")

                model.train()
                train_dataloader = self._train_dataloader(train)
                with tqdm(train_dataloader, position=1, leave=False) as batchbar:
                    batchbar.set_description("Training")

                    for batch in batchbar:
                        batch = move_to_device(batch, device)

                        output = model(**batch)
                        loss = output["loss"]

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batchbar.set_postfix(loss=f"{loss.item():.4f}")

                if valid is not None and self._valid_dataloader is not None:
                    model.eval()
                    valid_dataloader = self._valid_dataloader(valid)
                    with torch.no_grad(), tqdm(valid_dataloader, position=1, leave=False) as batchbar:
                        batchbar.set_description("Validating")

                        for batch in batchbar:
                            output = model(**batch)
                            loss = output["loss"]

                            batchbar.set_postfix(loss=f"{loss.item():.4f}")
