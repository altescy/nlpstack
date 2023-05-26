from __future__ import annotations

import typing
from typing import Any, Generic, Protocol, TypeVar

import torch

Inference = TypeVar("Inference")


@typing.runtime_checkable
class LazySetup(Protocol):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        ...


@typing.runtime_checkable
class TorchModelOutput(Protocol[Inference]):
    inference: Inference
    loss: torch.FloatTensor | None


class TorchModel(torch.nn.Module, Generic[Inference]):
    def forward(self, *args: Any, **kwargs: Any) -> TorchModelOutput[Inference]:
        raise NotImplementedError

    def infer(self, *args: Any, **kwargs: Any) -> Inference:
        return self.forward(*args, **kwargs).inference

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def setup(self, *args: Any, **kwargs: Any) -> None:
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, LazySetup):
                module.setup(*args, **kwargs)
