from __future__ import annotations

import typing
from typing import Any, Generic, Protocol, TypeVar

import torch

Output = TypeVar("Output")
Inference = TypeVar("Inference")


@typing.runtime_checkable
class LazySetup(Protocol):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        ...


class Model(torch.nn.Module, Generic[Output, Inference]):
    def forward(self, *args: Any, **kwargs: Any) -> Output:
        raise NotImplementedError

    def infer(self, *args: Any, **kwargs: Any) -> Inference:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        raise NotImplementedError

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def setup(self, *args: Any, **kwargs: Any) -> None:
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, LazySetup):
                module.setup(*args, **kwargs)
