from __future__ import annotations

from typing import Any, Type

import torch


class OptimizerFactory:
    def __init__(
        self,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: dict[str, Any],
    ) -> None:
        self._optimizer_cls = optimizer_cls
        self._kwargs = kwargs

    def setup(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self._optimizer_cls(model.parameters(), **self._kwargs)


class AdamFactory(OptimizerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.Adam, **kwargs)
