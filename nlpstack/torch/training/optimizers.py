from typing import Any, Type

import torch
from torch.optim import Optimizer

if torch.__version__ >= "2.0.0":
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class OptimizerFactory:
    def __init__(
        self,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        self._optimizer_cls = optimizer_cls
        self._kwargs = kwargs

    def setup(self, model: torch.nn.Module) -> Optimizer:
        return self._optimizer_cls(model.parameters(), **self._kwargs)


class AdamFactory(OptimizerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.Adam, **kwargs)


class AdamWFactory(OptimizerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.AdamW, **kwargs)


class SGDFactory(OptimizerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.SGD, **kwargs)


class RMSpropFactory(OptimizerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.RMSprop, **kwargs)


class RpropFactory(OptimizerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.Rprop, **kwargs)


class LRSchedulerFactory:
    def __init__(
        self,
        scheduler_cls: Type[LRScheduler],
        **kwargs: Any,
    ) -> None:
        self._scheduler_cls = scheduler_cls
        self._kwargs = kwargs

    def setup(self, optimizer: torch.optim.Optimizer) -> LRScheduler:
        return self._scheduler_cls(optimizer, **self._kwargs)  # type: ignore[arg-type]


class StepLRFactory(LRSchedulerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.lr_scheduler.StepLR, **kwargs)


class MultiStepLRFactory(LRSchedulerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.lr_scheduler.MultiStepLR, **kwargs)


class ExponentialLRFactory(LRSchedulerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.lr_scheduler.ExponentialLR, **kwargs)


class CosineAnnealingLRFactory(LRSchedulerFactory):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(torch.optim.lr_scheduler.CosineAnnealingLR, **kwargs)
