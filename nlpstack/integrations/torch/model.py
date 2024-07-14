import typing
from typing import Any, Generic, Optional, Protocol, Tuple, Type, TypeVar, cast

import torch

Inference = TypeVar("Inference")
ModelInputs = TypeVar("ModelInputs", bound=Tuple[Any, ...])
PredictionParams = TypeVar("PredictionParams")


@typing.runtime_checkable
class LazySetup(Protocol):
    def setup(self, *args: Any, **kwargs: Any) -> None: ...


@typing.runtime_checkable
class TorchModelOutput(Protocol[Inference]):
    inference: Inference
    loss: Optional[torch.FloatTensor]


class TorchModel(torch.nn.Module, Generic[Inference, ModelInputs, PredictionParams]):
    Inputs: Type[ModelInputs]

    def __call__(
        self,
        inputs: ModelInputs,
        params: Optional[PredictionParams] = None,
    ) -> TorchModelOutput[Inference]:
        return cast(TorchModelOutput[Inference], super().__call__(inputs, params))

    def forward(
        self,
        inputs: ModelInputs,
        params: Optional[PredictionParams] = None,
    ) -> TorchModelOutput[Inference]:
        raise NotImplementedError

    @torch.no_grad()
    def infer(
        self,
        inputs: ModelInputs,
        params: Optional[PredictionParams] = None,
    ) -> Inference:
        return self.forward(inputs, params).inference

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def setup(self, *args: Any, **kwargs: Any) -> None:
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, LazySetup):
                module.setup(*args, **kwargs)
