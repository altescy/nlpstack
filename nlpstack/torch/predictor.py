from typing import Any, Generic, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import torch

from nlpstack.common import ProgressBar, batched
from nlpstack.data import Collator, DataModule
from nlpstack.torch.model import TorchModel
from nlpstack.torch.util import move_to_device

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")


class TorchPredictor(Generic[Example, Inference, Prediction]):
    def __init__(
        self,
        datamodule: DataModule[Example, Inference, Prediction],
        model: TorchModel[Inference],
    ):
        self.datamodule = datamodule
        self.model = model
        self._batch_size = 64
        self._devices = [torch.device("cpu")]

    def setup(
        self,
        batch_size: Optional[int] = None,
        devices: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> None:
        if batch_size is not None:
            self._batch_size = batch_size
        if devices is not None:
            if isinstance(devices, (int, str)):
                devices = [devices]
            if len(devices) > 1:
                raise ValueError("Currently only single GPU is supported.")
            self._devices = [torch.device(device) for device in devices]

    def infer(
        self,
        examples: Iterable[Example],
        **kwargs: Any,
    ) -> Iterator[Inference]:
        if len(self._devices) > 1:
            raise ValueError("Currently only single GPU is supported.")
        device = self._devices[0]
        collator = Collator()
        self.model.eval()
        self.model.to(device)
        with torch.inference_mode(), ProgressBar(examples, desc="Predicting") as examples:
            for batched_examples in batched(examples, self._batch_size):
                instances = [self.datamodule.build_instance(example) for example in batched_examples]
                batch = move_to_device(collator(instances), device)
                inference = self.model.infer(**batch, **kwargs)
                yield inference

    def predict(
        self,
        examples: Iterable[Example],
        **kwargs: Any,
    ) -> Iterator[Prediction]:
        for inference in self.infer(examples, **kwargs):
            yield from self.datamodule.build_predictions(inference)
