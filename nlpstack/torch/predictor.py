from typing import Any, Generic, Iterable, Iterator, Optional, TypeVar, Union

import torch

from nlpstack.data import Collator, DataModule
from nlpstack.data.util import batched
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
        self._device = torch.device("cpu")

    def setup(
        self,
        batch_size: Optional[int] = None,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> None:
        if batch_size is not None:
            self._batch_size = batch_size
        if device is not None:
            if isinstance(device, (int, str)):
                device = torch.device(device)
            self._device = device

    def infer(
        self,
        examples: Iterable[Example],
        **kwargs: Any,
    ) -> Iterator[Inference]:
        collator = Collator()
        self.model.eval()
        self.model.to(self._device)
        with torch.no_grad():
            for batched_examples in batched(examples, self._batch_size):
                instances = [self.datamodule.build_instance(example) for example in batched_examples]
                batch = move_to_device(collator(instances), self._device)
                yield self.model.infer(**batch, **kwargs)

    def predict(
        self,
        examples: Iterable[Example],
        **kwargs: Any,
    ) -> Iterator[Prediction]:
        for inference in self.infer(examples, **kwargs):
            yield from self.datamodule.build_predictions(inference)
