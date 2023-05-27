from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, TypeVar

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

    def infer(
        self,
        examples: Iterable[Example],
        *,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterator[Inference]:
        collator = Collator()
        self.model.eval()
        with torch.no_grad():
            for batched_examples in batched(examples, batch_size):
                instances = [self.datamodule.build_instance(example) for example in batched_examples]
                batch = move_to_device(collator(instances), self.model.get_device())
                yield self.model.infer(**batch)

    def predict(
        self,
        examples: Iterable[Example],
        *,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterator[Prediction]:
        for inference in self.infer(examples, batch_size=batch_size):
            yield from self.datamodule.build_predictions(inference)
