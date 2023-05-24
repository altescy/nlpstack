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
TModel = TypeVar("TModel", bound=TorchModel)


class TorchPredictor(Generic[Example, Inference, Prediction]):
    def __init__(
        self,
        datamodule: DataModule[Example, Inference, Prediction],
        model: TorchModel[Any, Inference],
    ):
        self.datamodule = datamodule
        self.model = model

    def predict(
        self,
        examples: Iterable[Example],
        *,
        batch_size: int = 64,
    ) -> Iterator[Prediction]:
        collator = Collator()
        self.model.eval()
        with torch.no_grad():
            for batched_examples in batched(examples, batch_size):
                instances = [self.datamodule.build_instance(example) for example in batched_examples]
                batch = move_to_device(collator(instances), self.model.get_device())
                inference = self.model.infer(**batch)
                yield from self.datamodule.build_predictions(inference)
