from __future__ import annotations

from typing import Generic, Iterable, Iterator, TypeVar

import torch

from nlpstack.data import Collator, DataModule
from nlpstack.data.util import batched
from nlpstack.torch.models import Model

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")


class Predictor(Generic[Example, Prediction]):
    def __init__(
        self,
        datamodule: DataModule[Example],
        model: Model,
    ):
        self.datamodule = datamodule
        self.model = model

    def generate_predictions_from_model_output(self, model_output: dict[str, list]) -> Iterator[Prediction]:
        raise NotImplementedError

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
                batch = collator(instances)
                model_output = self.model(**batch)
                yield from self.generate_predictions_from_model_output(model_output)
