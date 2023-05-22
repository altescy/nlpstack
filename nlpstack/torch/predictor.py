from __future__ import annotations

from typing import Generic, Iterable, Iterator, TypeVar

import torch

from nlpstack.data import Collator, DataModule
from nlpstack.data.util import batched
from nlpstack.torch.models import Model
from nlpstack.torch.util import move_to_device, tensor_to_numpy

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")
TDataModule = TypeVar("TDataModule", bound=DataModule)
TModel = TypeVar("TModel", bound=Model)


class TorchPredictor(Generic[Example, Prediction, TDataModule[Example, Prediction]]):
    def __init__(
        self,
        datamodule: TDataModule,
        model: Model,
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
                model_output = tensor_to_numpy(self.model(**batch))
                yield from self.datamodule.build_predictions(**model_output)
