from typing import Generic, Iterable, Iterator, NamedTuple, Optional, Sequence, Union

import torch

from nlpstack.common import ProgressBar, batched
from nlpstack.data import Collator, DataModule
from nlpstack.integrations.torch.model import ModelInputs, PredictionParams, TorchModel
from nlpstack.integrations.torch.util import move_to_device
from nlpstack.types import Example, Inference, Prediction


class TorchPredictor(Generic[Example, Inference, Prediction, PredictionParams]):
    class SetupParams(NamedTuple):
        batch_size: Optional[int] = None
        devices: Optional[Union[int, str, Sequence[Union[int, str]]]] = None

    def __init__(
        self,
        datamodule: DataModule[Example, Inference, Prediction],
        model: TorchModel[Inference, ModelInputs, PredictionParams],
    ):
        self.datamodule = datamodule
        self.model = model
        self._batch_size = 64
        self._devices = [torch.device("cpu")]

    def setup(self, params: Optional["TorchPredictor.SetupParams"] = None) -> None:
        params = params or TorchPredictor.SetupParams()
        batch_size, devices = params
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
        params: Optional[PredictionParams],
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
                inputs = self.model.Inputs(**move_to_device(collator(instances), device))
                inference = self.model.infer(inputs, params)
                yield inference

    def predict(
        self,
        examples: Iterable[Example],
        params: Optional[PredictionParams] = None,
    ) -> Iterator[Prediction]:
        for inference in self.infer(examples, params):
            yield from self.datamodule.build_predictions(inference)
