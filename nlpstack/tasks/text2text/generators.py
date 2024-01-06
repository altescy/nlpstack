from typing import Any, List, Sequence

from nlpstack.data import TextGenerator
from nlpstack.rune import Rune

from .types import Text2TextExample, Text2TextPrediction


class Text2TextGenerator(TextGenerator):
    """
    A text generator using a rune model for text2text tasks.

    Args:
        model: The model to use.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(
        self,
        model: Rune[Text2TextExample, Text2TextPrediction],
        *,
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._model = model
        self._kwargs = kwargs

    def apply_batch(self, inputs: Sequence[str]) -> List[str]:
        examples = [Text2TextExample(text) for text in inputs]
        predictions = self._model.predict(examples, **self._kwargs)
        return [prediction.text for prediction in predictions]
