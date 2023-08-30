from typing import Any, List, Optional, Sequence

from nlpstack.data import TextGenerator
from nlpstack.rune import Rune

from .types import CausalLanguageModelingExample, CausalLanguageModelingPrediction


class CausalLanguageModelingTextGenerator(TextGenerator):
    """
    A text generator using a rune model for causal language modeling tasks.

    Args:
        model: The model to use.
        context: The shared context to use. Defaults to `None`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(
        self,
        model: Rune[CausalLanguageModelingExample, CausalLanguageModelingPrediction],
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._context = context
        self._kwargs = kwargs

    def __call__(self, inputs: Sequence[str], **kwargs: Any) -> List[str]:
        context = self._context or ""
        examples = [CausalLanguageModelingExample(context + text) for text in inputs]
        predictions = self._model.predict(examples, **self._kwargs, **kwargs)
        return [prediction.text for prediction in predictions]
