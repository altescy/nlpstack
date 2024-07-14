from typing import Any, List, Mapping, NamedTuple, Optional, Sequence

from nlpstack.data import TextGenerator
from nlpstack.rune import Rune

from .types import CausalLanguageModelingExample, CausalLanguageModelingPrediction


class CausalLanguageModelingTextGenerator(
    TextGenerator[
        "CausalLanguageModelingTextGenerator.Fixtures",
        Optional[Mapping[str, Any]],
    ]
):
    """
    A text generator using a rune model for causal language modeling tasks.

    Args:
        model: The model to use.
        context: The shared context to use. Defaults to `None`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    class Fixtures(NamedTuple):
        model: Rune[CausalLanguageModelingExample, CausalLanguageModelingPrediction, Any, Any, Any]
        context: Optional[str]
        kwargs: Mapping[str, Any]

    def __init__(
        self,
        model: Rune[CausalLanguageModelingExample, CausalLanguageModelingPrediction, Any, Any, Any],
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._fixtures = CausalLanguageModelingTextGenerator.Fixtures(model, context, kwargs)

    @property
    def fixtures(self) -> "CausalLanguageModelingTextGenerator.Fixtures":
        return self._fixtures

    def apply_batch(
        self,
        inputs: Sequence[str],
        fixtures: "CausalLanguageModelingTextGenerator.Fixtures",
        params: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        context = fixtures.context or ""
        examples = [CausalLanguageModelingExample(context + text) for text in inputs]
        predictions = fixtures.model.predict(examples, **fixtures.kwargs, **(params or {}))
        return [prediction.text for prediction in predictions]
