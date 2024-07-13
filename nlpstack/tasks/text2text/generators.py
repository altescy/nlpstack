from typing import Any, List, Mapping, NamedTuple, Optional, Sequence

from nlpstack.data import TextGenerator
from nlpstack.rune import Rune

from .types import Text2TextExample, Text2TextPrediction


class Text2TextGenerator(
    TextGenerator[
        "Text2TextGenerator.Fixtures",
        Optional[Mapping[str, Any]],
    ]
):
    """
    A text generator using a rune model for text2text tasks.

    Args:
        model: The model to use.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    class Fixtures(NamedTuple):
        model: Rune[Text2TextExample, Text2TextPrediction]
        kwargs: Mapping[str, Any]

    def __init__(
        self,
        model: Rune[Text2TextExample, Text2TextPrediction],
        *,
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._fixtures = Text2TextGenerator.Fixtures(model, kwargs)

    @property
    def fixtures(self) -> "Text2TextGenerator.Fixtures":
        return self._fixtures

    def apply_batch(
        self,
        inputs: Sequence[str],
        fixtures: "Text2TextGenerator.Fixtures",
        params: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        examples = [Text2TextExample(text) for text in inputs]
        predictions = fixtures.model.predict(examples, **fixtures.kwargs, **(params or {}))
        return [prediction.text for prediction in predictions]
