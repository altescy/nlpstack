import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class CausalLanguageModelingExample:
    text: Union[str, Sequence[Token]]
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class CausalLanguageModelingInference:
    pred_token_ids: numpy.ndarray
    pred_mask: numpy.ndarray
    scores: numpy.ndarray
    gold_token_ids: Optional[numpy.ndarray] = None
    gold_mask: Optional[numpy.ndarray] = None
    perplexity: Optional[float] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class CausalLanguageModelingPrediction:
    top_texts: Sequence[str]
    top_tokens: Sequence[Sequence[str]]
    scores: Sequence[float]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def text(self) -> str:
        return self.top_texts[0]

    @property
    def tokens(self) -> Sequence[str]:
        return self.top_tokens[0]

    @property
    def score(self) -> float:
        return self.scores[0]
