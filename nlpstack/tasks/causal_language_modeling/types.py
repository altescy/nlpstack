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
    gold_token_ids: Optional[numpy.ndarray] = None
    gold_mask: Optional[numpy.ndarray] = None
    perplexity: Optional[float] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class CausalLanguageModelingPrediction:
    top_tokens: Sequence[Sequence[str]]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def tokens(self) -> Sequence[str]:
        return self.top_tokens[0]
