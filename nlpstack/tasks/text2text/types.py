import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class Text2TextExample:
    source: Union[str, Sequence[Token]]
    target: Optional[Union[str, Sequence[Token]]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class Text2TextInference:
    pred_token_ids: numpy.ndarray
    pred_mask: numpy.ndarray
    gold_token_ids: Optional[numpy.ndarray] = None
    gold_mask: Optional[numpy.ndarray] = None
    perplexity: Optional[float] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class Text2TextPrediction:
    top_texts: Sequence[str]
    top_tokens: Sequence[Sequence[str]]

    @property
    def text(self) -> str:
        return self.top_texts[0]

    @property
    def tokens(self) -> Sequence[str]:
        return self.top_tokens[0]
