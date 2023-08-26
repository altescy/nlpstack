import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class Text2TextExample:
    """
    An example for text-to-text task.

    Parameters:
        source: The source text.
        target: The target text.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    source: Union[str, Sequence[Token]]
    target: Optional[Union[str, Sequence[Token]]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class Text2TextInference:
    """
    An inference for text-to-text task.

    pred_token_ids: NumPy array of predicted token IDs of shape `(batch_size, beam_size, sequence_length)`
    pred_mask: NumPy array of predicted sequence mask of shape `(batch_size, beam_size, sequence_length)`
    gold_token_ids: NumPy array of the gold token IDs of shape `(batch_size, sequence_length)`
    gold_mask: NumPy array of the gold sequence mask of shape `(batch_size, sequence_length)`
    perplexity: The perplexity of the predicted tokens. This is available only when the gold sequences are given.
    metadata: The sequence of metadata.
    """

    pred_token_ids: numpy.ndarray
    pred_mask: numpy.ndarray
    scores: numpy.ndarray
    gold_token_ids: Optional[numpy.ndarray] = None
    gold_mask: Optional[numpy.ndarray] = None
    perplexity: Optional[float] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class Text2TextPrediction:
    """
    A prediction for text-to-text task.

    Parameters:
        top_texts: The sequence of the top predicted texts.
        top_texts: The sequence of the top predicted tokens.
        scores: The scores of the top predicted texts.
    """

    top_texts: Sequence[str]
    top_tokens: Sequence[Sequence[str]]
    scores: Sequence[float]

    @property
    def text(self) -> str:
        return self.top_texts[0]

    @property
    def tokens(self) -> Sequence[str]:
        return self.top_tokens[0]

    @property
    def score(self) -> float:
        return self.scores[0]
