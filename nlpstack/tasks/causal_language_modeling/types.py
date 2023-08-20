import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class CausalLanguageModelingExample:
    """
    An example for causal language modeling.

    Parameters:
        text: The text.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    text: Union[str, Sequence[Token]]
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class CausalLanguageModelingInference:
    """
    An inference for causal language modeling.

    Parameters:
        pred_token_ids: NumPy array of predicted token IDs of shape (batch_size, beam_size, max_length).
        pred_mask: NumPy array of predicted mask of shape (batch_size, beam_size, max_length).
        scores: NumPy array of scores of shape (batch_size, beam_size).
        gold_token_ids: NumPy array of gold token IDs of shape (batch_size, max_length).
        gold_mask: NumPy array of gold mask of shape (batch_size, max_length).
        perplexity: The perplexity.
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
class CausalLanguageModelingPrediction:
    """
    A prediction for causal language modeling.

    Parameters:
        top_texts: The sequence of top generated texts.
        top_tokens: The sequence of top generated tokens corresponding to the top generated texts.
        scores: The sequence of scores corresponding to the top generated texts.
        metadata: The metadata.
    """

    top_texts: Sequence[str]
    top_tokens: Sequence[Sequence[str]]
    scores: Sequence[float]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def text(self) -> str:
        """
        The best generated text.
        """
        return self.top_texts[0]

    @property
    def tokens(self) -> Sequence[str]:
        """
        The tokens of the best generated text.
        """
        return self.top_tokens[0]

    @property
    def score(self) -> float:
        """
        The score of the best generated text.
        """
        return self.scores[0]
