import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class MultilabelClassificationExample:
    """
    An example for multilabel classification.

    Parameters:
        text: The text.
        labels: The labels.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    text: Union[str, Sequence[Token]]
    labels: Optional[Sequence[str]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class MultilabelClassificationInference:
    """
    An inference for multilabel classification.

    Parameters:
        probs: NumPy array of predicted probabilities of shape (batch_size, num_labels).
        labels: NumPy array of predicted labels of shape (batch_size, num_labels).
        metadata: The sequence of metadata.
        top_k: The top k parameter passed from the model. This will be used for building predictions.
        threshold: The threshold parameter passed from the model. This will be used for building predictions.
    """

    probs: numpy.ndarray
    labels: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None


@dataclasses.dataclass
class MultilabelClassificationPrediction:
    """
    A prediction for multilabel classification.

    Parameters:
        top_probs: The sequence of top predicted probabilities.
        top_labels: The sequence of top predicted labels.
        metadata: The metadata.
    """

    top_probs: Sequence[float]
    top_labels: Sequence[str]
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        assert len(self.top_probs) == len(self.top_labels)
