import dataclasses
from typing import Any, Dict, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class ClassificationExample:
    """
    An example for classification.

    Parameters:
        text: The text.
        label: The label.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    text: Union[str, Sequence[Token]]
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ClassificationInference:
    """
    An inference for classification.

    Parameters:
        probs: NumPy array of probabilities of shape `(batch_size, num_labels)`.
        labels: NumPy array of labels of shape `(batch_size,)`.
        metadata: The sequence of metadata.
        top_k: The top k parameter passed from the model. This will be used for building predictions.
        threshold: The threshold parameter passed from the model. This will be used for building predictions.
    """

    probs: numpy.ndarray
    labels: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Dict[str, Any]]] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None


@dataclasses.dataclass
class ClassificationPrediction:
    """
    A prediction for classification.

    Parameters:
        top_probs: The sequence of top probabilities.
        top_labels: The sequence of top labels corresponding to the top probabilities.
        metadata: The metadata.
    """

    top_probs: Sequence[float]
    top_labels: Sequence[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        assert len(self.top_probs) == len(self.top_labels)

    @property
    def label(self) -> str:
        return self.top_labels[0]

    @property
    def prob(self) -> float:
        return self.top_probs[0]
