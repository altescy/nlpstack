import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy

from nlpstack.data.tokenizers import Token

Decoding = Tuple[Sequence[int], float]


@dataclasses.dataclass
class SequenceLabelingExample:
    """
    An example for sequence labeling task.

    Parameters:
        text: The text.
        labels: The label sequence.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    text: Union[str, Sequence[Token]]
    labels: Optional[Sequence[str]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class SequenceLabelingInference:
    """
    An inference for sequence labeling task.

    Parameters:
        probs: NumPy array of label sequence probabilities of shape `(batch_size, sequence_length, num_labels)`.
        mask: Numpy array of sequence mask of shape `(batch_size, sequence_length)`.
        decodings: The decoding result generated by the CRF decoder.
        labels: NumPy array of gold sequence labels of shape `(batch_size, sequence_length)`.
        metadata: The sequence of metadata.
    """

    probs: numpy.ndarray
    mask: Optional[numpy.ndarray] = None
    decodings: Optional[Sequence[Sequence[Decoding]]] = None
    labels: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class SequenceLabelingPrediction:
    """
    A prediction for sequence labeling task.

    Parameters:
        tokens: The sequence of the given tokens.
        top_labels: The sequence of the predicted labels.
        metadata: The metadata.
    """

    tokens: Sequence[str]
    top_labels: Sequence[Sequence[str]]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def labels(self) -> Sequence[str]:
        return self.top_labels[0]
