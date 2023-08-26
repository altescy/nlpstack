import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class RepresentationLearningExample:
    """
    An example for representation learning.

    Parameters:
        text: The text.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    text: Union[str, Sequence[Token]]
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class RepresentationLearningInference:
    """
    An inference for representation learning.

    Parameters:
        embeddings: Numpy array of text representations of shape `(batch_size, embedding_size)`.
        metadata: The sequence of metadata.
    """

    embeddings: numpy.ndarray
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class RepresentationLearningPrediction:
    """
    A prediction for representation learning.

    Parameters:
        embeddings: The text representation vector.
        metadata: The metadata.
    """

    embedding: Sequence[float]
    metadata: Optional[Mapping[str, Any]] = None
