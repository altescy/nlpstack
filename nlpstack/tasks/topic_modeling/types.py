import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class TopicModelingExample:
    """
    An example for topic modeling task.

    Parameters:
        text: The text.
        metadata: The metadata. This will be passed to the inference and prediction.
    """

    text: Union[str, Sequence[Token]]
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class TopicModelingInference:
    """
    An inference for topic modeling task.

    topic_distribution: NumPy array of topic distribution of shape `(batch_size, num_topics)`.
    token_counts: NumPy array of token counts of shape `(batch_size,)`.
    perplexity: The perplexity score of the batch.
    metadata: The sequence of the metadata.
    """

    topic_distribution: numpy.ndarray
    token_counts: numpy.ndarray
    perplexity: float
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class TopicModelingPrediction:
    """
    A prediction for topic modeling task.

    topic_distribution: The topic distribution.
    metadata: The metadata.
    """

    topic_distribution: Sequence[float]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def topic(self) -> int:
        """The index of the top topic"""
        return int(numpy.argmax(self.topic_distribution))
