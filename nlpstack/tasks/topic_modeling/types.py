import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class TopicModelingExample:
    text: Union[str, Sequence[Token]]
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class TopicModelingInference:
    topic_distribution: numpy.ndarray
    token_counts: numpy.ndarray
    perplexity: float
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class TopicModelingPrediction:
    topic_distribution: Sequence[float]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def topic(self) -> int:
        return int(numpy.argmax(self.topic_distribution))
