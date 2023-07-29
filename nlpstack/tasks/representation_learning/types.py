import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class RepresentationLearningExample:
    text: Union[str, Sequence[Token]]
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class RepresentationLearningInference:
    embeddings: numpy.ndarray
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class RepresentationLearningPrediction:
    embedding: numpy.ndarray
    metadata: Optional[Mapping[str, Any]] = None
