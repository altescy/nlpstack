import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class MultilabelClassificationExample:
    text: Union[str, Sequence[Token]]
    labels: Optional[Sequence[str]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class MultilabelClassificationInference:
    probs: numpy.ndarray
    threshold: float
    labels: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class MultilabelClassificationPrediction:
    top_probs: Sequence[float]
    top_labels: Sequence[str]
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        assert len(self.top_probs) == len(self.top_labels)
