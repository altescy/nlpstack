import dataclasses
from typing import Any, Dict, Optional, Sequence, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class ClassificationExample:
    text: Union[str, Sequence[Token]]
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ClassificationInference:
    probs: numpy.ndarray
    labels: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Dict[str, Any]]] = None


@dataclasses.dataclass
class ClassificationPrediction:
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
