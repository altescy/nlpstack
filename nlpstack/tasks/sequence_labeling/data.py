import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy

from nlpstack.data.tokenizers import Token

Decoding = Tuple[Sequence[int], float]


@dataclasses.dataclass
class SequenceLabelingExample:
    text: Union[str, Sequence[Token]]
    labels: Optional[Sequence[str]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class SequenceLabelingInference:
    probs: numpy.ndarray
    mask: Optional[numpy.ndarray] = None
    decodings: Optional[Sequence[Sequence[Decoding]]] = None
    labels: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class SequenceLabelingPrediction:
    top_labels: Sequence[Sequence[str]]
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def labels(self) -> Sequence[str]:
        return self.top_labels[0]
