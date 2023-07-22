import dataclasses
from typing import Any, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class KeyphraseExtracionExample:
    text: Union[str, Sequence[Token]]
    spans: Optional[Set[Tuple[int, int]]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class KeyphraseExtractionInference:
    pred_spans: numpy.ndarray
    gold_spans: Optional[numpy.ndarray] = None
    metadata: Optional[Sequence[Mapping[str, Any]]] = None


@dataclasses.dataclass
class KeyphraseExtractionPrediction:
    spans: Set[Tuple[str, Tuple[int, int]]]
    metadata: Optional[Mapping[str, Any]] = None
