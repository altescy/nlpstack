import dataclasses
from typing import Any, List, Mapping, Optional, Sequence, Set, Union

from nlpstack.data import Token


@dataclasses.dataclass
class KeyphraseExtracionExample:
    text: Union[str, Sequence[Token]]
    phrases: Optional[Set[str]] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class KeyphraseExtractionInference:
    pred_phrases: List[str]
    gold_phrases: Optional[Set[str]] = None
    pred_scores: Optional[List[float]] = None


@dataclasses.dataclass
class KeyphraseExtractionPrediction:
    phrases: List[str]
    scores: Optional[List[float]] = None
    metadata: Optional[Mapping[str, Any]] = None
