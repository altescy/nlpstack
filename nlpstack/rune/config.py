import dataclasses
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Optional

from nlpstack.common import FromJsonnet

from .base import Rune
from .types import Example, Prediction


@dataclasses.dataclass
class RuneConfig(FromJsonnet, Generic[Example, Prediction]):
    model: Optional[Rune[Example, Prediction]] = None
    reader: Optional[Callable[[str], Iterator[Example]]] = None
    writer: Optional[Callable[[str, Iterable[Prediction]], None]] = None
    predictor: Optional[Mapping[str, Any]] = None
    evaluator: Optional[Mapping[str, Any]] = None
    train_dataset_filename: Optional[str] = None
    valid_dataset_filename: Optional[str] = None
    test_dataset_filename: Optional[str] = None
