from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class ClassificationExample:
    text: str | Sequence[Token]
    label: str | Sequence[str] | None = None


@dataclasses.dataclass
class ClassifierInference:
    probs: numpy.ndarray
    loss: float | None = None
    metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class ClassificationPrediction:
    label: str
    score: float
