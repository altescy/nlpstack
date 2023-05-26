from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy

from nlpstack.data import Token


@dataclasses.dataclass
class ClassificationExample:
    text: str | Sequence[Token]
    label: str | None = None
    metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class ClassificationInference:
    probs: numpy.ndarray
    labels: numpy.ndarray | None = None
    metadata: list[dict[str, Any]] | None = None


@dataclasses.dataclass
class ClassificationPrediction:
    top_probs: list[float]
    top_labels: list[str]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert len(self.top_probs) == len(self.top_labels)

    @property
    def label(self) -> str:
        return self.top_labels[0]

    @property
    def prob(self) -> float:
        return self.top_probs[0]
