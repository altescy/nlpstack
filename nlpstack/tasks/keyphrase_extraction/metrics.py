from typing import Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy

from nlpstack.evaluation import Metric

from .types import KeyphraseExtractionInference

FBetaAverage = Literal["micro", "macro"]


class FBeta(Metric[KeyphraseExtractionInference]):
    def __init__(
        self,
        beta: float = 1.0,
        average: Union[FBetaAverage, Sequence[FBetaAverage]] = "macro",
        topk: Optional[int] = None,
    ) -> None:
        self._beta = beta
        self._average = average
        self._topk = topk
        self._true_positives: List[int] = []
        self._false_positives: List[int] = []
        self._false_negatives: List[int] = []

    def update(self, inference: KeyphraseExtractionInference) -> None:
        assert inference.gold_phrases is not None

        sorted_indices = sorted(
            range(len(inference.pred_phrases)),
            key=lambda i: -inference.pred_scores[i] if inference.pred_scores else 0,
        )
        pred_phrases = set(inference.pred_phrases[i] for i in sorted_indices[: self._topk])
        gold_phrases = inference.gold_phrases or set()

        self._true_positives.append(len(pred_phrases & gold_phrases))
        self._false_positives.append(len(pred_phrases - gold_phrases))
        self._false_negatives.append(len(gold_phrases - pred_phrases))

    def compute(self) -> Mapping[str, float]:
        true_positives = numpy.array(self._true_positives)
        false_positives = numpy.array(self._false_positives)
        false_negatives = numpy.array(self._false_negatives)

        metrics: Dict[str, float] = {}
        if "macro" in self._average:
            precision = numpy.mean(true_positives / (true_positives + false_positives + 1e-13))
            recall = numpy.mean(true_positives / (true_positives + false_negatives + 1e-13))
            fbeta = (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall + 1e-13)
            metrics["macro_precision"] = precision
            metrics["macro_recall"] = recall
            metrics["macro_fbeta"] = fbeta
        if "micro" in self._average:
            precision = numpy.sum(true_positives) / (true_positives.sum() + false_positives.sum())
            recall = numpy.sum(true_positives) / (true_positives.sum() + false_negatives.sum())
            fbeta = (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall)
            metrics["micro_precision"] = precision
            metrics["micro_recall"] = recall
            metrics["micro_fbeta"] = fbeta

        return metrics

    def reset(self) -> None:
        self._true_positives = []
        self._false_positives = []
        self._false_negatives = []
