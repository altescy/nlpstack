from collections import defaultdict
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Set, Tuple

import numpy

from nlpstack.evaluation import Metric

from .data import SequenceLabelingInference
from .datamodules import SequenceLabelingDataModule
from .util import bio_tags_to_spans, bioul_tags_to_spans, bmes_tags_to_spans, iob1_tags_to_spans

LabelEncoding = Literal["BIO", "IOB1", "BIOUL", "BMES"]


class SequenceLabelingMetric(Metric[SequenceLabelingInference]):
    """Metric for sequence labeling tasks."""


class TokenBasedAccuracy(SequenceLabelingMetric):
    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, inference: SequenceLabelingInference) -> None:
        assert inference.labels is not None

        if inference.mask is None:
            mask = numpy.ones(inference.labels.shape, dtype=bool)
        else:
            mask = inference.mask.astype(bool)

        if inference.decodings is None:
            pred = inference.probs.argmax(axis=-1)[mask]
        else:
            pred = numpy.array([label_index for decodings in inference.decodings for label_index in decodings[0][0]])

        gold = inference.labels[mask]

        self._correct += (pred == gold).sum()
        self._total += len(gold)

    def compute(self) -> Dict[str, float]:
        return {"token_accuracy": self._correct / self._total if self._total > 0 else 0.0}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class SpanBasedF1(SequenceLabelingMetric):
    def __init__(
        self,
        label_encoding: LabelEncoding = "BIO",
        label_namespace: str = "labels",
        ignore_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self._label_encoding = label_encoding
        self._label_namespace = label_namespace
        self._ignore_classes = ignore_classes or []
        self._label_vocab: Optional[Mapping[int, str]] = None

        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def setup(self, *args: Any, datamodule: SequenceLabelingDataModule, **kwargs: Any) -> None:
        self._label_vocab = datamodule.vocab.get_index_to_token(self._label_namespace)

    def update(self, inference: SequenceLabelingInference) -> None:
        assert inference.labels is not None
        assert self._label_vocab is not None

        if inference.mask is None:
            mask = numpy.ones(inference.labels.shape, dtype=bool)
        else:
            mask = inference.mask.astype(bool)

        lengths = mask.sum(axis=-1)

        if inference.decodings is None:
            preds = [
                [self._label_vocab[i] for i in label_indices[:length]]
                for label_indices, length in zip(inference.probs.argmax(axis=-1).tolist(), lengths.tolist())
            ]
        else:
            preds = [[self._label_vocab[i] for i in decodings[0][0]] for decodings in inference.decodings]

        golds = [
            [self._label_vocab[i] for i in label_indices[:length]]
            for label_indices, length in zip(inference.labels.tolist(), lengths.tolist())
        ]

        if self._label_encoding == "BIO":
            tags_to_spans_function = bio_tags_to_spans
        elif self._label_encoding == "IOB1":
            tags_to_spans_function = iob1_tags_to_spans
        elif self._label_encoding == "BIOUL":
            tags_to_spans_function = bioul_tags_to_spans
        elif self._label_encoding == "BMES":
            tags_to_spans_function = bmes_tags_to_spans
        else:
            raise ValueError(f"Unexpected label encoding scheme '{self._label_encoding}'")

        for pred, gold in zip(preds, golds):
            pred_spans = tags_to_spans_function(pred, self._ignore_classes)
            gold_spans = tags_to_spans_function(gold, self._ignore_classes)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def compute(self) -> Dict[str, float]:
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "precision" + "_" + tag
            recall_key = "recall" + "_" + tag
            f1_key = "f1" + "_" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["precision_overall"] = precision
        all_metrics["recall_overall"] = recall
        all_metrics["f1_overall"] = f1_measure
        return all_metrics

    def _compute_metrics(
        self, true_positives: int, false_positives: int, false_negatives: int
    ) -> Tuple[float, float, float]:
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset(self) -> None:
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
