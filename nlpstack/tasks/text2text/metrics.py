import math
from typing import Any, Mapping, Optional, Sequence, Set

import numpy

from nlpstack.common import BLEU as _BLEU
from nlpstack.evaluation import Metric

from .datamodules import Text2TextDataModule
from .types import Text2TextInference

Text2TextMetric = Metric[Text2TextInference]


class Perplexity(Metric[Text2TextInference]):
    def __init__(self) -> None:
        self._total_nnl = 0.0
        self._total_count = 0

    def update(self, inference: Text2TextInference) -> None:
        if inference.perplexity is None:
            raise ValueError("Perplexity is not computed.")
        if inference.gold_mask is None or inference.gold_token_ids is None:
            raise ValueError("Gold tokens are not provided.")

        num_tokens = inference.gold_mask.sum()
        self._total_nnl += math.log(inference.perplexity) * num_tokens
        self._total_count += num_tokens

    def compute(self) -> Mapping[str, float]:
        return {"perplexity": math.exp(self._total_nnl / self._total_count) if self._total_count > 0 else float("inf")}

    def reset(self) -> None:
        self._total_nnl = 0.0
        self._total_count = 0


class BLEU(Metric[Text2TextInference]):
    def __init__(
        self,
        ngram_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        exclude_tokens: Optional[Set[str]] = None,
        namespace: str = "tokens",
    ) -> None:
        self._ngram_weights = ngram_weights
        self._exclude_tokens = exclude_tokens
        self._namespace = namespace
        self.__bleu: Optional[_BLEU] = None

    @property
    def _bleu(self) -> _BLEU:
        if self.__bleu is None:
            raise ValueError("BLEU is not setup.")
        return self.__bleu

    def setup(self, *args: Any, datamodule: Text2TextDataModule, **kwargs: Any) -> None:
        exclude_indices: Optional[Set[int]] = None
        if self._exclude_tokens is not None:
            exclude_indices = {
                datamodule.vocab.get_index_by_token(self._namespace, token) for token in self._exclude_tokens
            }
        self.__bleu = _BLEU(self._ngram_weights, exclude_indices)

    def update(self, inference: Text2TextInference) -> None:
        assert inference.gold_token_ids is not None

        batch_size = len(inference.pred_token_ids)
        pred_token_ids = inference.pred_token_ids[:, 0]  # select top-1 prediction
        gold_token_ids = inference.gold_token_ids
        pred_mask = (
            numpy.ones(pred_token_ids.shape, dtype=bool) if inference.pred_mask is None else inference.pred_mask[:, 0]
        )
        gold_mask = numpy.ones(gold_token_ids.shape, dtype=bool) if inference.gold_mask is None else inference.gold_mask

        candidates = [
            [token for token, mask in zip(pred_token_ids[i], pred_mask[i]) if mask] for i in range(batch_size)
        ]
        references = [
            [token for token, mask in zip(gold_token_ids[i], gold_mask[i]) if mask] for i in range(batch_size)
        ]

        self._bleu.update(candidates, references)

    def compute(self) -> Mapping[str, float]:
        return {"bleu": self.__bleu.compute() if self.__bleu else 0.0}

    def reset(self) -> None:
        self._bleu.reset()
