import math
from typing import Mapping

from nlpstack.evaluation import Metric

from .types import CausalLanguageModelingInference

CausalLanguageModelingMetric = Metric[CausalLanguageModelingInference]


class Perplexity(Metric[CausalLanguageModelingInference]):
    """
    The perplexity metric for causal language modeling.
    """

    def __init__(self) -> None:
        self._total_nnl = 0.0
        self._total_count = 0

    def update(self, inference: CausalLanguageModelingInference) -> None:
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
