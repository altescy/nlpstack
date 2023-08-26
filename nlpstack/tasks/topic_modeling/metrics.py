import itertools
import math
from collections import defaultdict
from typing import Dict, Mapping, Optional, Tuple

import numpy

from nlpstack.evaluation.metrics import Metric

from .types import TopicModelingInference


class Perplexity(Metric[TopicModelingInference]):
    """
    Perplexity metric for topic modeling.
    """

    def __init__(self) -> None:
        self._total_nll = 0.0
        self._total_count = 0

    def update(self, inference: TopicModelingInference) -> None:
        num_tokens = inference.token_counts.sum()
        nll = numpy.log(inference.perplexity) * num_tokens
        self._total_nll += nll
        self._total_count += num_tokens

    def compute(self) -> Mapping[str, float]:
        return {"perplexity": math.exp(self._total_nll / self._total_count) if self._total_count > 0 else 0.0}

    def reset(self) -> None:
        self._total_nll = 0.0
        self._total_count = 0


class NPMI(Metric[TopicModelingInference]):
    """
    Normalized Pointwise Mutual Information (NPMI) metric.

    https://aclanthology.org/E14-1056/

    Args:
        top_n: The number of words for each topic used for NPMI computation.
            Defaults to `10`.
    """

    def __init__(self, top_n: int = 10) -> None:
        self._top_n = top_n
        self._topic_word_counts: Optional[numpy.ndarray] = None
        self._topic_cooccurrence_counts: Optional[Dict[int, Dict[Tuple[int, int], int]]] = None

    def update(self, inference: TopicModelingInference) -> None:
        topic_distribution = inference.topic_distribution  # Shape: (batch_size, num_topics)
        token_counts = inference.token_counts  # Shape: (batch_size, vocab_size)
        vocab_size = token_counts.shape[1]
        num_topics = topic_distribution.shape[1]
        topics = numpy.argmax(topic_distribution, axis=1)  # Shape: (batch_size,)

        if self._topic_word_counts is None:
            self._topic_word_counts = numpy.zeros((num_topics, vocab_size), dtype=float)
        if self._topic_cooccurrence_counts is None:
            self._topic_cooccurrence_counts = defaultdict(lambda: defaultdict(int))

        for topic, token_count in zip(topics, token_counts):
            self._topic_word_counts[topic] += token_count
            nonzero_indices = token_count.nonzero()[0].tolist()
            for w1, w2 in itertools.combinations(nonzero_indices, 2):
                self._topic_cooccurrence_counts[topic][(w1, w2)] += 1

    def compute(self) -> Mapping[str, float]:
        if self._topic_word_counts is None or self._topic_cooccurrence_counts is None:
            return {"npmi": 0.0}
        num_topics, vocab_size = self._topic_word_counts.shape
        top_topic_words = [
            indices[word_count[indices] > 0].tolist()
            for word_count, indices in zip(
                self._topic_word_counts,
                numpy.argsort(self._topic_word_counts, axis=1)[:, -self._top_n :][:, ::-1],
            )
        ]
        coherence = 0.0
        for topic in range(num_topics):
            top_words = top_topic_words[topic]
            total_words = self._topic_word_counts[topic].sum()
            if total_words == 0:
                continue
            npmi = 0.0
            num_combinations = 0
            for w1, w2 in itertools.combinations(sorted(top_words), 2):
                num_combinations += 1
                if (w1, w2) not in self._topic_cooccurrence_counts[topic]:
                    continue
                w1_count = self._topic_word_counts[topic, w1]
                w2_count = self._topic_word_counts[topic, w2]
                pair_count = self._topic_cooccurrence_counts[topic][(w1, w2)]
                pmi = math.log(total_words * pair_count / (w1_count * w2_count), 10)
                npmi += pmi / -math.log(pair_count / total_words, 10)
            coherence += npmi / num_combinations
        coherence /= num_topics
        return {"npmi": coherence}

    def reset(self) -> None:
        self._topic_word_counts = None
        self._topic_cooccurrence_counts = None
