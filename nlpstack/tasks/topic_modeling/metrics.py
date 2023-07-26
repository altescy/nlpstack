import itertools
from collections import defaultdict
from typing import Dict, Mapping, Optional, Tuple

import numpy

from nlpstack.evaluation.metrics import Metric

from .types import TopicModelingInference


class Perplexity(Metric[TopicModelingInference]):
    def __init__(self) -> None:
        self._total_nll = 0.0
        self._total_count = 0

    def update(self, inference: TopicModelingInference) -> None:
        num_tokens = inference.token_counts.sum()
        nll = numpy.log(inference.perplexity) * num_tokens
        self._total_nll += nll
        self._total_count += num_tokens

    def compute(self) -> Mapping[str, float]:
        return {"perplexity": self._total_perplexity / self._total_count if self._total_count > 0 else 0.0}

    def reset(self) -> None:
        self._total_perplexity = 0.0
        self._total_count = 0


class NPMI(Metric[TopicModelingInference]):
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
            for pair in itertools.combinations(nonzero_indices, 2):
                self._topic_cooccurrence_counts[topic][pair] += 1

    def compute(self) -> Mapping[str, float]:
        if self._topic_word_counts is None or self._topic_cooccurrence_counts is None:
            return {"npmi": 0.0}
        num_topics, vocab_size = self._topic_word_counts.shape
        top_topic_words = [
            indices[word_count[indices] >= 0].tolist()
            for word_count, indices in zip(
                self._topic_word_counts,
                numpy.argsort(self._topic_word_counts, axis=-1)[:, : self._top_n][::-1],
            )
        ]
        coherence = 0.0
        for topic in range(num_topics):
            top_words = top_topic_words[topic]
            total_words = self._topic_word_counts[topic].sum()
            total_pairs = sum(self._topic_cooccurrence_counts[topic].values())
            if 0 in (total_words, total_pairs):
                continue
            for w1, w2 in itertools.combinations(top_words, 2):
                if (w1, w2) not in self._topic_cooccurrence_counts[topic]:
                    continue
                pair_prob = self._topic_cooccurrence_counts[topic][(w1, w2)] / total_pairs
                w1_prob = self._topic_word_counts[topic, w1] / total_words
                w2_prob = self._topic_word_counts[topic, w2] / total_words
                pmi = numpy.log(pair_prob / ((w1_prob * w2_prob)))
                coherence += pmi / (-numpy.log(pair_prob))
        coherence /= num_topics
        return {"npmi": coherence}

    def reset(self) -> None:
        self._topic_word_counts = None
        self._topic_cooccurrence_counts = None
