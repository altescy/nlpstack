import math
from collections import defaultdict
from typing import Dict, Hashable, Iterable, Optional, Sequence, Set, Tuple, TypeVar

Token = TypeVar("Token", bound=Hashable)


class BLEU:
    """
    Calculate BLEU score.

    This class can be used to calculate BLEU score between two sequences of
    tokens. The implementation is based on the following paper:

    https://api.semanticscholar.org/CorpusID:11080756?utm_source=wikipedia

    Args:
        ngram_weights: A list of ngram weights.
    """

    def __init__(
        self,
        ngram_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        exclude_tokens: Optional[Set[Token]] = None,
    ) -> None:
        self._ngram_weights = ngram_weights
        self._exclude_tokens = exclude_tokens
        self._precision_matches: Dict[int, int] = defaultdict(int)
        self._precision_totals: Dict[int, int] = defaultdict(int)
        self._candidate_length = 0
        self._reference_length = 0

    def update(
        self,
        candidates: Sequence[Sequence[Token]],
        references: Sequence[Sequence[Token]],
        ngram_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._compute_modified_ngram_precision_counts(
                candidates, references, ngram_size
            )
            self._precision_matches[ngram_size] += precision_matches
            self._precision_totals[ngram_size] += precision_totals
        for candidate, reference in zip(candidates, references):
            self._candidate_length += sum(
                not self._exclude_tokens or token not in self._exclude_tokens for token in candidate
            )
            self._reference_length += sum(
                not self._exclude_tokens or token not in self._exclude_tokens for token in reference
            )

    def compute(self) -> float:
        brevity_penalty = self._compute_brevity_penalty()
        ngram_scores = (
            weight * (math.log(self._precision_matches[n] + 1e-13) - math.log(self._precision_totals[n] + 1e-13))
            for n, weight in enumerate(self._ngram_weights, start=1)
        )
        return brevity_penalty * math.exp(sum(ngram_scores))

    def reset(self) -> None:
        self._precision_matches.clear()
        self._precision_totals.clear()
        self._candidate_length = 0
        self._reference_length = 0

    def _count_ngrams(self, tokens: Sequence[Token], ngram_size: int) -> Dict[Tuple[Token, ...], int]:
        ngram_counts: Dict[Tuple[Token, ...], int] = defaultdict(int)
        if ngram_size > len(tokens):
            return ngram_counts
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[i : i + ngram_size])
            if self._exclude_tokens and any(token in self._exclude_tokens for token in ngram):
                continue
            ngram_counts[ngram] += 1
        return ngram_counts

    def _compute_modified_ngram_precision_counts(
        self,
        candidates: Iterable[Sequence[Token]],
        references: Iterable[Sequence[Token]],
        ngram_size: int,
    ) -> Tuple[int, int]:
        clipped_matches = 0
        total_predicted = 0
        for candidate, reference in zip(candidates, references):
            candidate_gram_counts = self._count_ngrams(candidate, ngram_size)
            reference_gram_counts = self._count_ngrams(reference, ngram_size)
            for ngram, count in candidate_gram_counts.items():
                clipped_matches += min(count, reference_gram_counts[ngram])
                total_predicted += count
        return clipped_matches, total_predicted

    def _compute_brevity_penalty(self) -> float:
        if self._candidate_length > self._reference_length:
            return 1.0
        if self._candidate_length == 0 or self._reference_length == 0:
            return 0.0
        return math.exp(1.0 - self._reference_length / self._candidate_length)
