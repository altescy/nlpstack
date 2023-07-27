import math
import re
from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Pattern, Sequence, Tuple, Union

from nlpstack.common import ProgressBar
from nlpstack.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nlpstack.rune import Rune

from .types import KeyphraseExtracionExample, KeyphraseExtractionPrediction

logger = getLogger(__name__)


class CValue(Rune[KeyphraseExtracionExample, KeyphraseExtractionPrediction]):
    def __init__(
        self,
        *,
        top_k: int = 10,
        threshold: float = 0.0,
        ngram_range: Tuple[int, int] = (1, 3),
        tokenizer: Optional[Tokenizer] = None,
        candidate_postag_pattern: Optional[Union[str, Pattern]] = None,
    ) -> None:
        self._top_k = top_k
        self._threshold = threshold
        self._ngram_range = ngram_range
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._candidate_postag_pattern = (
            re.compile(candidate_postag_pattern) if candidate_postag_pattern is not None else None
        )

        self._extracted_phrases: Optional[Mapping[Tuple[Token, ...], float]] = None

    def get_cvalues(self) -> Mapping[Tuple[Token, ...], float]:
        if self._extracted_phrases is None:
            raise RuntimeError("CValue has not been trained yet.")
        return self._extracted_phrases

    def train(
        self,
        train_dataset: Sequence[KeyphraseExtracionExample],
        valid_dataset: Optional[Sequence[KeyphraseExtracionExample]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "CValue":
        def count_ngrams(tokens: Sequence[Token]) -> Mapping[Tuple[Token, ...], int]:
            ngrams: Dict[Tuple[Token, ...], int] = {}
            for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    ngrams[ngram] = ngrams.get(ngram, 0) + 1
            return ngrams

        def is_acceptable_phrase(phrase: Tuple[Token, ...]) -> bool:
            if self._candidate_postag_pattern is None:
                return True
            postag_pattern = "".join(token.postag or "NULL" for token in phrase)
            return bool(self._candidate_postag_pattern.match(postag_pattern))

        logger.info("[1/3] Counting n-grams...")
        phrase_frequencies: Dict[Tuple[Token, ...], int] = {}
        for example in ProgressBar(train_dataset, desc="[1/3] Counting n-grams  "):
            text = example.text
            tokens = self._tokenizer.tokenize(text) if isinstance(text, str) else text
            for ngram, freq in count_ngrams(tokens).items():
                if is_acceptable_phrase(ngram):
                    phrase_frequencies[ngram] = phrase_frequencies.get(ngram, 0) + freq

        logger.info("[2/3] Collecting phrases...")
        longer_phrase_average_frequencies: Dict[Tuple[Token, ...], float] = {}
        longer_phrase_count: Dict[Tuple[Token, ...], int] = {}
        for phrase in ProgressBar(phrase_frequencies, desc="[2/3] Collecting phrases"):
            for start in range(0, len(phrase) - self._ngram_range[0]):
                for end in range(start + self._ngram_range[0], len(phrase) + 1):
                    if start == 0 and end == len(phrase):
                        continue
                    if end - start < self._ngram_range[0] or end - start > self._ngram_range[1]:
                        continue
                    subphrase = phrase[start:end]
                    if subphrase in phrase_frequencies:
                        longer_phrase_count[subphrase] = longer_phrase_count.get(subphrase, 0) + 1
                        longer_phrase_average_frequencies[subphrase] = (
                            longer_phrase_average_frequencies.get(subphrase, 0.0) + phrase_frequencies[phrase]
                        )
        longer_phrase_average_frequencies = {
            phrase: freq / longer_phrase_count[phrase] for phrase, freq in longer_phrase_average_frequencies.items()
        }

        logger.info("[3/3] Computing C-values...")
        cvalues: Dict[Tuple[Token, ...], float] = {}
        for phrase, freq in ProgressBar(phrase_frequencies.items(), desc="[3/3] Computing C-values"):
            cvalue = math.log2(len(phrase)) * (freq - longer_phrase_average_frequencies.get(phrase, 0.0))
            if cvalue > self._threshold:
                cvalues[phrase] = cvalue

        self._extracted_phrases = cvalues

        logger.info("Done.")
        return self
