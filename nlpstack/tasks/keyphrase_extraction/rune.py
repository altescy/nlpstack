import math
import re
from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Pattern, Sequence, Tuple, Union

from nlpstack.common import ProgressBar, wrap_iterator
from nlpstack.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nlpstack.rune import Rune

from .types import KeyphraseExtracionExample, KeyphraseExtractionPrediction

logger = getLogger(__name__)


class CValue(
    Rune[
        KeyphraseExtracionExample,
        KeyphraseExtractionPrediction,
        "CValue.SetupParams",
        "CValue.PredictionParams",
        "CValue.EvaluationParams",
    ]
):
    Example = KeyphraseExtracionExample
    Prediction = KeyphraseExtractionPrediction

    class SetupParams(NamedTuple): ...

    class PredictionParams(NamedTuple):
        threshold: Optional[float] = None

    class EvaluationParams(NamedTuple):
        threshold: Optional[float] = None

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

    def _is_acceptable_phrase(self, phrase: Tuple[Token, ...]) -> bool:
        if self._candidate_postag_pattern is None:
            return True
        postags = "".join(f"<{token.postag or '__NULL__'}>" for token in phrase)
        return bool(self._candidate_postag_pattern.match(postags))

    def _iter_candidate_phrases(self, text: Union[str, Sequence[Token]]) -> Iterator[Tuple[Token, ...]]:
        tokens = self._tokenizer.tokenize(text) if isinstance(text, str) else text
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                phrase = tuple(tokens[i : i + n])
                if self._is_acceptable_phrase(phrase):
                    yield phrase

    def train(
        self,
        train_dataset: Sequence[KeyphraseExtracionExample],
        valid_dataset: Optional[Sequence[KeyphraseExtracionExample]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "CValue":
        logger.info("[1/3] Counting n-grams...")
        phrase_frequencies: Dict[Tuple[Token, ...], int] = {}
        for example in ProgressBar(train_dataset, desc="[1/3] Counting n-grams  "):
            text = example.text
            for phrase in self._iter_candidate_phrases(text):
                phrase_frequencies[phrase] = phrase_frequencies.get(phrase, 0) + 1

        logger.info("[2/3] Collecting phrases...")
        longer_phrase_average_frequencies: Dict[Tuple[Token, ...], float] = {}
        longer_phrase_count: Dict[Tuple[Token, ...], int] = {}
        for phrase in ProgressBar(phrase_frequencies, desc="[2/3] Collecting phrases"):
            for n in range(1, len(phrase) - 1):
                for i in range(len(phrase) - n + 1):
                    subphrase = phrase[i : i + n]
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

    def predict(
        self,
        dataset: Iterable[KeyphraseExtracionExample],
        params: Optional["CValue.PredictionParams"] = None,
    ) -> Iterator[KeyphraseExtractionPrediction]:
        params = params or CValue.PredictionParams()
        threshold = params.threshold if params.threshold is not None else self._threshold

        def prediction_iterator(
            dataset: Iterable[KeyphraseExtracionExample],
        ) -> Iterator[KeyphraseExtractionPrediction]:
            if self._extracted_phrases is None:
                raise RuntimeError("CValue has not been trained yet.")

            for example in dataset:
                text = example.text
                phrases: List[str] = []
                scores: List[float] = []
                for phrase in self._iter_candidate_phrases(text):
                    phrase_score = self._extracted_phrases.get(phrase)
                    if phrase_score is None or phrase_score < threshold:
                        continue
                    phrases.append(self._tokenizer.detokenize(phrase))
                    scores.append(phrase_score)

                sorted_indices = sorted(range(len(phrases)), key=lambda i: -scores[i])
                phrases = [phrases[i] for i in sorted_indices[: self._top_k]]
                scores = [scores[i] for i in sorted_indices[: self._top_k]]

                yield KeyphraseExtractionPrediction(
                    phrases=phrases,
                    scores=scores,
                    metadata=example.metadata,
                )

        return wrap_iterator(prediction_iterator, dataset)
