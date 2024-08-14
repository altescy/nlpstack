import itertools
import math
import re
from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Pattern, Sequence, Tuple, Union

import numpy

from nlpstack.common import ProgressBar, wrap_iterator
from nlpstack.data.embeddings import TextEmbedding
from nlpstack.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nlpstack.evaluation import EmptyMetric, Metric, MultiMetrics
from nlpstack.rune import Rune

from .types import KeyphraseExtracionExample, KeyphraseExtractionInference, KeyphraseExtractionPrediction
from .util import iter_candidate_phrases

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

    class _Phrase(NamedTuple):
        text: str
        tokens: Tuple[Token, ...]

        def __hash__(self) -> int:
            return hash(self.text)

        def __eq__(self, other: Any) -> bool:
            return bool(self.text == other.text)

    def __init__(
        self,
        *,
        top_k: int = 10,
        threshold: float = 0.0,
        nc_weight: float = 0.0,
        ngram_range: Tuple[int, int] = (1, 3),
        tokenizer: Optional[Tokenizer] = None,
        candidate_postag_pattern: Optional[Union[str, Pattern]] = None,
        lowercase: bool = False,
        metric: Optional[
            Union[Metric[KeyphraseExtractionInference], Sequence[Metric[KeyphraseExtractionInference]]]
        ] = None,
    ) -> None:
        if metric is None:
            metric = EmptyMetric()
        if isinstance(metric, Sequence):
            metric = MultiMetrics(metric)

        super().__init__()

        self._top_k = top_k
        self._threshold = threshold
        self._nc_weight = nc_weight
        self._ngram_range = ngram_range
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._candidate_postag_pattern = (
            re.compile(candidate_postag_pattern) if candidate_postag_pattern is not None else None
        )
        self._lowercase = lowercase
        self._metric = metric

        self._extracted_phrases: Optional[Mapping[str, float]] = None

    def get_keyphrases(self) -> Mapping[str, float]:
        if self._extracted_phrases is None:
            raise RuntimeError("CValue has not been trained yet.")
        return self._extracted_phrases

    def _build_phrase_from_tokens(self, tokens: Sequence[Token]) -> "CValue._Phrase":
        text = self._tokenizer.detokenize(tokens).strip()
        if self._lowercase:
            text = text.lower()
        return CValue._Phrase(text=text, tokens=tuple(tokens))

    def train(
        self,
        train_dataset: Sequence[KeyphraseExtracionExample],
        valid_dataset: Optional[Sequence[KeyphraseExtracionExample]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "CValue":
        logger.info("[1/3] Counting n-grams...")
        phrase_frequencies: Dict[CValue._Phrase, int] = {}
        for example in ProgressBar(train_dataset, desc="[1/3] Counting n-grams  "):
            tokens = self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text
            for phrase_tokens in iter_candidate_phrases(tokens, self._ngram_range, self._candidate_postag_pattern):
                phrase = self._build_phrase_from_tokens(phrase_tokens)
                phrase_frequencies[phrase] = phrase_frequencies.get(phrase, 0) + 1

        logger.info("[2/3] Collecting phrases...")
        longer_phrase_average_frequencies: Dict[CValue._Phrase, float] = {}
        longer_phrase_count: Dict[CValue._Phrase, int] = {}
        for phrase in ProgressBar(phrase_frequencies, desc="[2/3] Collecting phrases"):
            for n in range(1, len(phrase.tokens) - 1):
                for i in range(len(phrase.tokens) - n + 1):
                    subphrase = self._build_phrase_from_tokens(phrase.tokens[i : i + n])
                    if subphrase in phrase_frequencies:
                        longer_phrase_count[subphrase] = longer_phrase_count.get(subphrase, 0) + 1
                        longer_phrase_average_frequencies[subphrase] = (
                            longer_phrase_average_frequencies.get(subphrase, 0.0) + phrase_frequencies[phrase]
                        )
        longer_phrase_average_frequencies = {
            phrase: freq / longer_phrase_count[phrase] for phrase, freq in longer_phrase_average_frequencies.items()
        }

        logger.info("[3/3] Computing C-values...")
        cvalues: Dict[str, float] = {}
        for phrase, freq in ProgressBar(phrase_frequencies.items(), desc="[3/3] Computing C-values"):
            cvalue = cvalues.get(phrase.text, 0) + math.log2(len(phrase.tokens)) * (
                freq - longer_phrase_average_frequencies.get(phrase, 0.0)
            )
            cvalues[phrase.text] = cvalue

        if self._nc_weight > 0:
            logger.info("Applying NC weighting...")
            term_frequency: Dict[Token, int] = {}
            context_word_frequency: Dict[Tuple[str, Token], int] = {}
            phrase_to_tokens: Dict[str, Sequence[Token]] = {
                phrase.text: [Token(t.surface.strip(), t.postag) for t in phrase.tokens]
                for phrase in phrase_frequencies
            }
            for phrase_text in cvalues:
                for token in set(phrase_to_tokens[phrase_text]):
                    term_frequency[token] = term_frequency.get(token, 0) + 1
                    context_word_frequency[phrase_text, token] = context_word_frequency.get((phrase_text, token), 0) + 1
            for phrase_text, cvalue in cvalues.items():
                weight = sum(
                    term_frequency[t] * context_word_frequency[phrase_text, t] / len(cvalues)
                    for t in phrase_to_tokens[phrase_text]
                )
                cvalues[phrase_text] = (1 - self._nc_weight) * cvalue + self._nc_weight * weight

        if self._threshold > 0:
            cvalues = {phrase: cvalue for phrase, cvalue in cvalues.items() if cvalue >= self._threshold}

        self._extracted_phrases = cvalues

        if valid_dataset is not None:
            logger.info("Start validation...")
            valid_metrics = self.evaluate(valid_dataset)
            logger.info("Validation metrics %s", valid_metrics)

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
                phrases: Dict[str, float] = {}
                tokens = self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text
                for phrase_tokens in iter_candidate_phrases(tokens, self._ngram_range, self._candidate_postag_pattern):
                    phrase = self._build_phrase_from_tokens(phrase_tokens)
                    phrase_score = self._extracted_phrases.get(phrase.text)
                    if phrase_score is None or phrase_score < threshold:
                        continue
                    phrases[phrase.text] = phrase_score

                phrase_and_score_list = sorted(phrases.items(), key=lambda x: x[1], reverse=True)

                yield KeyphraseExtractionPrediction(
                    phrases=[phrase for phrase, _ in phrase_and_score_list],
                    scores=[score for _, score in phrase_and_score_list],
                    metadata=example.metadata,
                )

        return wrap_iterator(prediction_iterator, dataset)

    def evaluate(
        self,
        dataset: Iterable[KeyphraseExtracionExample],
        params: Optional["CValue.EvaluationParams"] = None,
    ) -> Mapping[str, Any]:
        prediction_params = CValue.PredictionParams(params.threshold) if params is not None else None
        dataset, dataset_for_prediction = itertools.tee(dataset)
        predictions = self.predict(dataset_for_prediction, prediction_params)

        self._metric.reset()
        with ProgressBar(dataset, desc="Evaluating") as progress:
            for example, prediction in zip(progress, predictions):
                assert example.phrases is not None
                inference = KeyphraseExtractionInference(
                    pred_phrases=prediction.phrases,
                    gold_phrases=example.phrases,
                )
                self._metric.update(inference)
                progress.set_postfix(**{key: f"{val:.2f}" for key, val in self._metric.compute().items()})

        return self._metric.compute()


class PatternRank(
    Rune[
        KeyphraseExtracionExample,
        KeyphraseExtractionPrediction,
        "PatternRank.SetupParams",
        "PatternRank.PredictionParams",
        "PatternRank.EvaluationParams",
    ]
):
    Example = KeyphraseExtracionExample
    Prediction = KeyphraseExtractionPrediction

    class SetupParams(NamedTuple): ...

    class PredictionParams(NamedTuple): ...

    class EvaluationParams(NamedTuple): ...

    def __init__(
        self,
        *,
        embedder: TextEmbedding,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 3),
        tokenizer: Optional[Tokenizer] = None,
        candidate_postag_pattern: Optional[Union[str, Pattern]] = None,
        metric: Optional[
            Union[Metric[KeyphraseExtractionInference], Sequence[Metric[KeyphraseExtractionInference]]]
        ] = None,
    ) -> None:
        if metric is None:
            metric = EmptyMetric()
        if isinstance(metric, Sequence):
            metric = MultiMetrics(metric)

        super().__init__()

        self._embedder = embedder
        self._top_k = top_k
        self._ngram_range = ngram_range
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._candidate_postag_pattern = (
            re.compile(candidate_postag_pattern) if candidate_postag_pattern is not None else None
        )
        self._metric = metric

    def train(
        self,
        train_dataset: Sequence[KeyphraseExtracionExample],
        valid_dataset: Optional[Sequence[KeyphraseExtracionExample]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "PatternRank":
        return self

    def predict(
        self,
        dataset: Iterable[KeyphraseExtracionExample],
        params: Optional["PatternRank.PredictionParams"] = None,
    ) -> Iterator[KeyphraseExtractionPrediction]:
        def text_iterator(examples: Iterable[KeyphraseExtracionExample]) -> Iterator[str]:
            for example in examples:
                yield example.text if isinstance(example.text, str) else self._tokenizer.detokenize(example.text)

        def tokenized_text_iterator(examples: Iterable[KeyphraseExtracionExample]) -> Iterator[Sequence[Token]]:
            for example in examples:
                yield self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text

        def text_embedding_iterator(examples: Iterable[KeyphraseExtracionExample]) -> Iterator[numpy.ndarray]:
            yield from self._embedder(text_iterator(examples))

        def phrase_iterator(examples: Iterable[KeyphraseExtracionExample]) -> Iterator[List[Tuple[str, numpy.ndarray]]]:
            for tokens in tokenized_text_iterator(examples):
                phrases = list(
                    set(
                        self._tokenizer.detokenize(phrase)
                        for phrase in iter_candidate_phrases(tokens, self._ngram_range, self._candidate_postag_pattern)
                    )
                )
                embeddings = list(self._embedder(phrases))
                yield list(zip(phrases, embeddings))

        def compute_similarity(
            text_embedding: numpy.ndarray,
            phrase_embeddings: numpy.ndarray,
        ) -> List[float]:
            normalized_text_embedding = text_embedding / numpy.linalg.norm(text_embedding)
            normalized_phrase_embeddings = phrase_embeddings / numpy.linalg.norm(phrase_embeddings, axis=1)[:, None]
            return [float(x) for x in (normalized_phrase_embeddings @ normalized_text_embedding)]

        a, b, c = itertools.tee(dataset, 3)
        for example, text_embedding, phrases in zip(a, text_embedding_iterator(b), phrase_iterator(c)):
            if not phrases:
                yield KeyphraseExtractionPrediction(phrases=[], scores=[], metadata=example.metadata)
                continue
            phrase_texts, phrase_embeddings = zip(*phrases)
            phrase_scores = compute_similarity(text_embedding, numpy.array(phrase_embeddings))
            sorted_indices = sorted(range(len(phrases)), key=lambda i: -phrase_scores[i])
            phrase_texts = [phrase_texts[i] for i in sorted_indices][: self._top_k]
            phrase_scores = [phrase_scores[i] for i in sorted_indices][: self._top_k]

            yield KeyphraseExtractionPrediction(
                phrases=phrase_texts,
                scores=phrase_scores,
                metadata=example.metadata,
            )

    def evaluate(
        self,
        dataset: Iterable[KeyphraseExtracionExample],
        params: Optional["PatternRank.EvaluationParams"] = None,
    ) -> Mapping[str, Any]:
        prediction_params = PatternRank.PredictionParams() if params is not None else None
        dataset, dataset_for_prediction = itertools.tee(dataset)
        predictions = self.predict(dataset_for_prediction, prediction_params)

        self._metric.reset()
        with ProgressBar(dataset, desc="Evaluating") as progress:
            for example, prediction in zip(progress, predictions):
                assert example.phrases is not None
                inference = KeyphraseExtractionInference(
                    pred_phrases=prediction.phrases,
                    gold_phrases=example.phrases,
                )
                self._metric.update(inference)
                progress.set_postfix(**{key: f"{val:.2f}" for key, val in self._metric.compute().items()})

        return self._metric.compute()
