from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import PassThroughPipeline, Pipeline
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import DataclassTokenizer, Tokenizer, WhitespaceTokenizer

from .types import RepresentationLearningExample, RepresentationLearningInference, RepresentationLearningPrediction

logger = getLogger(__name__)

RepresentationLearningPreprocessor = Pipeline[RepresentationLearningExample, RepresentationLearningExample, Any]
RepresentationLearningPostprocessor = Pipeline[RepresentationLearningPrediction, RepresentationLearningPrediction, Any]


class RepresentationLearningDataModule(
    DataModule[
        RepresentationLearningExample,
        RepresentationLearningInference,
        RepresentationLearningPrediction,
    ]
):
    """
    A data module for representation learning.

    Args:
        vocab: The vocabulary.
        tokenizer: The tokenizer. Defaults to `WhitespaceTokenizer()`
        token_indexers: The token indexers to index the tokens. Defaults to
            `{"tokens": SingleIdTokenIndexer()}`
        preprocessor: The preprocessor to apply to the dataset before tokenization.
            Defaults to `None`.
        postprocessor: The postprocessor to apply to the predictions after inference.
            Defaults to `None`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        preprocessor: Optional[RepresentationLearningPreprocessor] = None,
        postprocessor: Optional[RepresentationLearningPostprocessor] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._preprocessor = preprocessor or PassThroughPipeline()
        self._postprocessor = postprocessor or PassThroughPipeline()

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def setup(
        self,
        *args: Any,
        dataset: Optional[Sequence[RepresentationLearningExample]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Setup the data module.

        This method builds the vocabulary from the given dataset.

        Args:
            dataset: The dataset to tokenize and build the vocabulary from.
        """

        if dataset:
            self._build_vocab(dataset)

    def preprocess(
        self,
        dataset: Iterable[RepresentationLearningExample],
        **kwargs: Any,
    ) -> Iterator[RepresentationLearningExample]:
        pipeline = self._preprocessor | DataclassTokenizer[RepresentationLearningExample]({"text": self._tokenizer})
        return pipeline(dataset)

    def _build_vocab(self, dataset: Iterable[RepresentationLearningExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

    def build_instance(self, example: RepresentationLearningExample) -> Instance:
        """
        Build an instance from an example.
        If the given `example.text` is a string, it will be tokenized using the tokenizer.

        Args:
            example: The example to build the instance from.
        """

        text = example.text
        metadata = example.metadata

        if isinstance(text, str):
            text = self._tokenizer.tokenize(text)

        fields: Dict[str, Field] = {}
        fields["text"] = TextField(text, self.vocab, self._token_indexers)

        if metadata:
            fields["metadata"] = MetadataField(metadata)

        return Instance(**fields)

    def build_predictions(
        self, inference: RepresentationLearningInference
    ) -> Iterator[RepresentationLearningPrediction]:
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        def prediction_iterator() -> Iterator[RepresentationLearningPrediction]:
            for embedding in inference.embeddings:
                yield RepresentationLearningPrediction(embedding.tolist())

        yield from self._postprocessor(prediction_iterator())
