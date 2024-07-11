from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import PassThroughPipeline, Pipeline, wrap_iterator
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .types import TopicModelingExample, TopicModelingInference, TopicModelingPrediction

logger = getLogger(__name__)


class TopicModelingDataModule(
    DataModule[
        TopicModelingExample,
        TopicModelingInference,
        TopicModelingPrediction,
    ]
):
    """
    A data module for topic modeling.

    Args:
        vocab: The vocabulary.
        tokenizer: The tokenizer. Defaults to `WhitespaceTokenizer()`.
        token_indexers: The token indexers to index the tokens. Defaults to
            `{"tokens": SingleIdTokenIndexer()}`.
        preprocessor: The preprocessor to apply to the dataset before tokenization.
            Defaults to `None`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        preprocessor: Optional[Pipeline[TopicModelingExample, TopicModelingExample]] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._preprocessor = preprocessor or PassThroughPipeline()

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def setup(
        self,
        *args: Any,
        dataset: Optional[Sequence[TopicModelingExample]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Setup the data module.

        This method tokenizes the dataset and builds the vocabulary.

        Args:
            dataset: The dataset to tokenize and build the vocabulary from.
        """

        if dataset:
            self._build_vocab(dataset)

    def preprocess(self, dataset: Iterable[TopicModelingExample], **kwargs: Any) -> Iterator[TopicModelingExample]:
        return wrap_iterator(self._tokenize, self._preprocessor(dataset))

    def _tokenize(self, dataset: Iterable[TopicModelingExample]) -> Iterator[TopicModelingExample]:
        """
        Tokenize the dataset and return the tokenized dataset.

        Args:
            dataset: The dataset to tokenize.

        Returns:
            The tokenized dataset.
        """

        for example in dataset:
            if isinstance(example.text, str):
                tokenized_text = self._tokenizer.tokenize(example.text)
            else:
                tokenized_text = list(example.text)
            yield TopicModelingExample(text=tokenized_text)

    def _build_vocab(self, dataset: Iterable[TopicModelingExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

    def build_instance(self, example: TopicModelingExample) -> Instance:
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

    def build_predictions(self, inference: TopicModelingInference) -> Iterator[TopicModelingPrediction]:
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        for topic_distribution in inference.topic_distribution:
            yield TopicModelingPrediction(topic_distribution.tolist())
