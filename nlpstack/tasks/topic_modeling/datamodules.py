from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import FileBackendSequence, ProgressBar
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.data.util import iter_with_callback

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
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

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
            with ProgressBar[int](len(dataset) * 2) as progress:
                progress.set_description("Tokenizing dataset")
                dataset = self.tokenize(iter_with_callback(dataset, lambda _: progress.update()))
                progress.set_description("Building vocab    ")
                self._build_vocab(iter_with_callback(dataset, lambda _: progress.update()))

    def tokenize(self, dataset: Iterable[TopicModelingExample]) -> Sequence[TopicModelingExample]:
        """
        Tokenize the dataset and return the tokenized dataset.

        Args:
            dataset: The dataset to tokenize.

        Returns:
            The tokenized dataset.
        """

        if not dataset:
            return []

        def tokenized_document_generator() -> Iterator[TopicModelingExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield TopicModelingExample(text=tokenized_text)

        return FileBackendSequence.from_iterable(tokenized_document_generator())

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

    def read_dataset(self, dataset: Iterable[TopicModelingExample], **kwargs: Any) -> Iterator[Instance]:
        """
        Read the dataset and return a generator of instances.

        Args:
            dataset: The dataset to read.

        Returns:
            A generator of instances.
        """

        logger.info("Building instances...")
        for example in ProgressBar(dataset, desc="Building instances"):
            yield self.build_instance(example)
