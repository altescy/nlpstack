from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import FileBackendSequence, ProgressBar
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

    def _tokenize(self, dataset: Iterable[TopicModelingExample]) -> Sequence[TopicModelingExample]:
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

    def _build_vocab(self, dataset: Sequence[TopicModelingExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

    def build_instance(self, example: TopicModelingExample) -> Instance:
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
        for topic_distribution in inference.topic_distribution:
            yield TopicModelingPrediction(topic_distribution.tolist())

    def read_dataset(
        self,
        dataset: Iterable[TopicModelingExample],
        is_training: bool = False,
        **kwargs: Any,
    ) -> Iterator[Instance]:
        if is_training:
            logger.info("Tokenizing dataset and building vocabulary...")
            dataset = self._tokenize(ProgressBar(dataset, desc="Tokenizing dataset"))
            self._build_vocab(dataset)

        logger.info("Building instances...")
        for example in ProgressBar(dataset, desc="Building instances"):
            yield self.build_instance(example)
