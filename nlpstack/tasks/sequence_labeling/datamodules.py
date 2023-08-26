from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy

from nlpstack.common import FileBackendSequence, ProgressBar
from nlpstack.data import DataModule, Instance, Vocabulary
from nlpstack.data.fields import Field, MetadataField, SequenceLabelField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, Token, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .types import SequenceLabelingExample, SequenceLabelingInference, SequenceLabelingPrediction

logger = getLogger(__name__)


class SequenceLabelingDataModule(
    DataModule[
        SequenceLabelingExample,
        SequenceLabelingInference,
        SequenceLabelingPrediction,
    ]
):
    """
    A data module for classification.

    Args:
        vocab: The vocabulary.
        tokenizer: The tokenizer. Defaults to `WhitespaceTokenizer()`
        token_indexers: The token indexers to index the tokens. Defaults to
            `{"tokens": SingleIdTokenIndexer()}`
        label_namespace: The vocabulary namespace for the labels. Defaults to `"labels"`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        label_namespace: str = "labels",
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def label_namespace(self) -> str:
        return self._label_namespace

    def setup(
        self,
        *args: Any,
        dataset: Optional[Sequence[SequenceLabelingExample]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Setup the data module.

        This method tokenizes the dataset and builds the vocabulary.

        Args:
            dataset: The dataset to tokenize and build the vocabulary from.
        """

        if dataset:
            logger.info("Tokenizing dataset and building vocabulary...")
            dataset = self.tokenize(ProgressBar(dataset, desc="Tokenizing dataset"))
            self._build_vocab(dataset)

    def tokenize(self, dataset: Iterable[SequenceLabelingExample]) -> Sequence[SequenceLabelingExample]:
        """
        Tokenize the dataset and return the tokenized dataset.

        Args:
            dataset: The dataset to tokenize.

        Returns:
            The tokenized dataset.
        """

        if not dataset:
            return []

        def tokenized_document_generator() -> Iterator[SequenceLabelingExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield SequenceLabelingExample(
                    text=tokenized_text,
                    labels=example.labels,
                )

        return FileBackendSequence.from_iterable(tokenized_document_generator())

    def _build_vocab(self, dataset: Sequence[SequenceLabelingExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[Sequence[str]]:
            for example in dataset:
                assert example.labels is not None, "Dataset must have labels."
                yield example.labels

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self._vocab, text_iterator())

        self._vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: SequenceLabelingExample) -> Instance:
        """
        Build an instance from an example.
        If the given `example.text` is a string, it will be tokenized using the tokenizer.

        Args:
            example: The example to build the instance from.
        """

        text = example.text
        labels = example.labels
        metadata = example.metadata

        if isinstance(text, str):
            text = self._tokenizer.tokenize(text)

        text_field = TextField(text, self._vocab, self._token_indexers)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field

        if labels is not None:
            fields["labels"] = SequenceLabelField(
                labels,
                text_field,
                vocab=self.vocab.get_token_to_index(self._label_namespace),
            )

        fields["metadata"] = MetadataField(
            {
                **(metadata or {}),
                "__nlpstack_metadata__": {
                    "metadata_is_none": metadata is None,
                    "raw_tokens": [token.surface for token in text],
                },
            }
        )

        return Instance(**fields)

    def build_predictions(self, inference: SequenceLabelingInference) -> Iterator[SequenceLabelingPrediction]:
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        assert inference.metadata is not None, "Inference must have metadata."

        top_label_indices: Sequence[Sequence[Sequence[int]]]
        if inference.decodings is None:
            mask = inference.mask if inference.mask is not None else numpy.ones(inference.probs.shape[:-1], dtype=bool)
            lengths = mask.sum(axis=1).tolist()
            top_label_indices = [
                [label_indices[:length]]
                for label_indices, length in zip(numpy.argmax(inference.probs, axis=-1).tolist(), lengths)
            ]
        else:
            top_label_indices = [[label_indices for label_indices, _ in decodings] for decodings in inference.decodings]

        batch_size = len(top_label_indices)
        for batch_index in range(batch_size):
            top_labels = [
                [self._vocab.get_token_by_index(self._label_namespace, label_index) for label_index in label_indices]
                for label_indices in top_label_indices[batch_index]
            ]
            _metadata = dict(inference.metadata[batch_index])
            _nlpstack_metadata = _metadata.pop("__nlpstack_metadata__")
            tokens = _nlpstack_metadata["raw_tokens"]
            metadata = None if _nlpstack_metadata["metadata_is_none"] else _metadata
            yield SequenceLabelingPrediction(tokens=tokens, top_labels=top_labels, metadata=metadata)

    def read_dataset(self, dataset: Iterable[SequenceLabelingExample], **kwargs: Any) -> Iterator[Instance]:
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
