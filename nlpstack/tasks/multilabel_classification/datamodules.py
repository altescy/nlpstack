from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy

from nlpstack.common import PassThroughPipeline, Pipeline, wrap_iterator
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, MultiLabelField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .types import (
    MultilabelClassificationExample,
    MultilabelClassificationInference,
    MultilabelClassificationPrediction,
)

logger = getLogger(__name__)


class MultilabelClassificationDataModule(
    DataModule[
        MultilabelClassificationExample,
        MultilabelClassificationInference,
        MultilabelClassificationPrediction,
    ]
):
    """
    A data module for multilabel classification.

    Args:
        vocab: The vocabulary.
        tokenizer: The tokenizer.
        token_indexers: The token indexers to index the tokens. Defaults to
            `{"tokens": SingleIdTokenIndexer()}`
        label_namespace: The vocabulary namespace for the labels. Defaults to `"labels"`.
        labels: The set of labels. If not given, the labels will be collected from the
            training dataset. Defaults to `None`.
        preprocessor: The preprocessor to apply to the dataset before tokenization.
            Defaults to `None`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        label_namespace: str = "labels",
        labels: Optional[Sequence[str]] = None,
        preprocessor: Optional[Pipeline[MultilabelClassificationExample, MultilabelClassificationExample]] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace
        self._labels = sorted(set(labels)) if labels else None
        self._preprocessor = preprocessor or PassThroughPipeline()

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def label_namespace(self) -> str:
        return self._label_namespace

    @property
    def labels(self) -> Sequence[str]:
        if self._labels is None:
            return sorted(self._vocab.get_token_to_index(self.label_namespace))
        return self._labels

    def setup(
        self,
        *args: Any,
        dataset: Optional[Sequence[MultilabelClassificationExample]] = None,
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
        dataset: Iterable[MultilabelClassificationExample],
        **kwargs: Any,
    ) -> Iterator[MultilabelClassificationExample]:
        return self._tokenize(self._preprocessor(dataset))

    def _tokenize(
        self, dataset: Iterable[MultilabelClassificationExample]
    ) -> Iterator[MultilabelClassificationExample]:
        """
        Tokenize the dataset and return the tokenized dataset.

        Args:
            dataset: The dataset to tokenize.

        Returns:
            The tokenized dataset.
        """

        if not dataset:
            return iter([])

        def tokenized_document_generator(
            dataset: Iterable[MultilabelClassificationExample],
        ) -> Iterator[MultilabelClassificationExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield MultilabelClassificationExample(
                    text=tokenized_text,
                    labels=example.labels,
                )

        return wrap_iterator(tokenized_document_generator, dataset)

    def _build_vocab(self, dataset: Sequence[MultilabelClassificationExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[Sequence[str]]:
            if self._labels is not None:
                yield self._labels
            else:
                for example in dataset:
                    labels = example.labels
                    assert labels is not None, "Dataset must have labels."
                    yield labels

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

        self.vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: MultilabelClassificationExample) -> Instance:
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

        fields: Dict[str, Field] = {}
        fields["text"] = TextField(text, self.vocab, self._token_indexers)

        if metadata is not None:
            fields["metadata"] = MetadataField(metadata)

        if labels is not None:
            fields["labels"] = MultiLabelField(labels, vocab=self.vocab.get_token_to_index(self.label_namespace))

        return Instance(**fields)

    def build_predictions(
        self, inference: MultilabelClassificationInference
    ) -> Iterator[MultilabelClassificationPrediction]:
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        sorted_indices = inference.probs.argsort(axis=1)[:, ::-1]
        sorted_probs = numpy.take_along_axis(inference.probs, sorted_indices, axis=1)
        for i, (top_indices, top_probs) in enumerate(zip(sorted_indices.tolist(), sorted_probs.tolist())):
            num_labels_to_return = len(top_indices)
            if inference.threshold is not None:
                num_labels_to_return = sum(prob >= inference.threshold for prob in top_probs)
            if inference.top_k is not None:
                num_labels_to_return = min(num_labels_to_return, inference.top_k)
            top_probs = top_probs[:num_labels_to_return]
            top_indices = top_indices[:num_labels_to_return]
            yield MultilabelClassificationPrediction(
                top_probs=top_probs,
                top_labels=[self.vocab.get_token_by_index(self.label_namespace, index) for index in top_indices],
                metadata=inference.metadata[i] if inference.metadata is not None else None,
            )
