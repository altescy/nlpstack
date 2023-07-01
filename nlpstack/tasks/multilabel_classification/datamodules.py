from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy

from nlpstack.common import ProgressBar
from nlpstack.data import DataModule, Dataset, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TensorField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .data import MultilabelClassificationExample, MultilabelClassificationInference, MultilabelClassificationPrediction

logger = getLogger(__name__)


class MultilabelClassificationDataModule(
    DataModule[
        MultilabelClassificationExample,
        MultilabelClassificationInference,
        MultilabelClassificationPrediction,
    ]
):
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

    def _tokenize(
        self, dataset: Iterable[MultilabelClassificationExample]
    ) -> Sequence[MultilabelClassificationExample]:
        if not dataset:
            return []

        def tokenized_document_generator() -> Iterator[MultilabelClassificationExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield MultilabelClassificationExample(
                    text=tokenized_text,
                    labels=example.labels,
                )

        return Dataset.from_iterable(tokenized_document_generator())

    def _build_vocab(self, dataset: Sequence[MultilabelClassificationExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[Sequence[str]]:
            for example in dataset:
                labels = example.labels
                assert labels is not None, "Dataset must have labels."
                yield labels

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

        self.vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: MultilabelClassificationExample) -> Instance:
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
            label_indices = [self.vocab.get_index_by_token(self.label_namespace, label) for label in labels]
            fields["labels"] = TensorField(
                numpy.bincount(label_indices, minlength=self.vocab.get_vocab_size(self.label_namespace)).astype(int)
            )

        return Instance(**fields)

    def build_predictions(
        self, inference: MultilabelClassificationInference
    ) -> Iterator[MultilabelClassificationPrediction]:
        sorted_indices = inference.probs.argsort(axis=1)[:, ::-1]
        sorted_probs = numpy.take_along_axis(inference.probs, sorted_indices, axis=1)
        for i, (top_indices, top_probs) in enumerate(zip(sorted_indices.tolist(), sorted_probs.tolist())):
            num_labels_to_return = sum(p >= inference.threshold for p in top_probs)
            if inference.top_k is not None:
                num_labels_to_return = min(num_labels_to_return, inference.top_k)
            top_probs = top_probs[:num_labels_to_return]
            top_indices = top_indices[:num_labels_to_return]
            yield MultilabelClassificationPrediction(
                top_probs=top_probs,
                top_labels=[self.vocab.get_token_by_index(self.label_namespace, index) for index in top_indices],
                metadata=inference.metadata[i] if inference.metadata is not None else None,
            )

    def read_dataset(
        self,
        dataset: Iterable[MultilabelClassificationExample],
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
