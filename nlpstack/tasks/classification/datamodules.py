from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy

from nlpstack.common import FileBackendSequence, ProgressBar
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, LabelField, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .data import ClassificationExample, ClassificationInference, ClassificationPrediction

logger = getLogger(__name__)


class BasicClassificationDataModule(
    DataModule[
        ClassificationExample,
        ClassificationInference,
        ClassificationPrediction,
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

    def _tokenize(self, dataset: Iterable[ClassificationExample]) -> Sequence[ClassificationExample]:
        if not dataset:
            return []

        def tokenized_document_generator() -> Iterator[ClassificationExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield ClassificationExample(
                    text=tokenized_text,
                    label=example.label,
                )

        return FileBackendSequence.from_iterable(tokenized_document_generator())

    def _build_vocab(self, dataset: Sequence[ClassificationExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[List[str]]:
            for example in dataset:
                label = example.label
                assert label is not None, "Dataset must have labels."
                yield [label]

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

        self.vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: ClassificationExample) -> Instance:
        text = example.text
        label = example.label
        metadata = example.metadata

        if isinstance(text, str):
            text = self._tokenizer.tokenize(text)

        fields: Dict[str, Field] = {}
        fields["text"] = TextField(text, self.vocab, self._token_indexers)

        if metadata:
            fields["metadata"] = MetadataField(metadata)

        if label:
            fields["label"] = LabelField(label, vocab=self.vocab[self.label_namespace])

        return Instance(**fields)

    def build_predictions(self, inference: ClassificationInference) -> Iterator[ClassificationPrediction]:
        sorted_indices = inference.probs.argsort(axis=1)[:, ::-1]
        sorted_probs = numpy.take_along_axis(inference.probs, sorted_indices, axis=1)
        for i, (top_indices, top_probs) in enumerate(zip(sorted_indices.tolist(), sorted_probs.tolist())):
            num_labels_to_return = len(top_indices)
            if inference.threshold is not None:
                num_labels_to_return = sum(prob >= inference.threshold for prob in top_probs)
            if inference.top_k is not None:
                num_labels_to_return = min(num_labels_to_return, inference.top_k)
            top_indices = top_indices[:num_labels_to_return]
            top_probs = top_probs[:num_labels_to_return]
            yield ClassificationPrediction(
                top_probs=top_probs,
                top_labels=[self.vocab.get_token_by_index(self.label_namespace, index) for index in top_indices],
                metadata=inference.metadata[i] if inference.metadata is not None else None,
            )

    def read_dataset(
        self,
        dataset: Iterable[ClassificationExample],
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
