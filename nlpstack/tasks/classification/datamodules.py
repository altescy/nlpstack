import itertools
from logging import getLogger
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy

from nlpstack.common import FileBackendSequence, PassThroughPipeline, ProgressBar
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, LabelField, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .types import ClassificationExample, ClassificationInference, ClassificationPrediction

logger = getLogger(__name__)

ClassificationPreprocessor = Callable[[Iterable[ClassificationExample]], Iterable[ClassificationExample]]
ClassificationPostprocessor = Callable[[Iterable[ClassificationPrediction]], Iterable[ClassificationPrediction]]


class BasicClassificationDataModule(
    DataModule[
        ClassificationExample,
        ClassificationInference,
        ClassificationPrediction,
    ]
):
    """
    A data module for classification.

    Args:
        vocab: The vocabulary.
        tokenizer: The tokenizer. Defaults to `WhitespaceTokenizer()`.
        token_indexers: The token indexers to index the tokens. Defaults to
            `{"tokens": SingleIdTokenIndexer()}`.
        label_namespace: The vocabulary namespace for the labels. Defaults to `"labels"`.
        labels: The set of labels. If not given, the labels will be collected from the
            training dataset. Defaults to `None`.
        preprocessors: The preprocessors to apply to the dataset before tokenization.
            Defaults to `None`.
        postprocessors: The postprocessors to apply to the predictions after inference.
            Defaults to `None`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        label_namespace: str = "labels",
        labels: Optional[Sequence[str]] = None,
        preprocessor: Optional[ClassificationPreprocessor] = None,
        postprocessor: Optional[ClassificationPostprocessor] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace
        self._labels = sorted(set(labels)) if labels else None
        self._preprocessor = preprocessor or PassThroughPipeline[ClassificationExample]()
        self._postprocessor = postprocessor or PassThroughPipeline[ClassificationPrediction]()

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
        dataset: Optional[Sequence[ClassificationExample]] = None,
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
            dataset = self.tokenize(ProgressBar(self._preprocessor(dataset), desc="Tokenizing dataset"))
            self._build_vocab(dataset)

    def tokenize(self, dataset: Iterable[ClassificationExample]) -> Sequence[ClassificationExample]:
        """
        Tokenize the dataset and return the tokenized dataset.

        Args:
            dataset: The dataset to tokenize.

        Returns:
            The tokenized dataset.
        """

        if not dataset:
            return []

        def get_text_to_tokenize(example: ClassificationExample) -> str:
            if isinstance(example.text, str):
                return example.text
            return ""

        def tokenized_document_generator() -> Iterator[ClassificationExample]:
            nonlocal dataset
            dataset, dataset_for_text = itertools.tee(dataset)
            tokenized_texts = self._tokenizer(map(get_text_to_tokenize, dataset_for_text))
            for example, tokenized_text in zip(dataset, tokenized_texts):
                tokenized_text = tokenized_text if isinstance(example.text, str) else list(example.text)
                yield ClassificationExample(
                    text=tokenized_text,
                    label=example.label,
                    metadata=example.metadata,
                )

        return FileBackendSequence.from_iterable(tokenized_document_generator())

    def _build_vocab(self, dataset: Sequence[ClassificationExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        def label_iterator() -> Iterator[List[str]]:
            if self._labels is not None:
                yield self._labels
            else:
                for example in dataset:
                    label = example.label
                    assert label is not None, "Dataset must have labels."
                    yield [label]

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

        self.vocab.build_vocab_from_documents(self._label_namespace, label_iterator())

    def build_instance(self, example: ClassificationExample) -> Instance:
        """
        Build an instance from an example.
        If the given `example.text` is a string, it will be tokenized using the tokenizer.

        Args:
            example: The example to build the instance from.
        """

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
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        def prediction_iterator() -> Iterator[ClassificationPrediction]:
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

        yield from self._postprocessor(prediction_iterator())

    def read_dataset(self, dataset: Iterable[ClassificationExample], **kwargs: Any) -> Iterator[Instance]:
        """
        Read the dataset and return a generator of instances.

        Args:
            dataset: The dataset to read.

        Returns:
            A generator of instances.
        """

        logger.info("Building instances...")
        for example in ProgressBar(self._preprocessor(dataset), desc="Building instances"):
            yield self.build_instance(example)
