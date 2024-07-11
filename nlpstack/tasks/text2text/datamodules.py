import itertools
from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import wrap_iterator
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer

from .types import Text2TextExample, Text2TextInference, Text2TextPrediction

logger = getLogger(__name__)


class Text2TextDataModule(
    DataModule[
        Text2TextExample,
        Text2TextInference,
        Text2TextPrediction,
    ]
):
    """
    A data module for text-to-text task.

    Args:
        vocab: The vocabulary.
        source_tokenizer: The tokenizer for source text. Defaults to `WhitespaceTokenizer()`.
        target_tokenizer: The tokenizer for target text. If not given, `source_tokenizer` is used. Defaults to `None`.
        source_token_indexers: The token indexers to index the source tokens. Defaults to
            `{"tokens": SingleIdTokenIndexer()}`.
        target_token_indexers: The token indexers to index the target tokens. If not given,
            `source_token_indexers` is used. Defaults to `None`.
        source_namespace: The vocabulary namespace for source text. Defaults to `"tokens"`.
        target_namespace: The vocabulary namespace for target text. Defaults to `"tokens"`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_tokenizer: Optional[Tokenizer] = None,
        target_tokenizer: Optional[Tokenizer] = None,
        source_token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        target_token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        source_namespace: str = "tokens",
        target_namespace: str = "tokens",
    ) -> None:
        self._vocab = vocab
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def source_tokenizer(self) -> Tokenizer:
        return self._source_tokenizer

    @property
    def target_tokenizer(self) -> Tokenizer:
        return self._target_tokenizer

    def setup(
        self,
        *args: Any,
        dataset: Optional[Sequence[Text2TextExample]] = None,
        **kwargs: Any,
    ) -> None:
        if dataset:
            self._build_vocab(dataset)

    def preprocess(self, dataset: Iterable[Text2TextExample], **kwargs: Any) -> Iterator[Text2TextExample]:
        return self._tokenize(dataset)

    def _tokenize(self, dataset: Iterable[Text2TextExample]) -> Iterator[Text2TextExample]:
        """
        Setup the data module.

        This method tokenizes the dataset and builds the vocabulary.

        Args:
            dataset: The dataset to tokenize and build the vocabulary from.
        """

        def tokenized_document_generator(dataset: Iterable[Text2TextExample]) -> Iterator[Text2TextExample]:
            for example in dataset:
                if isinstance(example.source, str):
                    tokenized_source = self._source_tokenizer.tokenize(example.source)
                else:
                    tokenized_source = list(example.source)
                if example.target is not None:
                    if isinstance(example.target, str):
                        tokenized_target = self._target_tokenizer.tokenize(example.target)
                    else:
                        tokenized_target = list(example.target)
                else:
                    tokenized_target = None
                yield Text2TextExample(source=tokenized_source, target=tokenized_target)

        return wrap_iterator(tokenized_document_generator, dataset)

    def _build_vocab(self, dataset: Iterable[Text2TextExample]) -> None:
        def source_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.source, str), "Dataset must be tokenized."
                yield example.source

        def target_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert example.target is not None
                assert not isinstance(example.target, str), "Dataset must be tokenized."
                yield example.target

        if self._source_token_indexers is self._target_token_indexers:
            for name, token_indexer in self._source_token_indexers.items():
                token_indexer.build_vocab(
                    self._vocab,
                    itertools.chain(source_iterator(), target_iterator()),
                )
        else:
            for name, token_indexer in self._source_token_indexers.items():
                token_indexer.build_vocab(self.vocab, source_iterator())
            for name, token_indexer in self._target_token_indexers.items():
                token_indexer.build_vocab(self.vocab, target_iterator())

    def build_instance(self, example: Text2TextExample) -> Instance:
        """
        Build an instance from an example.
        If the given `example.source` / `example.target` is a string, it will be tokenized using the tokenizer.

        Args:
            example: The example to build the instance from.
        """

        source = example.source
        target = example.target
        metadata = example.metadata

        if isinstance(source, str):
            source = self._source_tokenizer.tokenize(source)
        if self._vocab.has_bos_token(self._source_namespace):
            source = [Token(self._vocab.get_bos_token(self._source_namespace))] + list(source)
        if self._vocab.has_eos_token(self._source_namespace):
            source = list(source) + [Token(self._vocab.get_eos_token(self._source_namespace))]

        fields: Dict[str, Field] = {}
        fields["source"] = TextField(source, self.vocab, self._source_token_indexers)

        if target is not None:
            if isinstance(target, str):
                target = self._target_tokenizer.tokenize(target)
            if self._vocab.has_bos_token(self._target_namespace):
                target = [Token(self._vocab.get_bos_token(self._target_namespace))] + list(target)
            if self._vocab.has_eos_token(self._target_namespace):
                target = list(target) + [Token(self._vocab.get_eos_token(self._target_namespace))]
            fields["target"] = TextField(target, self.vocab, self._target_token_indexers)

        if metadata:
            fields["metadata"] = MetadataField(metadata)

        return Instance(**fields)

    def build_predictions(self, inference: Text2TextInference) -> Iterator[Text2TextPrediction]:
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        token_indices_to_ignore = {self._vocab.get_pad_index(self._target_namespace)}
        if self._vocab.has_bos_token(self._target_namespace):
            token_indices_to_ignore.add(self._vocab.get_bos_index(self._target_namespace))
        if self._vocab.has_eos_token(self._target_namespace):
            token_indices_to_ignore.add(self._vocab.get_eos_index(self._target_namespace))
        for top_token_ids, scores in zip(inference.pred_token_ids.tolist(), inference.scores.tolist()):
            top_tokens = [
                [
                    self.vocab.get_token_by_index(self._target_namespace, token_id)
                    for token_id in token_ids
                    if token_id not in token_indices_to_ignore
                ]
                for token_ids in top_token_ids
            ]
            top_texts = [self._target_tokenizer.detokenize(tokens) for tokens in top_tokens]
            yield Text2TextPrediction(top_texts, top_tokens, scores)
