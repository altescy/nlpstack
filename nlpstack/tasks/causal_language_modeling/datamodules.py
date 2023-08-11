from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import FileBackendSequence, ProgressBar
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.data.util import iter_with_callback

from .types import CausalLanguageModelingExample, CausalLanguageModelingInference, CausalLanguageModelingPrediction

logger = getLogger(__name__)


class CausalLanguageModelingDataModule(
    DataModule[
        CausalLanguageModelingExample,
        CausalLanguageModelingInference,
        CausalLanguageModelingPrediction,
    ]
):
    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        namespace: str = "tokens",
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._namespace = namespace
        self._generation_mode = False

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def setup(
        self,
        *args: Any,
        dataset: Optional[Sequence[CausalLanguageModelingExample]] = None,
        generation_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if dataset:
            with ProgressBar[int](len(dataset) * 2) as progress:
                progress.set_description("Tokenizing dataset")
                dataset = self.tokenize(iter_with_callback(dataset, lambda _: progress.update()))
                progress.set_description("Building vocab    ")
                self._build_vocab(iter_with_callback(dataset, lambda _: progress.update()))
        if generation_mode is not None:
            self._generation_mode = generation_mode

    def tokenize(self, dataset: Iterable[CausalLanguageModelingExample]) -> Sequence[CausalLanguageModelingExample]:
        if not dataset:
            return []

        def tokenized_document_generator() -> Iterator[CausalLanguageModelingExample]:
            for example in dataset:
                if isinstance(example.text, str):
                    tokenized_text = self._tokenizer.tokenize(example.text)
                else:
                    tokenized_text = list(example.text)
                yield CausalLanguageModelingExample(text=tokenized_text)

        return FileBackendSequence.from_iterable(tokenized_document_generator())

    def _build_vocab(self, dataset: Iterable[CausalLanguageModelingExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

    def build_instance(self, example: CausalLanguageModelingExample) -> Instance:
        text = example.text
        metadata = example.metadata
        labels: Optional[Sequence[Token]] = None

        if isinstance(text, str):
            text = self._tokenizer.tokenize(text)

        if self._vocab.has_bos_token(self._namespace):
            text = [Token(self._vocab.get_bos_token(self._namespace))] + list(text)

        if not self._generation_mode:
            if self._vocab.has_eos_token(self._namespace):
                text = list(text) + [Token(self._vocab.get_eos_token(self._namespace))]
            labels = text[1:]
            text = text[:-1]

        fields: Dict[str, Field] = {}
        fields["text"] = TextField(text, self.vocab, self._token_indexers)

        if labels is not None:
            fields["labels"] = TextField(labels, self.vocab, self._token_indexers)

        if metadata:
            fields["metadata"] = MetadataField(metadata)

        return Instance(**fields)

    def build_predictions(
        self, inference: CausalLanguageModelingInference
    ) -> Iterator[CausalLanguageModelingPrediction]:
        token_indices_to_ignore = {self._vocab.get_pad_index(self._namespace)}
        if self._vocab.has_bos_token(self._namespace):
            token_indices_to_ignore.add(self._vocab.get_bos_index(self._namespace))
        if self._vocab.has_eos_token(self._namespace):
            token_indices_to_ignore.add(self._vocab.get_eos_index(self._namespace))
        for top_token_ids, scores in zip(inference.pred_token_ids.tolist(), inference.scores.tolist()):
            top_tokens = [
                [
                    self.vocab.get_token_by_index(self._namespace, token_id)
                    for token_id in token_ids
                    if token_id not in token_indices_to_ignore
                ]
                for token_ids in top_token_ids
            ]
            top_texts = [self._tokenizer.detokenize(tokens) for tokens in top_tokens]
            yield CausalLanguageModelingPrediction(top_texts, top_tokens, scores)

    def read_dataset(
        self,
        dataset: Iterable[CausalLanguageModelingExample],
        is_training: bool = False,
        **kwargs: Any,
    ) -> Iterator[Instance]:
        logger.info("Building instances...")
        for example in ProgressBar(dataset, desc="Building instances"):
            yield self.build_instance(example)
