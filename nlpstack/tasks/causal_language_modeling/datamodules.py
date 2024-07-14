from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import PassThroughPipeline, Pipeline
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import DataclassTokenizer, Tokenizer, WhitespaceTokenizer

from .types import CausalLanguageModelingExample, CausalLanguageModelingInference, CausalLanguageModelingPrediction

logger = getLogger(__name__)

CausalLanguageModelingPreprocessor = Pipeline[
    CausalLanguageModelingExample, CausalLanguageModelingExample, Any, Optional[Any]
]
CausalLanguageModelingPostprocessor = Pipeline[
    CausalLanguageModelingPrediction, CausalLanguageModelingPrediction, Any, Optional[Any]
]


class CausalLanguageModelingDataModule(
    DataModule[
        CausalLanguageModelingExample,
        CausalLanguageModelingInference,
        CausalLanguageModelingPrediction,
    ]
):
    """
    A data module for causal language modeling.

    Args:
        vocab: The vocabulary.
        tokenizer: The tokenizer.
        token_indexers: The token indexers to index the tokens.
        namespace: The vocabulary namespace for the tokens.
        preprocessor: The preprocessor to apply to the dataset before tokenization.
            Defaults to `None`.
        postprocessor: The postprocessor to apply to the predictions after inference.
            Defaults to `None`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        namespace: str = "tokens",
        preprocessor: Optional[CausalLanguageModelingPreprocessor] = None,
        postprocessor: Optional[CausalLanguageModelingPostprocessor] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._namespace = namespace
        self._preprocessor = preprocessor or PassThroughPipeline()
        self._postprocessor = postprocessor or PassThroughPipeline()

        self._generation_mode = False

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def set_generation_mode(self, generation_mode: bool) -> None:
        """
        Set the generation mode.

        Args:
            generation_mode: Whether to set the generation mode. If `False`, the data module will
            build instances for training (with labels / drop the last token). If `True`, the data
            module will build instances for generation (without labels / keep the last token).
        """

        self._generation_mode = generation_mode

    def setup(self, dataset: Sequence[CausalLanguageModelingExample]) -> None:
        """
        Setup the data module.

        This method builds the vocabulary from the dataset.

        Args:
            dataset: The dataset to tokenize and build the vocabulary from.
        """

        self._build_vocab(dataset)

    def preprocess(
        self,
        dataset: Iterable[CausalLanguageModelingExample],
    ) -> Iterator[CausalLanguageModelingExample]:
        pipeline = self._preprocessor | DataclassTokenizer[CausalLanguageModelingExample, Any](
            {"text": self._tokenizer}
        )
        return pipeline(dataset)

    def _build_vocab(self, dataset: Sequence[CausalLanguageModelingExample]) -> None:
        def text_iterator() -> Iterator[Sequence[Token]]:
            for example in dataset:
                assert not isinstance(example.text, str), "Dataset must be tokenized."
                yield example.text

        for name, token_indexer in self._token_indexers.items():
            token_indexer.build_vocab(self.vocab, text_iterator())

    def build_instance(self, example: CausalLanguageModelingExample) -> Instance:
        """
        Build an instance from an example.
        If the given `example.text` is a string, it will be tokenized using the tokenizer.

        Args:
            example: The example to build the instance from.
        """

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
        """
        Build predictions from an batched inference result.

        Args:
            inference: The batched inference result.

        Returns:
            The predictions.
        """

        def prediction_iterator() -> Iterator[CausalLanguageModelingPrediction]:
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

        return self._postprocessor(prediction_iterator())
