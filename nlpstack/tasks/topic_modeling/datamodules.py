from logging import getLogger
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from nlpstack.common import PassThroughPipeline, Pipeline
from nlpstack.data import DataModule, Instance, Token, Vocabulary
from nlpstack.data.fields import Field, MetadataField, TextField
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import DataclassTokenizer, Tokenizer, WhitespaceTokenizer

from .types import TopicModelingExample, TopicModelingInference, TopicModelingPrediction

logger = getLogger(__name__)

TopicModelingPreprocessor = Pipeline[TopicModelingExample, TopicModelingExample, Any]
TopicModelingPostprocessor = Pipeline[TopicModelingPrediction, TopicModelingPrediction, Any]


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
        preprocessor: Optional[TopicModelingPreprocessor] = None,
        postprocessor: Optional[TopicModelingPostprocessor] = None,
    ) -> None:
        self._vocab = vocab
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._preprocessor = preprocessor or PassThroughPipeline()
        self._postprocessor = postprocessor or PassThroughPipeline()

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
            self._build_vocab(dataset)

    def preprocess(self, dataset: Iterable[TopicModelingExample], **kwargs: Any) -> Iterator[TopicModelingExample]:
        pipeline = self._preprocessor | DataclassTokenizer[TopicModelingExample]({"text": self._tokenizer})
        return pipeline(dataset)

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

        def prediction_iterator() -> Iterator[TopicModelingPrediction]:
            for topic_distribution in inference.topic_distribution:
                yield TopicModelingPrediction(topic_distribution.tolist())

        yield from self._postprocessor(prediction_iterator())
