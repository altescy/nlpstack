import os
import warnings
from contextlib import suppress
from logging import getLogger
from os import PathLike
from typing import (
    Any,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import minato
import numpy
import requests
from sklearn.utils import murmurhash3_32

from nlpstack.common import FileBackendMapping, Pipeline, cached_property
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.data.vocabulary import Vocabulary
from nlpstack.transformers import cache as transformers_cache

from .util import masked_pool

try:
    import fasttext
except ModuleNotFoundError:
    fasttext = None

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

try:
    import sentence_transformers
except ModuleNotFoundError:
    sentence_transformers = None  # type: ignore[assignment]

try:
    import openai
except ModuleNotFoundError:
    openai = None  # type: ignore[assignment]


logger = getLogger(__name__)

_T_Fixtures = TypeVar("_T_Fixtures")


class WordEmbedding:
    """
    A word embedding.
    """

    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getitem__(self, word: str) -> numpy.ndarray:
        raise NotImplementedError

    def __contains__(self, word: str) -> bool:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def extend_vocab(self, vocab: Vocabulary, namespace: str) -> None:
        warnings.warn(f"extend_vocab is not implemented for {self.__class__.__name__}")


class MinhashWordEmbedding(WordEmbedding):
    """Compute minhash vector of character n-grams.

    Args:
        num_features: The number of features of the embedding.
        num_hashes: The number of hashes to use.
        ngram_range: The range of n-grams to use. Defaults to (3, 3).
    """

    def __init__(
        self,
        num_features: int,
        num_hashes: int = 64,
        ngram_range: Union[int, Tuple[int, int]] = 3,
    ) -> None:
        if isinstance(ngram_range, int):
            ngram_range = (ngram_range, ngram_range)
        if ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range[0] must be less than or equal to ngram_range[1]")
        if min(ngram_range) < 1:
            raise ValueError("ngram_size must be greater than or equal to 1")
        self._num_features = num_features
        self._num_hashes = num_hashes
        self._ngram_range = ngram_range

    def _iter_character_ngrams(self, text: str) -> Iterator[str]:
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                yield text[i : i + n]

    def _compute_fingerprint(self, text: str) -> List[int]:
        ngrams = set(self._iter_character_ngrams(text))
        return [min(murmurhash3_32(ngram, seed) for ngram in ngrams) for seed in range(self._num_hashes)]

    def __getitem__(self, word: str) -> numpy.ndarray:
        embedding = numpy.zeros(self._num_features, dtype=float)
        for value in self._compute_fingerprint(f"<{word}>"):
            embedding[value % self._num_features] += 1.0
        return embedding

    def __contains__(self, word: str) -> bool:
        return True

    def get_output_dim(self) -> int:
        return self._num_features


class PretrainedWordEmbedding(WordEmbedding):
    """Load a word embedding file.

    Args:
        filename: The path to the word embedding file.
        unknown_vector: The vector to use for unknown words. If None, unknown words will raise a KeyError. If "zero",
            unknown words will be represented by a vector of all zeros. If "mean", unknown words will be represented
            by the mean of all known word vectors. If a numpy.ndarray, unknown words will be represented by the given
            vector.
    """

    @staticmethod
    def _read_embeddings_file(filename: Union[str, PathLike]) -> Mapping[str, numpy.ndarray]:
        embeddings = FileBackendMapping[str, numpy.ndarray]()
        with minato.open(filename, decompress="auto") as txtfile:
            for line in txtfile:
                key, *values = line.rstrip().split()
                embeddings[key] = numpy.asarray(values, dtype=float)
        assert (
            len(set(len(embedding) for embedding in embeddings.values())) == 1
        ), "All embeddings must have the same dimension."
        return embeddings

    def __init__(
        self,
        filename: Union[str, PathLike],
        unknown_vector: Optional[Union[Literal["zero", "mean"], numpy.ndarray]] = None,
    ) -> None:
        self._filename = filename
        self._unknown_vector: Optional[numpy.ndarray] = None
        if unknown_vector == "zero":
            self._unknown_vector = numpy.zeros(self.get_output_dim(), dtype=float)
        elif unknown_vector == "mean":
            self._unknown_vector = numpy.mean(numpy.asarray(self._embeddings.values()), axis=0)
        elif isinstance(unknown_vector, numpy.ndarray):
            self._unknown_vector = unknown_vector
        elif unknown_vector is not None:
            raise ValueError(
                f"unknown_vector must be one of 'zero', 'mean', or a numpy.ndarray, but got {unknown_vector}"
            )

        self._embeddings  # load the embeddings file

    @cached_property
    def _embeddings(self) -> Mapping[str, numpy.ndarray]:
        return self._read_embeddings_file(self._filename)

    def __getitem__(self, word: str) -> numpy.ndarray:
        if word in self._embeddings:
            return cast(numpy.ndarray, self._embeddings[word])
        if self._unknown_vector is not None:
            return self._unknown_vector
        raise KeyError(word)

    def __contains__(self, word: str) -> bool:
        return word in self._embeddings

    def get_output_dim(self) -> int:
        return len(next(iter(self._embeddings.values())))

    def extend_vocab(self, vocab: Vocabulary, namespace: str) -> None:
        vocab.extend_vocab(namespace, self._embeddings.keys())


class PretrainedFasttextWordEmbedding(WordEmbedding):
    """Word embedding model of pretrained fastText.

    Args:
        filename : Path to the pretrained fastText model.
        allow_unknown_words : If False, raise `KeyError` when the word is not in the model. Defaults to `True`.

    Attributes:
        fasttext : The pretrained fastText model.
    """

    def __init__(self, filename: Union[str, PathLike], allow_unknown_words: bool = True) -> None:
        if fasttext is None:
            raise ModuleNotFoundError("fasttext is not installed")

        self._filename = filename
        self._allow_unknown_words = allow_unknown_words

        self.fasttext  # load the fasttext model

    @cached_property
    def fasttext(self) -> "fasttext.FastText":
        if fasttext is None:
            raise ModuleNotFoundError("fasttext is not installed")
        pretrained_filename = minato.cached_path(self._filename)
        return fasttext.load_model(str(pretrained_filename))

    def __getitem__(self, word: str) -> numpy.ndarray:
        if word in self.fasttext:
            return cast(numpy.ndarray, self.fasttext[word])
        if self._allow_unknown_words:
            return cast(numpy.ndarray, self.fasttext[word])
        raise KeyError(word)

    def __contains__(self, word: str) -> bool:
        return self._allow_unknown_words or (word in self.fasttext)

    def get_output_dim(self) -> int:
        return int(self.fasttext.get_dimension())

    def extend_vocab(self, vocab: Vocabulary, namespace: str) -> None:
        vocab.extend_vocab(namespace, self.fasttext.get_words())


class PretrainedTransformerWordEmbedding(WordEmbedding):
    """Word embedding model of pretrained transformer.

    Args:
        pretrained_model_name: Path to the pretrained transformer model.
        embedding_layer: Specify the embedding layer to use. Defaults to `input`.
        submodule: Specify the submodule to use. Defaults to `None`.
    """

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        embedding_layer: Literal["input", "output"] = "input",
        submodule: Optional[str] = None,
    ) -> None:
        if embedding_layer not in ("input", "output"):
            raise ValueError(f"embedding_layer must be one of 'input' or 'output', but got {embedding_layer}")

        self._pretrained_model_name = pretrained_model_name
        self._embedding_layer = embedding_layer
        self._submodule = submodule

        self.tokenizer  # load the tokenizer
        self.model  # load the model

    @cached_property
    def tokenizer(self) -> "transformers.PreTrainedTokenizer":
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    @cached_property
    def model(self) -> "transformers.PreTrainedModel":
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        model = transformers_cache.get_pretrained_model(
            pretrained_model_name,
            auto_cls=transformers.AutoModelWithLMHead if self._embedding_layer == "output" else None,
        )
        if self._submodule is not None:
            model = getattr(model, self._submodule)
        return model

    def __getitem__(self, word: str) -> numpy.ndarray:
        if word not in self.tokenizer.vocab:
            raise KeyError(word)
        index = self.tokenizer.vocab[word]
        if self._embedding_layer == "input":
            embeddings = self.model.get_input_embeddings()
        elif self._embedding_layer == "output":
            embeddings = self.model.get_output_embeddings()
        else:
            raise ValueError(f"embedding_layer must be one of 'input' or 'output', but got {self._embedding_layer}")
        return cast(numpy.ndarray, embeddings.weight.data[index].detach().numpy())

    def __contains__(self, word: str) -> bool:
        return word in self.tokenizer.vocab

    def get_output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def extend_vocab(self, vocab: Vocabulary, namespace: str) -> None:
        vocab.extend_vocab(namespace, self.tokenizer.vocab.keys())


class TextEmbedding(Pipeline[str, numpy.ndarray, _T_Fixtures], Generic[_T_Fixtures]):
    """
    A text embedding model.
    """

    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_output_dim(self) -> int:
        raise NotImplementedError


class BagOfEmbeddingsTextEmbedding(TextEmbedding["BagOfEmbeddingsTextEmbedding.Fixtures"]):
    """
    A text embedding model using bag-of-embeddings.

    The following pooling methods are supported:

    - `"mean"`: Mean pooling.
    - `"max"`: Max pooling.
    - `"min"`: Min pooling.
    - `"sum"`: Sum pooling.
    - `"hier"`: Hierarchical pooling: https://arxiv.org/abs/1805.09843

    Args:
        word_embedding: A word embedding model.
        tokenizer: A tokenizer.
        pooling: A pooling method. Defaults to `"mean"`.
        normalize: If `True`, normalize word embeddings before pooling. Defaults to `False`.
        window_size: A window size for hierarchical pooling. Defaults to `None`.
    """

    class Fixtures(NamedTuple):
        word_embedding: WordEmbedding
        tokenizer: Tokenizer

    def __init__(
        self,
        word_embedding: WordEmbedding,
        tokenizer: Optional[Tokenizer] = None,
        pooling: Literal["mean", "max", "min", "sum", "hier"] = "mean",
        normalize: bool = False,
        window_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if pooling not in ("mean", "max", "min", "sum", "hier"):
            raise ValueError(f"pooling must be one of 'mean', 'max', 'min', 'sum', or 'hier', but got {pooling}")
        if pooling == "hier" and window_size is None:
            raise ValueError("window_size must be specified when pooling is 'hier'")

        super().__init__(**kwargs)
        self._pooling = pooling
        self._normalize = normalize
        self._window_size = window_size
        self._output_dim = word_embedding.get_output_dim()
        self._fixtures = BagOfEmbeddingsTextEmbedding.Fixtures(
            word_embedding,
            tokenizer or WhitespaceTokenizer(),
        )

    def get_output_dim(self) -> int:
        return self._output_dim

    @property
    def fixtures(self) -> "BagOfEmbeddingsTextEmbedding.Fixtures":  # type: ignore[override]
        return self._fixtures

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "BagOfEmbeddingsTextEmbedding.Fixtures",
    ) -> List[numpy.ndarray]:
        if not batch:
            return list(numpy.zeros((0, self.get_output_dim()), dtype=numpy.float32))

        batch_size = len(batch)

        batch_tokens = [fixtures.tokenizer.tokenize(text) for text in batch]
        max_length = max(len(tokens) for tokens in batch_tokens)

        embeddings = numpy.zeros((batch_size, max_length, self.get_output_dim()), dtype=float)
        mask = numpy.zeros((batch_size, max_length), dtype=bool)

        for batch_index, tokens in enumerate(batch_tokens):
            for token_index, token in enumerate(tokens):
                if token.surface in fixtures.word_embedding:
                    embedding = fixtures.word_embedding[token.surface]
                    embeddings[batch_index, token_index] = embedding
                    mask[batch_index, token_index] = True

        return list(
            masked_pool(
                embeddings,
                mask,
                pooling=self._pooling,
                normalize=self._normalize,
                window_size=self._window_size,
            )
        )


class PretrainedTransformerTextEmbedding(TextEmbedding["PretrainedTransformerTextEmbedding.Fixtures"]):
    """
    A text embedding model using pretrained transformer models.

    Args:
        pretrained_model_name: A pretrained model name.
        submodule: A submodule name. Defaults to `None`.
        pooling: A pooling method. Defaults to `"mean"`.
        normalize: If `True`, normalize word embeddings before pooling. Defaults to `False`.
        window_size: A window size for hierarchical pooling. Defaults to `None`.
    """

    class Fixtures(NamedTuple):
        pipeline: "transformers.Pipeline"

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        submodule: Optional[str] = None,
        pooling: Literal["mean", "max", "min", "sum", "hier", "first", "last"] = "mean",
        normalize: bool = False,
        window_size: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if transformers is None:
            raise ModuleNotFoundError("transformers is not installed")

        if pooling == "hier" and window_size is None:
            raise ValueError("window_size must be specified when pooling is 'hier'")

        super().__init__(**kwargs)
        self._pretrained_model_name = pretrained_model_name
        self._submodule = submodule
        self._pooling = pooling
        self._normalize = normalize
        self._window_size = window_size
        self._device = device

    def get_tokenizer(self) -> "transformers.PreTrainedTokenizer":
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    def get_model(self) -> "transformers.PreTrainedModel":
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        model = transformers_cache.get_pretrained_model(pretrained_model_name)
        if self._submodule is not None:
            model = getattr(model, self._submodule)
        return model

    def get_output_dim(self) -> int:
        return int(self.fixtures.pipeline.model.config.hidden_size)

    @cached_property
    def fixtures(self) -> "PretrainedTransformerTextEmbedding.Fixtures":  # type: ignore[override]
        return PretrainedTransformerTextEmbedding.Fixtures(
            transformers.pipeline(
                "feature-extraction",
                model=self.get_model(),
                tokenizer=self.get_tokenizer(),
                device=self._device,
            )
        )

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "PretrainedTransformerTextEmbedding.Fixtures",
    ) -> List[numpy.ndarray]:
        features = fixtures.pipeline(list(batch))
        return [
            masked_pool(
                numpy.array(x),
                pooling=self._pooling,
                normalize=self._normalize,
                window_size=self._window_size,
            )
            for x in features
        ]


class SentenceTransformerTextEmbedding(TextEmbedding["SentenceTransformerTextEmbedding.Fixtures"]):
    """
    A text embedding model using sentence-transformers.

    Args:
        model_name: A model name. Defaults to `"all-MiniLM-L6-v2"`.
        normalize_embeddings: Whether to normalize output embeddings. Defaults to `False`.
        device: A device to use. Defaults to `"cpu"`.
    """

    class Fixtures(NamedTuple):
        model: "sentence_transformers.SentenceTransformer"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if sentence_transformers is None:
            raise ModuleNotFoundError("sentence-transformers is not installed")

        super().__init__(**kwargs)
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._device = device

    def get_model(self) -> "sentence_transformers.SentenceTransformer":
        assert sentence_transformers is not None
        return sentence_transformers.SentenceTransformer(self._model_name, device=self._device)

    @cached_property
    def fixtures(self) -> "SentenceTransformerTextEmbedding.Fixtures":  # type: ignore[override]
        return SentenceTransformerTextEmbedding.Fixtures(self.get_model())

    def get_output_dim(self) -> int:
        return int(self.fixtures.model.get_sentence_embedding_dimension())

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "SentenceTransformerTextEmbedding.Fixtures",
    ) -> List[numpy.ndarray]:
        if not batch:
            return list(numpy.zeros((0, self.get_output_dim()), dtype=numpy.float32))
        embeddings = fixtures.model.encode(
            list(batch),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )
        return cast(List[numpy.ndarray], list(embeddings))


class OpenAITextEmbedding(TextEmbedding["OpenAITextEmbedding.Fixtures"]):
    """
    A text embedding model using OpenAI API.

    You need to set `OPENAI_API_KEY` environment variable.

    Args:
        model_name: A model name. Defaults to `"text-embedding-ada-002"`.
        organization_id: An organization ID. Defaults to `None`.
        api_base: The base path for the API. Defaults to `None`.
        api_type: The type of the API deployment. Defaults to `None`.
        api_version: The version of the API. Defaults to `None`.
    """

    class Fixtures(NamedTuple):
        client: "openai.OpenAI"

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        if openai is None:
            raise ModuleNotFoundError("openai is not installed")

        super().__init__(**kwargs)
        self._organization = organization
        self._base_url = base_url
        self._api_key = api_key
        self._max_retries = max_retries
        self._model_name = model_name

    def get_client(self) -> "openai.OpenAI":
        assert openai is not None
        return openai.OpenAI(
            organization=self._organization,
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._max_retries,
        )

    @cached_property
    def fixtures(self) -> "OpenAITextEmbedding.Fixtures":  # type: ignore[override]
        return OpenAITextEmbedding.Fixtures(self.get_client())

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "OpenAITextEmbedding.Fixtures",
    ) -> List[numpy.ndarray]:
        batch = [t.replace("\n", " ") for t in batch]
        response = fixtures.client.embeddings.create(input=batch, model=self._model_name)
        embeddings = sorted(response.data, key=lambda e: e.index)  # type: ignore
        return [numpy.array(embedding.embedding) for embedding in embeddings]


class HuggingFaceTextEmbedding(TextEmbedding["HuggingFaceTextEmbedding.Fixtures"]):
    """
    A text embedding model using HuggingFace API.

    You need to set `HUGGINGFACE_API_KEY` environment variable.

    Args:
        model_name: A model name. Defaults to `"sentence-transformers/all-MiniLM-L6-v2"`.
    """

    class Fixtures(NamedTuple):
        session: requests.Session

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    def get_session(self) -> requests.Session:
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if api_key is None:
            raise ValueError("Please provide an HuggingFace API key.")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {api_key}"})
        return session

    @cached_property
    def fixtures(self) -> "HuggingFaceTextEmbedding.Fixtures":  # type: ignore[override]
        return HuggingFaceTextEmbedding.Fixtures(self.get_session())

    def apply_batch(
        self,
        batch: Sequence[str],
        fixtures: "HuggingFaceTextEmbedding.Fixtures",
    ) -> List[numpy.ndarray]:
        # Call HuggingFace Embedding API for each document
        return list(
            numpy.array(
                fixtures.session.post(  # type: ignore
                    self._api_url, json={"inputs": batch, "options": {"wait_for_model": True}}
                ).json()
            )
        )
