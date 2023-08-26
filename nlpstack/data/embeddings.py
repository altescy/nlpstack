import warnings
from contextlib import suppress
from logging import getLogger
from os import PathLike
from typing import Any, Iterator, List, Literal, Mapping, Optional, Sequence, Union, cast

import minato
import numpy

from nlpstack.common import FileBackendMapping, cached_property, murmurhash3
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.data.vocabulary import Vocabulary
from nlpstack.transformers import cache as transformers_cache

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
    sentence_transformers = None

try:
    import openai
except ModuleNotFoundError:
    openai = None  # type: ignore[assignment]


logger = getLogger(__name__)


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
        ngram_size: The size of the character n-grams to use.
    """

    def __init__(
        self,
        num_features: int,
        num_hashes: int = 64,
        ngram_size: int = 3,
    ) -> None:
        if ngram_size < 1:
            raise ValueError("ngram_size must be greater than or equal to 1")
        self._num_features = num_features
        self._num_hashes = num_hashes
        self._ngram_size = ngram_size

    def _iter_character_ngrams(self, text: str) -> Iterator[str]:
        length = max(len(text), self._ngram_size)
        for start, end in zip(range(0, length), range(self._ngram_size, length + 1)):
            yield text[start:end]

    def _compute_fingerprint(self, text: str) -> List[int]:
        ngrams = set(self._iter_character_ngrams(text))
        return [min(murmurhash3(ngram, seed) for ngram in ngrams) for seed in range(self._num_hashes)]

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
            pretrained_model_name, with_head=self._embedding_layer == "output"
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


class TextEmbedding:
    """
    A text embedding model.
    """

    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def __call__(self, texts: Sequence[str]) -> numpy.ndarray:
        raise NotImplementedError


class BagOfEmbeddingsTextEmbedding:
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

    def __init__(
        self,
        word_embedding: WordEmbedding,
        tokenizer: Optional[Tokenizer] = None,
        pooling: Literal["mean", "max", "min", "sum", "hier"] = "mean",
        normalize: bool = False,
        window_size: Optional[int] = None,
    ) -> None:
        if pooling not in ("mean", "max", "min", "sum", "hier"):
            raise ValueError(f"pooling must be one of 'mean', 'max', 'min', 'sum', or 'hier', but got {pooling}")
        if pooling == "hier" and window_size is None:
            raise ValueError("window_size must be specified when pooling is 'hier'")

        self._word_embedding = word_embedding
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._pooling = pooling
        self._normalize = normalize
        self._window_size = window_size

    def get_output_dim(self) -> int:
        return self._word_embedding.get_output_dim()

    def __call__(self, texts: Sequence[str]) -> numpy.ndarray:
        if not texts:
            return numpy.zeros((0, self.get_output_dim()), dtype=numpy.float32)

        batch_tokens = [self._tokenizer.tokenize(text) for text in texts]
        max_length = max(len(tokens) for tokens in batch_tokens)

        embeddings = numpy.zeros((len(texts), max_length, self.get_output_dim()), dtype=float)
        mask = numpy.zeros((len(texts), max_length), dtype=bool)

        for batch_index, tokens in enumerate(batch_tokens):
            for token_index, token in enumerate(tokens):
                if token.surface in self._word_embedding:
                    embedding = self._word_embedding[token.surface]
                    if self._normalize:
                        embedding /= numpy.linalg.norm(embedding) + 1e-13
                    embeddings[batch_index, token_index] = embedding
                    mask[batch_index, token_index] = True

        if self._pooling == "mean":
            return cast(numpy.ndarray, embeddings.sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-13))

        if self._pooling == "max":
            embeddings[~mask] = float("-inf")
            return cast(numpy.ndarray, embeddings.max(axis=1))

        if self._pooling == "min":
            embeddings[~mask] = float("inf")
            return cast(numpy.ndarray, embeddings.min(axis=1))

        if self._pooling == "sum":
            return cast(numpy.ndarray, embeddings.sum(axis=1))

        if self._pooling == "hier":
            return numpy.array([self._hierarchical_pooling(x, m) for x, m in zip(embeddings, mask)])

        raise ValueError(f"pooling must be one of 'mean', 'max', 'min', 'sum', or 'hier', but got {self._pooling}")

    def _hierarchical_pooling(self, vectors: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
        assert self._window_size is not None
        vectors = vectors[mask]
        if len(vectors) < self._window_size:
            padding_size = numpy.ceil((self._window_size - len(vectors)) / 2)
            vectors = numpy.pad(vectors, ((padding_size, padding_size), (0, 0)), "constant")
        output = -numpy.inf * numpy.ones(self.get_output_dim())
        for offset in range(len(vectors) - self._window_size + 1):
            window = vectors[offset : offset + self._window_size]
            output = numpy.maximum(output, window.mean(0))
        return output


class SentenceTransformerTextEmbedding(TextEmbedding):
    """
    A text embedding model using sentence-transformers.

    Args:
        model_name: A model name. Defaults to `"all-MiniLM-L6-v2"`.
        normalize_embeddings: Whether to normalize embeddings. Defaults to `False`.
        device: A device to use. Defaults to `"cpu"`.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
        device: str = "cpu",
    ) -> None:
        if sentence_transformers is None:
            raise ModuleNotFoundError("sentence-transformers is not installed")

        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._device = device

        self.model  # load model

    @cached_property
    def model(self) -> "sentence_transformers.SentenceTransformer":
        assert sentence_transformers is not None
        return sentence_transformers.SentenceTransformer(self._model_name, device=self._device)

    def get_output_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def __call__(self, texts: Sequence[str]) -> numpy.ndarray:
        if not texts:
            return numpy.zeros((0, self.get_output_dim()), dtype=numpy.float32)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )
        return cast(numpy.ndarray, embeddings)


class OpenAITextEmbedding(TextEmbedding):
    """
    A text embedding model using OpenAI API.

    Args:
        api_key: An API key. Defaults to `None`.
        model_name: A model name. Defaults to `"text-embedding-ada-002"`.
        organization_id: An organization ID. Defaults to `None`.
        api_base: The base path for the API. Defaults to `None`.
        api_type: The type of the API deployment. Defaults to `None`.
        api_version: The version of the API. Defaults to `None`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        organization_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        if openai is None:
            raise ModuleNotFoundError("openai is not installed")

        if api_key is not None:
            openai.api_key = api_key
        elif openai.api_key is None:
            raise ValueError("Please provide an OpenAI API key.")

        if api_base is not None:
            openai.api_base = api_base

        if api_version is not None:
            openai.api_version = api_version

        if api_type is not None:
            openai.api_type = api_type

        if organization_id is not None:
            openai.organization = organization_id

        self._client = openai.Embedding
        self._model_name = model_name

    def __call__(self, texts: Sequence[str]) -> numpy.ndarray:
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self._client.create(input=texts, engine=self._model_name)["data"]  # type: ignore[no-untyped-call]
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore
        return numpy.array([result["embedding"] for result in sorted_embeddings])


class HuggingFaceTextEmbedding(TextEmbedding):
    """
    A text embedding model using HuggingFace API.

    Args:
        api_key: An API key. Defaults to `None`.
        model_name: A model name. Defaults to `"sentence-transformers/all-MiniLM-L6-v2"`.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        import requests

        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __call__(self, texts: Sequence[str]) -> numpy.ndarray:
        # Call HuggingFace Embedding API for each document
        return numpy.array(
            self._session.post(  # type: ignore
                self._api_url, json={"inputs": texts, "options": {"wait_for_model": True}}
            ).json()
        )
