import warnings
from contextlib import suppress
from logging import getLogger
from os import PathLike
from typing import Any, Iterator, List, Literal, Mapping, Optional, Union, cast

import minato
import numpy

from nlpstack.common import FileBackendMapping, cached_property, murmurhash3
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


logger = getLogger(__name__)


class WordEmbedding:
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
