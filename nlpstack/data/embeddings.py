from os import PathLike
from typing import Any, Iterator, List, Mapping, Union, cast

import minato
import numpy

from nlpstack.common import cached_property, murmurhash3

try:
    import fasttext
except ModuleNotFoundError:
    fasttext = None


class WordEmbedding:
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getitem__(self, word: str) -> numpy.ndarray:
        raise NotImplementedError

    def __contains__(self, word: str) -> bool:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class MinhashWordEmbedding(WordEmbedding):
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


class PretrainedWordEmbedding(WordEmbedding):
    @staticmethod
    def _read_embeddings_file(filename: Union[str, PathLike]) -> Mapping[str, numpy.ndarray]:
        with minato.open(filename, decompress="auto") as txtfile:
            embeddings = {line.split()[0]: numpy.array(line.split()[1:], dtype=float) for line in txtfile}
        assert (
            len(set(len(embedding) for embedding in embeddings.values())) == 1
        ), "All embeddings must have the same dimension."
        return embeddings

    def __init__(self, filename: Union[str, PathLike]) -> None:
        self._embeddings = self._read_embeddings_file(filename)

    def __getitem__(self, word: str) -> numpy.ndarray:
        return self._embeddings[word]

    def __contains__(self, word: str) -> bool:
        return word in self._embeddings

    def get_output_dim(self) -> int:
        return len(next(iter(self._embeddings.values())))


class PretrainedFasttextEmbedding(WordEmbedding):
    def __init__(self, filename: Union[str, PathLike], allow_unknown_words: bool = True) -> None:
        if fasttext is None:
            raise ModuleNotFoundError("fasttext is not installed")

        self._filename = filename
        self._allow_unknown_words = allow_unknown_words

    @cached_property
    def fasttext(self) -> "fasttext.FastText":
        if fasttext is None:
            raise ModuleNotFoundError("fasttext is not installed")
        pretrained_filename = minato.cached_path(self._filename)
        return fasttext.load_model(str(pretrained_filename))

    def __getitem__(self, word: str) -> numpy.ndarray:
        return cast(numpy.ndarray, self.fasttext.get_word_vector(word))

    def __contains__(self, word: str) -> bool:
        return self._allow_unknown_words or (word in self.fasttext)

    def get_output_dim(self) -> int:
        return int(self.fasttext.get_dimension())
