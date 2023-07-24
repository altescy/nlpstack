from os import PathLike
from typing import Dict, Union, cast

import minato
import numpy

from nlpstack.common import cached_property

try:
    import fasttext
except ModuleNotFoundError:
    fasttext = None


class WordEmbedding:
    def __getitem__(self, word: str) -> numpy.ndarray:
        raise NotImplementedError

    def __contains__(self, word: str) -> bool:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class PretrainedWordEmbedding:
    @staticmethod
    def _read_embeddings_file(filename: Union[str, PathLike]) -> Dict[str, numpy.ndarray]:
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


class PretrainedFasttextEmbedding:
    def __init__(self, filename: Union[str, PathLike]) -> None:
        if fasttext is None:
            raise ModuleNotFoundError("fasttext is not installed")

        self._filename = filename

    @cached_property
    def fasttext(self) -> "fasttext.FastText":
        if fasttext is None:
            raise ModuleNotFoundError("fasttext is not installed")
        pretrained_filename = minato.cached_path(self._filename)
        return fasttext.load_model(str(pretrained_filename))

    def __getitem__(self, word: str) -> numpy.ndarray:
        return cast(numpy.ndarray, self.fasttext.get_word_vector(word))

    def __contains__(self, word: str) -> bool:
        return word in self.fasttext

    def get_output_dim(self) -> int:
        return int(self.fasttext.get_dimension())
