from contextlib import suppress
from os import PathLike
from typing import Callable, List, NamedTuple, Optional, Sequence, Union

import minato
import numpy

from nlpstack.common import cached_property
from nlpstack.transformers import cache as transformers_cache

try:
    import transformers
except ModuleNotFoundError:
    transformers = None


try:
    import spacy
except ModuleNotFoundError:
    spacy = None  # type: ignore[assignment]

try:
    import fugashi
except ModuleNotFoundError:
    fugashi = None


class Token(NamedTuple):
    surface: str
    postag: Optional[str] = None
    lemma: Optional[str] = None
    vector: Optional[numpy.ndarray] = None


class Tokenizer:
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError

    def decode(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(surface) for surface in text.split()]

    def decode(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return " ".join(tokens)


class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(surface) for surface in text]

    def decode(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return "".join(tokens)


class SpacyTokenizer(Tokenizer):
    def __init__(self, lang: str) -> None:
        self._lang = lang

    @cached_property
    def nlp(self) -> "spacy.language.Language":
        if spacy is None:
            raise ModuleNotFoundError("spacy is not installed.")
        return spacy.load(self._lang)

    def tokenize(self, text: str) -> List[Token]:
        doc = self.nlp(text)
        return [
            Token(
                t.text_with_ws,
                t.pos_,
                t.lemma_,
                vector=numpy.array(t.vector) if t.has_vector else None,
            )
            for t in doc
        ]

    def decode(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return "".join(tokens)


class PretrainedTransformerTokenizer(Tokenizer):
    def __init__(self, pretrained_model_name: Union[str, PathLike]) -> None:
        self._pretrained_model_name = pretrained_model_name

    @cached_property
    def tokenizer(self) -> "transformers.PreTrainedTokenizer":
        if transformers is None:
            raise ModuleNotFoundError("transformers is not installed.")
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.tokenizer.tokenize(text)  # type: ignore
        return [Token(t) for t in tokens]

    def decode(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return str(self.tokenizer.convert_tokens_to_string(tokens))


class FugashiTokenizer(Tokenizer):
    def __init__(
        self,
        system_dictionary_path: Optional[Union[str, PathLike]] = None,
        user_dictionary_path: Optional[Union[str, PathLike]] = None,
        with_whitespace: bool = False,
    ) -> None:
        self._system_dictionary_path = system_dictionary_path or "unidic-lite"
        self._user_dictionary_path = user_dictionary_path
        self._with_whitespace = with_whitespace

        self.tagger  # load tagger
        self.parse_feature  # load parse_feature

    @cached_property
    def tagger(self) -> "fugashi.GenericTagger":
        system_dictionary_path = self._system_dictionary_path
        user_dictionary_path = self._user_dictionary_path
        if system_dictionary_path == "ipadic":
            import ipadic

            system_dictionary_path = ipadic.DICDIR
        elif system_dictionary_path == "unidic":
            import unidic

            system_dictionary_path = unidic.DICDIR
        elif system_dictionary_path == "unidic-lite":
            import unidic_lite

            system_dictionary_path = unidic_lite.DICDIR

        options = ["-r /dev/null", f"-d {minato.cached_path(system_dictionary_path)}"]
        if user_dictionary_path:
            options.append(f"-u {minato.cached_path(user_dictionary_path)}")

        return fugashi.GenericTagger(" ".join(options))

    @cached_property
    def parse_feature(self) -> Callable[["fugashi.fugashi.Node"], Token]:
        def parse_feature_for_ipadic(node: fugashi.fugashi.Node) -> Token:
            """
            Details about the ipadic parsed result:
            https://taku910.github.io/mecab/
            """
            return Token(
                surface=node.white_space + node.surface if self._with_whitespace else node.surface,
                postag=node.feature[0],
                lemma=None if node.feature[0] != "記号" and node.feature[6] == "*" else node.feature[6],
            )

        def parse_feature_for_unidic(node: fugashi.fugashi.Node) -> Token:
            """
            Details about the unidic parsed result:
            https://clrd.ninjal.ac.jp/unidic/faq.html
            """
            return Token(
                surface=node.white_space + node.surface if self._with_whitespace else node.surface,
                postag=node.feature[0],
                lemma=node.feature[7] if len(node.feature) >= 8 else None,
            )

        if "ipadic" in str(self._system_dictionary_path):
            return parse_feature_for_ipadic

        if "unidic" in str(self._system_dictionary_path):
            return parse_feature_for_unidic

        raise ValueError("system_dictionary_path must contain 'ipadic' or 'unidic'")

    def tokenize(self, text: str) -> List[Token]:
        return [self.parse_feature(node) for node in self.tagger(text)]

    def decode(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return "".join(tokens)
