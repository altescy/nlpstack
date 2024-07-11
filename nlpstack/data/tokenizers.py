"""
Tokenizers for NLPSTACK, which are used to split a text into tokens and also to join tokens into a text.

Example:
    >>> from nlpstack.data import SpacyTokenizer
    >>> tokenizer = SpacyTokenizer("en_core_web_sm", with_whitespace=True)
    >>> tokens = tokenizer.tokenize("It is a good day.")
    >>> detokenized_text = tokenizer.detokenize(tokens)
"""

from contextlib import suppress
from os import PathLike
from typing import Any, Callable, Generic, Iterator, List, NamedTuple, Optional, Sequence, TypeVar, Union

import minato
import numpy

from nlpstack.common import Pipeline, cached_property
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


_T_Fixtures = TypeVar("_T_Fixtures")


class Token(NamedTuple):
    """
    A token in a text.

    Parameters:
        surface: The surface form of the token.
        postag: The part-of-speech tag of the token. Defaults to `None`.
        lemma: The lemma of the token. Defaults to `None`.
        vector: The vector of the token. Defaults to `None`.
    """

    surface: str
    postag: Optional[str] = None
    lemma: Optional[str] = None
    vector: Optional[numpy.ndarray] = None


class Tokenizer(Pipeline[str, List[Token], _T_Fixtures], Generic[_T_Fixtures]):
    """
    A base class for tokenizers.
    """

    def tokenize(self, text: str) -> List[Token]:
        return next(self([text], batch_size=1, max_workers=1))

    def detokenize(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        raise NotImplementedError

    def tokenize_batch(self, batch: Sequence[str], fixtures: _T_Fixtures) -> List[List[Token]]:
        raise NotImplementedError

    def detokenize_batch(
        self, batch: Sequence[Union[Sequence[str], Sequence[Token]]], fixtures: _T_Fixtures
    ) -> List[str]:
        raise NotImplementedError

    def apply_batch(self, batch: Sequence[str], fixtures: _T_Fixtures) -> List[List[Token]]:
        return self.tokenize_batch(batch, fixtures)

    def tokenize_pipeline(self) -> Pipeline[str, List[Token], _T_Fixtures]:
        return self

    @cached_property
    def detokenize_pipeline(self) -> Pipeline[Union[Sequence[str], Sequence[Token]], str, Any]:
        return Pipeline.from_callable(self.detokenize_batch, self.fixtures)


class WhitespaceTokenizer(Tokenizer[None]):
    """
    A tokenizer that splits a text into tokens by whitespace.
    """

    fixtures = None

    def tokenize_batch(self, batch: Sequence[str], fixtures: None) -> List[List[Token]]:
        return [[Token(surface) for surface in text.split()] for text in batch]

    def detokenize(self, tokens: Union[Sequence[str], Sequence[Token]], **kwargs: Any) -> str:
        del kwargs
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return " ".join(tokens)


class CharacterTokenizer(Tokenizer[None]):
    """
    A tokenizer that splits a text into character tokens.
    """

    fixtures = None

    def tokenize_batch(self, batch: Sequence[str], fixtures: None) -> List[List[Token]]:
        return [[Token(surface) for surface in text] for text in batch]

    def detokenize(self, tokens: Union[Sequence[str], Sequence[Token]], **kwargs: Any) -> str:
        del kwargs
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return "".join(tokens)


class SpacyTokenizer(Tokenizer["SpacyTokenizer.Fixture"]):
    """
    A tokenizer that uses spaCy.

    Args:
        lang: The language of the tokenizer.
        with_whitespace: If `True`, each token can contain succeeding whitespaces.
            Please set `True` if you want reconstruct the original text from tokens
            by using `detokenize()` method. Defaults to `False`.
    """

    class Fixture(NamedTuple):
        nlp: "spacy.language.Language"

    def __init__(
        self,
        lang: str,
        with_whitespace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._lang = lang
        self._with_whitespace = with_whitespace

    def get_nlp(self) -> "spacy.language.Language":
        return spacy.load(self._lang)

    @cached_property
    def fixtures(self) -> "SpacyTokenizer.Fixture":  # type: ignore[override]
        return SpacyTokenizer.Fixture(self.get_nlp())

    def tokenize_batch(
        self,
        batch: Sequence[str],
        fixtures: "SpacyTokenizer.Fixture",
    ) -> List[List[Token]]:
        return [
            [
                Token(
                    t.text_with_ws if self._with_whitespace else t.text,
                    t.pos_,
                    t.lemma_,
                    vector=numpy.array(t.vector) if t.has_vector else None,
                )
                for t in fixtures.nlp(text)
            ]
            for text in batch
        ]

    def detokenize(self, tokens: Union[Sequence[str], Sequence[Token]]) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return "".join(tokens)


class PretrainedTransformerTokenizer(Tokenizer["PretrainedTransformerTokenizer.Fixture"]):
    """
    A tokenizer uses a model from Huggingface's transformers library.
    We take a model name as an argument, which will pass to `AutoTokenizer.from_pretrained`

    Args:
        pretrained_model_name: The name, path or URL of the pretrained model.
    """

    class Fixture(NamedTuple):
        tokenizer: "transformers.PreTrainedTokenizer"

    def __init__(
        self,
        pretrained_model_name: Union[str, PathLike],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pretrained_model_name = pretrained_model_name

    def get_tokenizer(self) -> "transformers.PreTrainedTokenizer":
        if transformers is None:
            raise ModuleNotFoundError("transformers is not installed.")
        pretrained_model_name = self._pretrained_model_name
        with suppress(FileNotFoundError):
            pretrained_model_name = minato.cached_path(pretrained_model_name)
        return transformers_cache.get_pretrained_tokenizer(pretrained_model_name)

    @cached_property
    def fixtures(self) -> "PretrainedTransformerTokenizer.Fixture":  # type: ignore[override]
        return PretrainedTransformerTokenizer.Fixture(self.get_tokenizer())

    def tokenize_batch(
        self,
        batch: Sequence[str],
        fixtures: "PretrainedTransformerTokenizer.Fixture",
    ) -> List[List[Token]]:
        return [[Token(t) for t in fixtures.tokenizer.tokenize(text)] for text in batch]

    def detokenize(  # type: ignore[override]
        self,
        tokens: Union[Sequence[str], Sequence[Token]],
        *,
        tokenizer: "transformers.PreTrainedTokenizer",
    ) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return str(tokenizer.convert_tokens_to_string(tokens))


class FugashiTokenizer(Tokenizer["FugashiTokenizer.Fixture"]):
    """
    A tokenizer that uses fugashi for Japanese text.

    Args:
        system_dictionary_path: The path or URL to the system dictionary.
        user_dictionary_path: The path or URL to the user dictionary.
        with_whitespace: If `True`, whitespaces between tokens are returned as `空白`
            tokens. Please set `True` if you want reconstruct the original text from
            tokens by using `detokenize()` method. Defaults to `False`.
    """

    class Fixture(NamedTuple):
        tagger: "fugashi.GenericTagger"
        parser: Callable[["fugashi.fugashi.Node"], Iterator[Token]]

    def __init__(
        self,
        system_dictionary_path: Optional[Union[str, PathLike]] = None,
        user_dictionary_path: Optional[Union[str, PathLike]] = None,
        with_whitespace: bool = False,
        **kwargs: Any,
    ) -> None:
        if fugashi is None:
            raise ModuleNotFoundError("fugashi is not installed.")

        super().__init__(**kwargs)
        self._system_dictionary_path = system_dictionary_path or "unidic-lite"
        self._user_dictionary_path = user_dictionary_path
        self._with_whitespace = with_whitespace

    def get_tagger(self) -> "fugashi.GenericTagger":
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

    def get_parser(self) -> Callable[["fugashi.fugashi.Node"], Iterator[Token]]:
        def parse_feature_for_ipadic(node: fugashi.fugashi.Node) -> Iterator[Token]:
            """
            Details about the ipadic parsed result:
            https://taku910.github.io/mecab/
            """
            if self._with_whitespace:
                for whitespace in node.white_space:
                    yield Token(whitespace, "空白", " ", vector=None)
            yield Token(
                surface=node.surface,
                postag=node.feature[0],
                lemma=None if node.feature[0] != "記号" and node.feature[6] == "*" else node.feature[6],
            )

        def parse_feature_for_unidic(node: fugashi.fugashi.Node) -> Iterator[Token]:
            """
            Details about the unidic parsed result:
            https://clrd.ninjal.ac.jp/unidic/faq.html
            """
            if self._with_whitespace:
                for whitespace in node.white_space:
                    yield Token(whitespace, "空白", " ", vector=None)
            yield Token(
                surface=node.surface,
                postag=node.feature[0],
                lemma=node.feature[7] if len(node.feature) >= 8 else None,
            )

        if "ipadic" in str(self._system_dictionary_path):
            return parse_feature_for_ipadic

        if "unidic" in str(self._system_dictionary_path):
            return parse_feature_for_unidic

        raise ValueError("system_dictionary_path must contain 'ipadic' or 'unidic'")

    @cached_property
    def fixtures(self) -> "FugashiTokenizer.Fixture":  # type: ignore[override]
        return FugashiTokenizer.Fixture(self.get_tagger(), self.get_parser())

    def tokenize_batch(
        self,
        batch: Sequence[str],
        fixtures: "FugashiTokenizer.Fixture",
    ) -> List[List[Token]]:
        return [[token for node in fixtures.tagger(text) for token in fixtures.parser(node)] for text in batch]

    def detokenize(self, tokens: Union[Sequence[str], Sequence[Token]], **kwargs: Any) -> str:
        tokens = [token.surface if isinstance(token, Token) else token for token in tokens]
        return "".join(tokens)
