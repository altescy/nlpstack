from typing import Any, Dict, Set

import pytest

from nlpstack.data import SpacyTokenizer
from nlpstack.tasks.keyphrase_extraction.util import iter_candidate_phrases


@pytest.mark.parametrize(
    "text, params, expected",
    [
        (
            "The quick brown fox jumps over the lazy dog.",
            {"postag_pattern": r"^(<ADJ:[A-Z]+>)*(<NOUN:[A-Z]+>)+$"},
            {"fox", "brown fox", "quick brown fox", "dog", "lazy dog"},
        ),
    ],
)
def test_iter_candidate_phrases(
    text: str,
    params: Dict[str, Any],
    expected: Set[str],
) -> None:
    tokenizer = SpacyTokenizer("en_core_web_sm", with_whitespace=True)
    tokens = tokenizer.tokenize(text)
    print(tokens)
    phrases = {
        tokenizer.detokenize(phrase_tokens).strip() for phrase_tokens in iter_candidate_phrases(tokens, **params)
    }
    assert phrases == expected
