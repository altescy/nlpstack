from typing import Iterator, Optional, Pattern, Sequence, Tuple

from nlpstack.data import Token


def iter_candidate_phrases(
    tokens: Sequence[Token],
    ngram_range: Tuple[int, int] = (1, 3),
    postag_pattern: Optional[Pattern] = None,
) -> Iterator[Tuple[Token, ...]]:
    def _is_acceptable_phrase(phrase: Sequence[Token]) -> bool:
        if postag_pattern is None:
            return True
        postags = "".join(f"<{token.postag or '__NULL__'}>" for token in phrase)
        return bool(postag_pattern.match(postags))

    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            phrase = tuple(tokens[i : i + n])
            if _is_acceptable_phrase(phrase):
                yield phrase
