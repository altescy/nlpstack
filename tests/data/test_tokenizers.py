import pytest

from nlpstack.data.tokenizers import (
    CharacterTokenizer,
    FugashiTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    Tokenizer,
    WhitespaceTokenizer,
)


class TestTokenizer:
    @pytest.mark.parametrize(
        "tokenizer",
        [
            CharacterTokenizer(),
            FugashiTokenizer(),
            PretrainedTransformerTokenizer("bert-base-uncased"),
            SpacyTokenizer("en_core_web_sm"),
            WhitespaceTokenizer(),
        ],
    )
    def test_tokenizer_can_work_in_multiprocess(self, tokenizer: Tokenizer) -> None:
        texts = ["This is a test.", "This is another test."] * 10
        for _ in tokenizer(texts, batch_size=2, max_workers=2):
            pass
