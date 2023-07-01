from typing import Any, Dict, cast

from nlpstack.data import Collator, Instance, Token, Vocabulary
from nlpstack.data.fields import TextField
from nlpstack.data.indexers import SingleIdTokenIndexer


def test_text_field() -> None:
    texts = [
        [Token("hello"), Token("world")],
        [Token("this"), Token("is"), Token("a"), Token("test")],
    ]

    vocab = Vocabulary(pad_token={"tokens": "@@PADDING@@"}, special_tokens={"tokens": {"@@PADDING@@"}})
    indexers = {"tokens": SingleIdTokenIndexer()}
    for indexer in indexers.values():
        indexer.build_vocab(vocab, texts)

    instances = [Instance(text=TextField(text, vocab, indexers)) for text in texts]

    arrays = Collator()(instances)

    text = cast(Dict[str, Any], arrays["text"])

    assert isinstance(arrays, dict)
    assert set(text.keys()) == {"tokens"}
    assert set(text["tokens"].keys()) == {"token_ids", "mask"}
    assert text["tokens"]["token_ids"].shape == (2, 4)
    assert text["tokens"]["mask"].shape == (2, 4)

    token_ids = text["tokens"]["token_ids"]
    mask = text["tokens"]["mask"]

    flattened_token_ids = token_ids[mask].tolist()
    flattened_mask = mask[mask].tolist()
    assert flattened_token_ids == [
        vocab.get_index_by_token("tokens", token.surface) for text in texts for token in text
    ]
    assert flattened_mask == [True] * sum(len(text) for text in texts)
