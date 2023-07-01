import numpy

from nlpstack.data import Token, Vocabulary
from nlpstack.data.indexers import PretrainedFasttextIndexer


def test_pretrained_fasttext_indexer() -> None:
    indexer = PretrainedFasttextIndexer("tests/fixtures/models/fasttext.bin")
    tokens = [Token("Hello"), Token("World")]
    # Currently, PretrainedFasttextIndexer does not use vocab.
    output = indexer(tokens, Vocabulary())

    assert set(output.keys()) == {"embeddings", "mask"}
    assert isinstance(output["embeddings"], numpy.ndarray)
    assert output["embeddings"].shape == (2, 100)
    assert output["mask"].tolist() == [True, True]
