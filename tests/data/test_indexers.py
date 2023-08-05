import numpy

from nlpstack.data import Token, Vocabulary
from nlpstack.data.embeddings import PretrainedFasttextWordEmbedding
from nlpstack.data.indexers import PretrainedEmbeddingIndexer


def test_pretrained_fasttext_indexer() -> None:
    indexer = PretrainedEmbeddingIndexer(PretrainedFasttextWordEmbedding("tests/fixtures/models/fasttext.bin"))
    tokens = [Token("Hello"), Token("World")]
    # Currently, PretrainedEmbeddingIndexer does not use vocab.
    output = indexer(tokens, Vocabulary())

    assert set(output.keys()) == {"embeddings", "mask"}
    assert isinstance(output["embeddings"], numpy.ndarray)
    assert output["embeddings"].shape == (2, 100)
    assert output["mask"].tolist() == [True, True]
