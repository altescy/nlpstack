import numpy
import pytest

from nlpstack.data import BagOfEmbeddingsTextEmbedding, MinhashWordEmbedding, TextEmbedding


@pytest.mark.parametrize(
    "text_embedding",
    [
        BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="mean"),
        BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="max"),
        BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="hier", window_size=3),
        BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="mean", normalize=True),
    ],
)
def test_text_embedding(text_embedding: TextEmbedding) -> None:
    texts = [
        "this is a test sentence",
        "this is another test sentence",
        "this is a third test sentence",
    ]

    embeddings = numpy.array(list(text_embedding(texts)))
    assert embeddings.shape == (3, text_embedding.get_output_dim())
    assert not numpy.any(numpy.isinf(embeddings))
    assert not numpy.any(numpy.isnan(embeddings))
