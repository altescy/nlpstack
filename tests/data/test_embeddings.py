import numpy
import pytest

from nlpstack.data import (
    BagOfEmbeddingsTextEmbedding,
    MinhashWordEmbedding,
    PretrainedTransformerTextEmbedding,
    TextEmbedding,
)


class TestEmbedding:
    @pytest.mark.parametrize(
        "text_embedding",
        [
            BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="mean"),
            BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="max"),
            BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="hier", window_size=3),
            BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="mean", normalize=True),
        ],
    )
    def test_text_embedding(self, text_embedding: TextEmbedding) -> None:
        texts = [
            "this is a test sentence",
            "this is another test sentence",
            "this is a third test sentence",
        ]

        embeddings = numpy.array(list(text_embedding(texts)))
        assert embeddings.shape == (3, text_embedding.get_output_dim())
        assert not numpy.any(numpy.isinf(embeddings))
        assert not numpy.any(numpy.isnan(embeddings))

    @pytest.mark.parametrize(
        "text_embedding",
        [
            BagOfEmbeddingsTextEmbedding(MinhashWordEmbedding(64), pooling="mean"),
        ],
    )
    def test_text_embedding_work_in_multiprocess(self, text_embedding: TextEmbedding) -> None:
        texts = [
            "this is a test sentence",
            "this is another test sentence",
        ] * 10

        for _ in text_embedding(texts, batch_size=2, max_workers=2):
            pass
