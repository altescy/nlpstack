import pytest
import torch

from nlpstack.torch.modules.seq2vec_encoders import (
    BagOfEmbeddings,
    CnnEncoder,
    ConcatSeq2VecEncoder,
    SelfAttentiveSeq2VecEncoder,
    Seq2VecEncoder,
    TokenPooler,
)


@pytest.mark.parametrize(
    "seq2vec_encoder",
    [
        BagOfEmbeddings(6, "mean"),
        BagOfEmbeddings(6, "max"),
        BagOfEmbeddings(6, "sum"),
        CnnEncoder(6, 3, (2, 3)),
        TokenPooler(6, (0, -1), 6),
        ConcatSeq2VecEncoder([BagOfEmbeddings(6, "mean"), BagOfEmbeddings(6, "max")], output_dim=6),
        SelfAttentiveSeq2VecEncoder(6),
    ],
)
def test_seq2vec_encoders(seq2vec_encoder: Seq2VecEncoder) -> None:
    inputs = torch.randn(4, 5, 6)
    mask = torch.BoolTensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )

    output = seq2vec_encoder(inputs, mask)
    assert output.shape == (4, 6)
    assert not torch.isnan(output).any()

    if any(p.requires_grad for p in seq2vec_encoder.parameters()):
        loss = output.sum()
        loss.backward()
        assert all(p.grad is not None for p in seq2vec_encoder.parameters() if p.requires_grad)
        assert all(
            not p.grad.isnan().any() for p in seq2vec_encoder.parameters() if p.requires_grad and p.grad is not None
        )
