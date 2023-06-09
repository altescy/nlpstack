import pytest
import torch

from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.seq2seq_encoders import (
    ComposeSeq2SeqEncoder,
    FeedForwardSeq2SeqEncoder,
    GruSeq2SeqEncoder,
    LstmSeq2SeqEncoder,
    PassThroughSeq2SeqEncoder,
    ResidualSeq2SeqEncoder,
    RnnSeq2SeqEncoder,
    Seq2SeqEncoder,
    TransformerSeq2SeqEncoder,
    WindowConcatEncoder,
)


@pytest.mark.parametrize(
    "seq2seq_encoder",
    [
        FeedForwardSeq2SeqEncoder(FeedForward(6, [6])),
        GruSeq2SeqEncoder(6, 3, 1, bidirectional=True),
        LstmSeq2SeqEncoder(6, 3, 1, bidirectional=True),
        PassThroughSeq2SeqEncoder(6),
        RnnSeq2SeqEncoder(6, 3, 1, bidirectional=True),
        TransformerSeq2SeqEncoder(6, 2, 6, 2, positional_encoding="sinusoidal"),
        ComposeSeq2SeqEncoder(
            [
                LstmSeq2SeqEncoder(6, 3, 1, bidirectional=True),
                FeedForwardSeq2SeqEncoder(FeedForward(6, [6])),
            ],
        ),
        WindowConcatEncoder(6, 2, output_dim=6),
        ResidualSeq2SeqEncoder(LstmSeq2SeqEncoder(6, 3, 1, bidirectional=True)),
    ],
)
def test_seq2seq_encoder(seq2seq_encoder: Seq2SeqEncoder) -> None:
    inputs = torch.randn(4, 5, 6)
    mask = torch.BoolTensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )

    output = seq2seq_encoder(inputs, mask)
    assert output.shape == (4, 5, 6)
