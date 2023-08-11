import pytest
import torch

from nlpstack.torch.modules.seq2seq_decoders import LstmSeq2SeqDecoder, Seq2SeqDecoder, TransformerSeq2SeqDecoder


@pytest.mark.parametrize(
    "decoder",
    [
        LstmSeq2SeqDecoder(16, 16, 2),
        TransformerSeq2SeqDecoder(16, 2, 16, 4),
    ],
)
def test_seq2seq_decoder(decoder: Seq2SeqDecoder) -> None:
    inputs = torch.rand(2, 4, 16)
    mask = torch.ones(2, 4).bool()

    otputs, _ = decoder(inputs, inputs_mask=mask)
    assert otputs.shape == (2, 4, decoder.get_output_dim())

    if decoder.can_take_memory():
        memory = torch.randn(4, 8, 6)
        memory_mask = torch.BoolTensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        output = decoder(
            inputs=inputs,
            memory=memory,
            inputs_mask=mask,
            memory_mask=memory_mask,
        )
        assert output.shape == (4, 5, 6)
