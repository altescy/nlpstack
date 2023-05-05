from random import randint
from typing import cast

import torch

from nlpstack.torch.modules.crf import ConditionalRandomField


def test_crf() -> None:
    batch_size = 4
    sequence_length = 10
    num_tags = 5

    inputs = torch.randn(batch_size, sequence_length, num_tags)
    tags = torch.tensor([[randint(0, num_tags - 1) for _ in range(sequence_length)] for _ in range(batch_size)])
    mask = cast(torch.BoolTensor, torch.ones(batch_size, sequence_length, dtype=torch.bool))

    crf = ConditionalRandomField(num_tags)

    log_likelihoods = crf(inputs, tags, mask)
    assert log_likelihoods.shape == (batch_size,)

    topk = 3
    decoded = crf.viterbi_decode(inputs, mask, topk=topk)
    assert len(decoded) == batch_size
    for paths in decoded:
        assert len(paths) == topk
        for path, score in paths:
            assert len(path) == sequence_length
            assert isinstance(score, float)
