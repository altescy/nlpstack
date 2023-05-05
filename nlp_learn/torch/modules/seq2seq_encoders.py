from typing import cast

import torch


class Seq2SeqEncoder(torch.nn.Module):
    def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class LstmSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._lstm = torch.nn.LSTM(  # type: ignore[no-untyped-call]
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        """
        :param token_ids: (batch_size, seq_len, input_dim)
        :param mask: (batch_size, seq_len)
        :return: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = token_ids.size()
        sorted_seq_len = mask.sum(dim=1)
        sorted_seq_len, sorted_idx = sorted_seq_len.sort(descending=True)
        _, unsorted_idx = sorted_idx.sort()
        sorted_token_ids = token_ids[sorted_idx]
        packed_token_ids = torch.nn.utils.rnn.pack_padded_sequence(sorted_token_ids, sorted_seq_len, batch_first=True)
        packed_output, _ = self._lstm(packed_token_ids)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[unsorted_idx]
        return cast(torch.FloatTensor, output)

    def get_input_dim(self) -> int:
        return self._lstm.input_size

    def get_output_dim(self) -> int:
        return self._lstm.hidden_size * 2 if self._lstm.bidirectional else self._lstm.hidden_size
