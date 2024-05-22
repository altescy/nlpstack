import math
from typing import List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F

from nlpstack.torch.modules.block_linear import BlockLinear


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class CausalConvolution(torch.nn.Conv1d):
    def __init__(self, kernel_size: int) -> None:
        self._padding = kernel_size - 1
        super().__init__(
            1,
            1,
            kernel_size,
            padding=self._padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = super().forward(x.unsqueeze(1)).squeeze(1)
        h = h[..., : -self._padding]
        return h


class SLSTMBlock(torch.nn.Module):
    class State(NamedTuple):
        prevhidden: torch.Tensor  # Shape: (batch_size, hidden_dim)
        memorycell: torch.Tensor  # Shape: (batch_size, hidden_dim)
        normalizer: torch.Tensor  # Shape: (batch_size, hidden_dim)
        stabilizer: torch.Tensor  # Shape: (batch_size, hidden_dim)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        use_conv: bool = False,
        kernel_size: int = 4,
        projection_factor: float = 4 / 3,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._head_dim = hidden_dim // num_heads
        self._projection_dim = int(hidden_dim * projection_factor)
        self._linear_z = torch.nn.Linear(input_dim, hidden_dim)
        self._linear_i = torch.nn.Linear(input_dim, hidden_dim)
        self._linear_f = torch.nn.Linear(input_dim, hidden_dim)
        self._linear_o = torch.nn.Linear(input_dim, hidden_dim)
        self._r_z = BlockLinear([(self._head_dim, self._head_dim)] * num_heads)
        self._r_i = BlockLinear([(self._head_dim, self._head_dim)] * num_heads)
        self._r_f = BlockLinear([(self._head_dim, self._head_dim)] * num_heads)
        self._r_o = BlockLinear([(self._head_dim, self._head_dim)] * num_heads)
        self._causal_conv = CausalConvolution(kernel_size) if use_conv else None
        self._up_projection = torch.nn.Linear(hidden_dim, 2 * self._projection_dim)
        self._down_projection = torch.nn.Linear(self._projection_dim, input_dim)
        self._layer_norm = torch.nn.LayerNorm(input_dim)
        self._group_norm = torch.nn.GroupNorm(num_heads, hidden_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        x: torch.Tensor,
        state: State,
    ) -> Tuple[torch.Tensor, State]:
        """
        Args:
            x: tensor of shape (batch_size, input_dim)
            state: tuple of hidden, memory, normalizer, and stabilizer states of shape
            (batch_size, hidden_dim) each

        Returns:
            tuple of the following tensors:
            - output tensor of shape (batch_size, hidden_dim)
            - tuple of hidden and memory states:
            - hidden state tensor of shape (batch_size, hidden_dim)
              - memory state of shape (batch_size, hidden_dim)
              - normalizer state of shape (batch_size, hidden_dim)
              - stabilizer state of shape (batch_size, hidden_dim)
        """
        (h_prev, c_prev, n_prev, m_prev) = state

        x_oz = self._layer_norm(x)
        x_fi = x_oz if self._causal_conv is None else swish(self._causal_conv(x_oz))

        z_tilda = self._linear_z(x_oz) + self._r_z(h_prev)
        i_tilda = self._linear_i(x_fi) + self._r_i(h_prev)
        f_tilda = self._linear_f(x_fi) + self._r_f(h_prev)
        o_tilda = self._linear_o(x_fi) + self._r_o(h_prev)

        zt = torch.tanh(z_tilda)  # Shape: (batch_size, hidden_dim)
        it = torch.exp(i_tilda)  # Shape: (batch_size, hidden_dim)
        ft = torch.exp(f_tilda)  # Shape: (batch_size, hidden_dim)
        ot = torch.sigmoid(o_tilda)  # Shape: (batch_size, hidden_dim)

        mt = torch.max(f_tilda + m_prev, i_tilda)  # Shape: (batch_size, hidden_dim)

        it = torch.exp(i_tilda - mt)  # Shape: (batch_size, hidden_dim)
        ft = torch.exp(f_tilda + m_prev - mt)  # Shape: (batch_size, hidden_dim)

        ct = ft * c_prev + it * zt  # Shape: (batch_size, hidden_dim)
        nt = ft * n_prev + it  # Shape: (batch_size, hidden_dim)

        h_tilda = ct / (nt + 1e-13)  # Shape: (batch_size, hidden_dim)
        ht = ot * h_tilda  # Shape: (batch_size, hidden_dim)

        ht_norm = self._group_norm(ht)

        s, g = torch.chunk(self._up_projection(ht_norm), 2, dim=-1)
        g = F.gelu(g)
        y = self._down_projection(s * g) + x

        return y, SLSTMBlock.State(ht, ct, nt, mt)

    def init_state(self, batch_size: int) -> State:
        """
        Returns:
            tuple of hidden, memory, normalizer, and stabilizer states of shape
            (batch_size, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)
            and (batch_size, hidden_dim) respectively
        """
        return SLSTMBlock.State(
            torch.zeros(batch_size, self._hidden_dim),
            torch.zeros(batch_size, self._hidden_dim),
            torch.zeros(batch_size, self._hidden_dim),
            torch.zeros(batch_size, self._hidden_dim),
        )


class MLSTMBlock(torch.nn.Module):
    class State(NamedTuple):
        memorycell: torch.Tensor  # Shape: (batch_size, num_heads, head_dim, head_dim)
        normalizer: torch.Tensor  # Shape: (batch_size, num_heads, head_dim)
        stabilizer: torch.Tensor  # Shape: (batch_size, num_heads, head_dim)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        use_conv: bool = False,
        kernel_size: int = 4,
        projection_factor: float = 2.0,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._head_dim = hidden_dim // num_heads
        self._projection_dim = int(input_dim * projection_factor)
        self._dim_norm = 1.0 / math.sqrt(hidden_dim)
        self._linear_q = torch.nn.Linear(self._projection_dim, hidden_dim)
        self._linear_k = torch.nn.Linear(self._projection_dim, hidden_dim)
        self._linear_v = torch.nn.Linear(self._projection_dim, hidden_dim)
        self._linear_i = torch.nn.Linear(self._projection_dim, num_heads)
        self._linear_f = torch.nn.Linear(self._projection_dim, num_heads)
        self._linear_o = torch.nn.Linear(self._projection_dim, hidden_dim)
        self._skip = torch.nn.Conv1d(self._projection_dim, hidden_dim, kernel_size=1, bias=False)
        self._causal_conv = CausalConvolution(kernel_size) if use_conv else None
        self._up_projection = torch.nn.Linear(input_dim, self._projection_dim + hidden_dim)
        self._down_projection = torch.nn.Linear(hidden_dim, input_dim)
        self._layer_norm = torch.nn.LayerNorm(input_dim)
        self._group_norm = torch.nn.GroupNorm(num_heads, hidden_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        x: torch.Tensor,
        state: State,
    ) -> Tuple[torch.Tensor, State]:
        """
        Args:
            x: input tensor of shape (batch_size, input_dim)
            state: tuple of memory, normalizer, and stabilizer states of shape
                (batch_size, num_heads, head_dim, head_dim), (batch_size, num_heads, head_dim)
                and (batch_size, num_heads) respectively

        Returns:
            tuple of the following tensors:
            - output tensor of shape (batch_size, hidden_dim)
            - tuple of hidden and memory states:
              - memory state of shape (batch_size, num_heads, hidden_dim, head_dim)
              - normalizer state of shape (batch_size, num_heads, head_dim)
              - stabilizer state of shape (batch_size, num_heads, head_dim)
        """
        batch_size = x.size(0)

        x_up = self._up_projection(self._layer_norm(x))  # Shape: (batch_size, projection_dim + hidden_dim)
        x_qkv, x_g = x_up.split([self._projection_dim, self._hidden_dim], dim=-1)

        x_v = x_qkv
        x_qk = x_v if self._causal_conv is None else swish(self._causal_conv(x_v))

        (C_prev, n_prev, m_prev) = state
        qt = (self._linear_q(x_qk)).view(
            batch_size, self._num_heads, self._head_dim
        )  # Shape: (batch_size, num_heads, head_dim)
        kt = (self._linear_k(x_qk) * self._dim_norm).view(
            batch_size, self._num_heads, self._head_dim
        )  # Shape: (batch_size, num_heads, head_dim)
        vt = (self._linear_v(x_v)).view(
            batch_size, self._num_heads, self._head_dim
        )  # Shape: (batch_size, num_heads, head_dim)

        i_tilda = self._linear_i(x_qkv)
        f_tilda = self._linear_f(x_qkv)
        o_tilda = self._linear_o(x_qkv)

        it = torch.exp(i_tilda)  # Shape: (batch_size, num_heads)
        ft = torch.exp(f_tilda)  # Shape: (batch_size, num_heads)
        ot = torch.sigmoid(o_tilda)  # Shape: (batch_size, hidden_dim)

        mt = torch.max(f_tilda + m_prev, i_tilda)  # Shape: (batch_size, num_heads)

        it = torch.exp(i_tilda - mt)  # Shape: (batch_size, num_heads)
        ft = torch.exp(f_tilda + m_prev - mt)  # Shape: (batch_size, num_heads)

        Ct = ft[:, :, None, None] * C_prev + it[  # Shape: (batch_size, num_heads, head_dim, head_dim)
            :, :, None, None
        ] * torch.einsum("bhi,bhj->bhij", vt, kt)
        nt = ft[:, :, None] * n_prev + it[:, :, None] * kt  # Shape: (batch_size, num_heads, head_dim)

        max_nqt = (nt * qt).sum(-1, keepdim=True).abs().clamp(min=1.0)  # Shape: (batch_size, num_heads, 1)
        h_tilde = ((Ct * qt.unsqueeze(2)).sum(3) / max_nqt).view(
            batch_size, self._hidden_dim
        )  # Shape: (batch_size, hidden_dim)
        ht = ot * h_tilde  # Shape: (batch_size, hidden_dim)

        ht_norm = self._group_norm(ht)

        s = ht_norm + self._skip(x_qk.unsqueeze(-1)).squeeze(-1)
        g = swish(x_g)
        y = self._down_projection(s * g) + x

        return y, MLSTMBlock.State(Ct, nt, mt)

    def init_state(self, batch_size: int) -> State:
        """
        Returns:
            tuple of memory, normalizer, and stabilizer states of shape
            (batch_size, num_heads, head_dim, head_dim), (batch_size, num_heads, head_dim)
            and (batch_size, num_heads) respectively
        """
        return MLSTMBlock.State(
            torch.zeros(batch_size, self._num_heads, self._head_dim, self._head_dim),
            torch.zeros(batch_size, self._num_heads, self._head_dim),
            torch.zeros(batch_size, self._num_heads),
        )


class XLSTM(torch.nn.Module):
    class State(NamedTuple):
        forward: Tuple[Union[SLSTMBlock.State, MLSTMBlock.State], ...]
        backward: Optional[Tuple[Union[SLSTMBlock.State, MLSTMBlock.State], ...]] = None

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        layers: Sequence[Literal["s", "m"]],
        use_conv: bool = True,
        kernel_size: int = 4,
        projection_factors: Tuple[float, float] = (2.0, 4 / 3),
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._input_dim = input_dim

        m_projection_factor, s_projection_factor = projection_factors

        m_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "use_conv": use_conv,
            "kernel_size": kernel_size,
            "projection_factor": m_projection_factor,
        }
        s_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "use_conv": use_conv,
            "kernel_size": kernel_size,
            "projection_factor": s_projection_factor,
        }

        self._forward_blocks: Sequence[Union[SLSTMBlock, MLSTMBlock]] = torch.nn.ModuleList(  # type: ignore[assignment]
            [
                MLSTMBlock(**m_params) if layer == "m" else SLSTMBlock(**s_params)  # type: ignore[arg-type]
                for layer in layers
            ]
        )
        self._backward_blocks: Optional[Sequence[Union[SLSTMBlock, MLSTMBlock]]] = (
            torch.nn.ModuleList(  # type: ignore[assignment]
                [
                    MLSTMBlock(**m_params) if layer == "m" else SLSTMBlock(**s_params)  # type: ignore[arg-type]
                    for layer in layers
                ]
            )
            if bidirectional
            else None
        )
        self._bidirectional_projections = (
            torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.LayerNorm(input_dim * 2),
                        torch.nn.Linear(input_dim * 2, input_dim),
                        torch.nn.GELU(),
                    )
                    for _ in layers
                ]
            )
            if bidirectional
            else None
        )
        self._dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[State] = None,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, State]:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, input_dim)
            state: tuple of memory, normalizer, and stabilizer states of shape
            mask: mask tensor of shape (batch_size, seq_len)

        Returns:
            tuple of output tensor of shape (batch_size, seq_len, hidden_dim) and
            XLSTM state
        """
        batch_size, sequence_length, _ = x.size()
        if state is None:
            state = XLSTM.State(
                forward=tuple(block.init_state(batch_size) for block in self._forward_blocks),
                backward=None
                if self._backward_blocks is None
                else tuple(block.init_state(batch_size) for block in self._backward_blocks),
            )

        new_forward_states: List[Union[SLSTMBlock.State, MLSTMBlock.State]] = []
        new_backward_states: Optional[List[Union[SLSTMBlock.State, MLSTMBlock.State]]] = (
            [] if self._backward_blocks is not None else None
        )

        for i in range(len(self._forward_blocks)):
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

            new_x: List[torch.Tensor] = []

            # forward processing
            forward_block = self._forward_blocks[i]
            forward_state = state.forward[i]
            for t in range(sequence_length):
                xt, forward_state = forward_block(x[:, t, :], forward_state)
                new_x.append(xt)
                new_forward_states.append(forward_state)

            # backward processing
            if self._backward_blocks is not None:
                assert state.backward is not None
                assert new_backward_states is not None
                assert self._bidirectional_projections is not None
                backward_block = self._backward_blocks[i]
                backward_state = state.backward[i]
                bidirectional_projection = self._bidirectional_projections[i]
                for t in range(sequence_length - 1, -1, -1):
                    xt, backward_state = backward_block(x[:, t, :], backward_state)
                    xt = torch.cat([xt, new_x[t]], dim=-1)
                    xt = bidirectional_projection(xt) + x[:, t, :]
                    new_x[t] = xt
                    new_backward_states.append(backward_state)

            x = torch.stack(new_x, dim=1)

            if self._dropout is not None:
                x = self._dropout(x)

        return cast(torch.FloatTensor, x), XLSTM.State(
            forward=tuple(new_forward_states),
            backward=None if new_backward_states is None else tuple(new_backward_states),
        )


if __name__ == "__main__":
    model = XLSTM(10, 16, 4, ("m", "s", "m", "s"), bidirectional=True)
    x = torch.randn(8, 5, 10)
    m = torch.randn(8, 5).bool()
    y, state = model(x, mask=m)
    print(y.shape)
