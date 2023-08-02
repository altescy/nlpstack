from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F


class CausalTransformerDecoderLayer(torch.nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        use_cross_attention: bool = False,
        device: Optional[torch.device] = None,
        dtype: Any = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batch_first = batch_first
        self.use_cross_attention = use_cross_attention
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = (
            torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
            if use_cross_attention
            else None
        )
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if use_cross_attention else None
        self.norm3 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout) if use_cross_attention else None
        self.dropout3 = torch.nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = torch.nn.modules.transformer._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)  # type: ignore[no-untyped-call]

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_is_causal: bool = False,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), cache=cache)
            if memory is not None:
                if not self.use_cross_attention:
                    raise ValueError("memory is not None but use_cross_attention is False")
                assert self.norm2 is not None
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, cache=cache))
            if memory is not None:
                if not self.use_cross_attention:
                    raise ValueError("memory is not None but use_cross_attention is False")
                assert self.norm2 is not None
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = x
        k = v = x if cache is None else torch.cat([cache, x], dim=1 if self.batch_first else 0)
        x = self.self_attn(
            q,
            k,
            v,
            is_causal=True,
            need_weights=False,
        )[0]
        return cast(torch.Tensor, self.dropout1(x))

    # multihead attention block
    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        assert self.multihead_attn is not None
        assert self.dropout2 is not None
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask if not is_causal else None,
            key_padding_mask=key_padding_mask if not is_causal else None,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return cast(torch.Tensor, self.dropout2(x))

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return cast(torch.Tensor, self.dropout3(x))


class CausalTransformerDecoder(torch.nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        decoder_layer: CausalTransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)  # type: ignore[no-untyped-call]
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_first = next(iter(self.layers)).batch_first
        sequence_dim = 1 if batch_first else 0
        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache should be None in training mode")
            for layer in self.layers:
                output = layer(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_is_causal=memory_is_causal,
                )
            return output, None

        new_token_cache: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                memory,
                memory_mask=memory_mask,
                memory_is_causal=memory_is_causal,
                cache=cache[i - 1] if cache is not None and i > 0 else None,
            )
            if cache is not None:
                new_token_cache.append(torch.cat([cache[i], output], dim=sequence_dim))
            else:
                new_token_cache.append(output)

        new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache
