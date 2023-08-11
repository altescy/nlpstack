from typing import Any, Generic, Optional, Tuple, TypeVar, cast

import torch

SamplerState = TypeVar("SamplerState")


class Sampler(Generic[SamplerState]):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> SamplerState:
        """
        Args:
            token_ids: Tensor of shape `(batch_size, beam_size, given_length)`.
            mask: Tensor of shape `(batch_size, beam_size, given_length)`.
        Returns:
            state: Initial state of the sampler.
        """
        raise NotImplementedError

    def sample_nodes(
        self,
        log_probs: torch.Tensor,
        size: int,
        state: SamplerState,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.LongTensor, SamplerState]:
        """
        Args:
            log_probs: Tensor of shape `(batch_size, num_nodes)`.
            size: Number of nodes to sample.
            state: State of the sampler.

        Returns:
            selected_log_probs: Tensor of shape `(batch_size, size)`.
            selected_indices: Tensor of shape `(batch_size, size)`.
            state: Updated state of the sampler.
        """
        raise NotImplementedError

    def sample_beams(
        self,
        log_probs: torch.Tensor,
        size: int,
        state: SamplerState,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.LongTensor, SamplerState]:
        size = min(size, log_probs.size(-1))
        selected_log_probs, selected_indices = torch.topk(log_probs, size, dim=-1)
        return selected_log_probs, selected_indices, state


class DeterministicSampler(Sampler[None]):
    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> None:
        return None

    def sample_nodes(
        self,
        log_probs: torch.Tensor,
        size: int,
        state: None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.LongTensor, None]:
        size = min(size, log_probs.size(-1))
        selected_log_probs, selected_indices = torch.topk(log_probs, size, dim=-1)
        return selected_log_probs, selected_indices, state


class MultinomialSampler(Sampler[None]):
    def __init__(
        self,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        with_replacement: bool = False,
    ) -> None:
        self._top_k = top_k
        self._top_p = top_p
        self._temperature = temperature
        self._with_replacement = with_replacement

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> None:
        return None

    def sample_nodes(
        self,
        log_probs: torch.Tensor,
        size: int,
        state: None,
        *,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        with_replacement: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.LongTensor, None]:
        top_k = self._top_k if top_k is None else top_k
        top_p = self._top_p if top_p is None else top_p
        temperature = self._temperature if temperature is None else temperature
        with_replacement = self._with_replacement if with_replacement is None else with_replacement

        if top_k is not None and not size <= top_k <= log_probs.size(-1):
            raise ValueError("top_k must be size <= top_k <= vocabulary size")
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be 0.0 <= top_p <= 1.0")

        *dims, vocab_size = log_probs.size()

        flattened_log_probs = log_probs.view(-1, vocab_size)

        if top_k is not None:
            _values, _indices = flattened_log_probs.topk(top_k, dim=-1)
            _mask = torch.zeros_like(flattened_log_probs).scatter(-1, _indices, 1.0)
            flattened_log_probs = flattened_log_probs.masked_fill(_mask == 0.0, -float("inf"))

        if self._temperature != 1.0:
            probabilities = (flattened_log_probs / temperature).softmax(1)
        else:
            probabilities = flattened_log_probs.exp()

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
            cumulative_probs = sorted_probs.cumsum(1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, float("-inf"))
            probabilities = sorted_probs.gather(1, sorted_indices).softmax(-1)

        if temperature > 0.0:
            selected_indices = cast(
                torch.LongTensor,
                torch.multinomial(probabilities, size, replacement=with_replacement),
            )
        else:
            selected_indices = probabilities.topk(size, dim=-1, sorted=True)[1]
        selected_log_probs = flattened_log_probs.gather(1, selected_indices)

        selected_indices = cast(torch.LongTensor, selected_indices.view(*dims, size))
        selected_log_probs = selected_log_probs.view(*dims, size)

        return selected_log_probs, selected_indices, state
