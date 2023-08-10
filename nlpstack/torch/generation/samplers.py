from typing import Any, Generic, Tuple, TypeVar, cast

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
    ) -> Tuple[torch.Tensor, torch.LongTensor, None]:
        size = min(size, log_probs.size(-1))
        selected_log_probs, selected_indices = torch.topk(log_probs, size, dim=-1)
        return selected_log_probs, selected_indices, state


class MultinomialSampler(Sampler[None]):
    def __init__(
        self,
        temperature: float = 1.0,
        with_replacement: bool = False,
    ) -> None:
        self._temperature = temperature
        self._with_replacement = with_replacement

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> None:
        return None

    def sample_nodes(
        self,
        log_probs: torch.Tensor,
        size: int,
        state: None,
    ) -> Tuple[torch.Tensor, torch.LongTensor, None]:
        *dims, vocab_size = log_probs.size()

        flattened_log_probs = log_probs.view(-1, vocab_size)

        if self._temperature != 1.0:
            _probabilities = (flattened_log_probs / self._temperature).softmax(1)
        else:
            _probabilities = flattened_log_probs.exp()

        selected_indices = cast(
            torch.LongTensor,
            torch.multinomial(_probabilities, size, replacement=self._with_replacement),
        )
        selected_log_probs = flattened_log_probs.gather(1, selected_indices)

        selected_indices = cast(torch.LongTensor, selected_indices.view(*dims, size))
        selected_log_probs = selected_log_probs.view(*dims, size)

        return selected_log_probs, selected_indices, state
