from typing import Any, Generic, Tuple, TypeVar

import torch

SamplerState = TypeVar("SamplerState")


class Sampler(Generic[SamplerState]):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_state(self, token_ids: torch.LongTensor) -> SamplerState:
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
    def init_state(self, log_probs: torch.Tensor) -> None:
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
