from typing import Any, Callable, List, NamedTuple, Optional, Protocol, Tuple, TypeVar, cast

import torch

from .samplers import DeterministicSampler, Sampler


class StepStateInterface(Protocol):
    def update(self, backpointer: torch.LongTensor) -> None:
        """
        Args:
            backpointer: Tensor of shape `(batch_size, beam_size)`.
        """
        ...


StepState = TypeVar("StepState", bound=StepStateInterface)


class BeamSearchOutput(NamedTuple):
    """Output of beam search.

    Attributes:
        token_ids: Tensor of shape `(batch_size, beam_size, max_steps)`.
        mask: Tensor of shape `(batch_size, beam_size, max_steps)`.
        scores: Tensor of shape `(batch_size, beam_size)`.
    """

    token_ids: torch.LongTensor
    mask: torch.BoolTensor
    scores: torch.Tensor


class BeamSearch:
    def __init__(
        self,
        max_steps: int = 50,
        beam_size: int = 10,
        sampling_size_per_beam: Optional[int] = None,
        sampler: Optional[Sampler] = None,
    ) -> None:
        self._max_steps = max_steps
        self._beam_size = beam_size
        self._sampling_size_per_beam = sampling_size_per_beam or beam_size
        self._sampler = sampler or DeterministicSampler()
        self._eos_index: Optional[int] = None

    def setup(self, *args: Any, eos_index: int, **kwargs: Any) -> None:
        self._eos_index = eos_index
        self._sampler.setup(*args, eos_inde=eos_index, **kwargs)

    @torch.no_grad()
    def search(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        state: StepState,
        step: Callable[[torch.LongTensor, StepState], Tuple[torch.Tensor, StepState]],
    ) -> BeamSearchOutput:
        """
        Args:
            token_ids: Tensor of shape `(batch_size, initial_length)`.
            mask: Tensor of shape `(batch_size, initial_length)`.
            state: State of the step function.
            step: Step function that takes `token_ids` and `state` as inputs and
                returns a tuple of `(log_probs, state)`. `log_probs` is a tensor of
                shape `(batch_size, vocab_size)`.

        Returns:
            Output of beam search with attributes `token_ids` and `scores`.
        """

        beam_size = self._beam_size
        batch_size = token_ids.size(0)
        sampling_size_per_node = beam_size

        initial_token_ids = token_ids
        initial_mask = mask
        initial_lengths = initial_mask.sum(dim=1)
        min_initial_length = int(initial_lengths.min().item())
        max_initial_length = int(initial_lengths.max().item())

        sampler_state = self._sampler.init_state(token_ids)
        cumulated_log_probs = torch.zeros((batch_size, beam_size), device=token_ids.device)
        backpointer = cast(torch.LongTensor, torch.zeros((batch_size, beam_size), dtype=torch.long))

        # Shape: (batch_size, beam_size, min_initial_length)
        token_ids = cast(
            torch.LongTensor, token_ids[:, :min_initial_length].unsqueeze(1).repeat_interleave(beam_size, dim=1)
        )
        # Shape: (batch_size, beam_size, min_initial_length)
        mask = cast(torch.BoolTensor, mask[:, :min_initial_length].unsqueeze(1).repeat_interleave(beam_size, dim=1))
        state.update(backpointer)

        predictions: List[torch.Tensor] = list(token_ids.unbind(dim=2))
        backpointers: List[torch.Tensor] = list(torch.zeros_like(token_ids).unbind(dim=2))

        for timestep in range(min_initial_length, self._max_steps):
            if self._eos_index is not None and (token_ids == self._eos_index).all():
                break

            # Shape: (batch_size, beam_size, vocab_size)
            log_probs, state = step(token_ids, state)
            (
                top_log_probs,  # Shape: (batch_size, beam_size, sampling_size_per_node)
                top_next_token_ids,  # Shape: (batch_size, beam_size, sampling_size_per_node)
                sampler_state,
            ) = self._sampler.sample_nodes(log_probs, sampling_size_per_node, sampler_state)

            # Shape: (batch_size, beam_size * sampling_size_per_node)
            expanded_cumulated_log_probs = (top_log_probs + cumulated_log_probs.unsqueeze(2)).view(batch_size, -1)

            (
                beam_log_probs,  # Shape: (batch_size, beam_size)
                beam_indices,  # Shape: (batch_size, beam_size)
                sampler_state,
            ) = self._sampler.sample_beams(expanded_cumulated_log_probs, beam_size, sampler_state)

            cumulated_log_probs = beam_log_probs

            # Shape: (batch_size, beam_size)
            new_token_ids = top_next_token_ids.view(batch_size, -1).gather(1, beam_indices)

            if timestep < max_initial_length:
                new_token_ids = torch.where(
                    initial_mask[:, timestep : timestep + 1].expand(batch_size, beam_size),
                    initial_token_ids[:, timestep : timestep + 1].expand(batch_size, beam_size),
                    new_token_ids,
                )
                cumulated_log_probs = torch.where(
                    initial_mask[:, timestep : timestep + 1].expand(batch_size, beam_size),
                    torch.zeros_like(cumulated_log_probs),
                    cumulated_log_probs,
                )
                beam_indices = cast(
                    torch.LongTensor,
                    torch.where(
                        initial_mask[:, timestep : timestep + 1].expand(batch_size, beam_size),
                        torch.zeros_like(beam_indices),
                        beam_indices,
                    ),
                )

            predictions.append(new_token_ids)

            # Shape: (batch_size, beam_size, 1)
            token_ids = cast(torch.LongTensor, new_token_ids.unsqueeze(2))

            # Shape: (batch_size, beam_size)
            backpointer = cast(
                torch.LongTensor,
                torch.divide(beam_indices, sampling_size_per_node, rounding_mode="trunc").long(),
            )

            state.update(backpointer)
            backpointers.append(backpointer)

        # Shape: (batch_size, beam_size)
        final_scores = cumulated_log_probs
        # Shape: (batch_size, beam_size, max_steps)
        final_token_ids = torch.cat(list(reversed(self._reconstruct_sequences(predictions, backpointers))), 2)

        sorted_final_scores, sorted_final_indices = final_scores.sort(dim=1, descending=True)
        sorted_final_token_ids = final_token_ids.gather(
            1, sorted_final_indices.unsqueeze(-1).expand_as(final_token_ids)
        )
        sorted_final_mask = (
            torch.ones_like(sorted_final_token_ids, dtype=torch.bool)
            if self._eos_index is None
            else (sorted_final_token_ids != self._eos_index)
        )

        return BeamSearchOutput(
            cast(torch.LongTensor, sorted_final_token_ids),
            cast(torch.BoolTensor, sorted_final_mask),
            sorted_final_scores,
        )

    @staticmethod
    def _reconstruct_sequences(predictions: List[torch.Tensor], backpointers: List[torch.Tensor]) -> List[torch.Tensor]:
        # Shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        if not backpointers:
            return reconstructed_predictions

        # Shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # Shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # Shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep].gather(1, cur_backpointers)

        # Shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        return reconstructed_predictions
