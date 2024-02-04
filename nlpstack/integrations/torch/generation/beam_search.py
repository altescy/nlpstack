from functools import lru_cache
from typing import Any, Callable, List, NamedTuple, Optional, Protocol, Tuple, TypeVar, Union, cast

import torch

from .constraints import Constraint, MultiConstraint
from .samplers import MultinomialSampler, Sampler
from .scorers import BeamScorer, SequenceLogProbabilityScorer

StepStateSelf = TypeVar("StepStateSelf", bound="StepStateInterface")


class StepStateInterface(Protocol):
    def update(self: StepStateSelf, backpointer: torch.LongTensor) -> StepStateSelf:
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
        sampling_size_per_node: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        scorer: Optional[BeamScorer] = None,
        constraint: Optional[Union[Constraint, List[Constraint]]] = None,
    ) -> None:
        if isinstance(constraint, list):
            constraint = cast(Constraint, MultiConstraint(constraint))

        self._max_steps = max_steps
        self._beam_size = beam_size
        self._sampling_size_per_node = sampling_size_per_node or beam_size
        self._sampler = sampler or MultinomialSampler()
        self._scorer = scorer or SequenceLogProbabilityScorer()
        self._constraint = constraint
        self._eos_index: Optional[int] = None

    def setup(
        self,
        *args: Any,
        eos_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._eos_index = eos_index
        self._sampler.setup(*args, eos_index=eos_index, **kwargs)
        self._scorer.setup(*args, eos_index=eos_index, **kwargs)
        if self._constraint is not None:
            self._constraint.setup(*args, eos_index=eos_index, **kwargs)

    @torch.no_grad()
    def search(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        state: StepState,
        step: Callable[[torch.LongTensor, StepState], Tuple[torch.Tensor, StepState]],
        *,
        beam_size: Optional[int] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
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

        beam_size = self._beam_size if beam_size is None else beam_size
        batch_size = token_ids.size(0)
        sampling_size_per_node = self._sampling_size_per_node
        max_steps = self._max_steps if max_steps is None else max_steps

        sampler = self._sampler
        scorer = self._scorer
        constraint = self._constraint

        initial_token_ids = token_ids
        initial_mask = mask
        initial_lengths = initial_mask.sum(dim=1)
        min_initial_length = int(initial_lengths.min().item())
        max_initial_length = int(initial_lengths.max().item())

        backpointer = cast(torch.LongTensor, torch.zeros((batch_size, beam_size), dtype=torch.long))

        # Shape: (batch_size, beam_size, min_initial_length)
        token_ids = cast(
            torch.LongTensor, token_ids[:, :min_initial_length].unsqueeze(1).repeat_interleave(beam_size, dim=1)
        )
        # Shape: (batch_size, beam_size, min_initial_length)
        mask = cast(torch.BoolTensor, mask[:, :min_initial_length].unsqueeze(1).repeat_interleave(beam_size, dim=1))
        # Shape: (batch_size, beam_size)
        last_token_ids = cast(torch.LongTensor, token_ids[:, :, -1])
        # Shape: (batch_size, beam_size, max_initial_length - min_initial_length)
        rest_token_ids = cast(
            torch.LongTensor, initial_token_ids[:, min_initial_length:].unsqueeze(1).repeat_interleave(beam_size, dim=1)
        )
        # Shape: (batch_size, beam_size, max_initial_length - min_initial_length)
        rest_mask = cast(
            torch.BoolTensor, initial_mask[:, min_initial_length:].unsqueeze(1).repeat_interleave(beam_size, dim=1)
        )

        state = state.update(backpointer)
        sampler_state = sampler.init_state(token_ids, mask)
        scorer_state = scorer.init_state(token_ids, mask)
        constraint_state = constraint.init_state(token_ids, mask, rest_token_ids, rest_mask) if constraint else None

        predictions: List[torch.Tensor] = list(token_ids.unbind(dim=2))
        backpointers: List[torch.Tensor] = list(torch.zeros_like(token_ids).unbind(dim=2))

        @lru_cache
        def get_eos_mask(vocab_size: int) -> torch.BoolTensor:
            """Return one-hot mask for EOS token."""
            assert self._eos_index is not None
            return cast(torch.BoolTensor, (torch.arange(vocab_size) == self._eos_index))

        for timestep in range(min_initial_length, max_steps):
            if self._eos_index is not None and (last_token_ids == self._eos_index).all():
                break

            # Shape: (batch_size, beam_size, vocab_size)
            log_probs, state = step(token_ids, state)

            if self._constraint is not None:
                # Shape: (batch_size, beam_size, vocab_size)
                log_probs = self._constraint.apply(constraint_state, log_probs)

            # Enforce to sample eos token if last token is eos.
            if self._eos_index is not None:
                # Shape: (batch_size, beam_size, 1)
                beam_mask = (last_token_ids == self._eos_index).unsqueeze(2)
                # Shape: (1, vocab_size)
                eos_mask = get_eos_mask(log_probs.size(2)).unsqueeze(0)
                # Shape: (batch_size, beam_size, vocab_size)
                log_probs = log_probs.masked_fill(beam_mask * eos_mask, 0.0)
                log_probs = log_probs.masked_fill(beam_mask * ~eos_mask, float("-inf"))

            (
                top_log_probs,  # Shape: (batch_size, beam_size, sampling_size_per_node)
                top_next_token_ids,  # Shape: (batch_size, beam_size, sampling_size_per_node)
                sampler_state,
            ) = self._sampler.sample_nodes(log_probs, sampling_size_per_node, sampler_state, **kwargs)

            # Shape: (batch_size, beam_size * sampling_size_per_node)
            top_scores = scorer.score(scorer_state, top_log_probs).view(batch_size, -1)

            (
                beam_scores,  # Shape: (batch_size, beam_size)
                beam_indices,  # Shape: (batch_size, beam_size)
                sampler_state,
            ) = self._sampler.sample_beams(top_scores, beam_size, sampler_state, **kwargs)

            # Shape: (batch_size, beam_size)
            last_token_ids = cast(torch.LongTensor, top_next_token_ids.view(batch_size, -1).gather(1, beam_indices))

            if timestep < max_initial_length:
                last_token_ids = cast(
                    torch.LongTensor,
                    torch.where(
                        initial_mask[:, timestep : timestep + 1].expand(batch_size, beam_size),
                        initial_token_ids[:, timestep : timestep + 1].expand(batch_size, beam_size),
                        last_token_ids,
                    ),
                )
                beam_indices = cast(
                    torch.LongTensor,
                    torch.where(
                        initial_mask[:, timestep : timestep + 1].expand(batch_size, beam_size),
                        torch.zeros_like(beam_indices),
                        beam_indices,
                    ),
                )

            predictions.append(last_token_ids)

            # Shape: (batch_size, beam_size, 1)
            token_ids = cast(torch.LongTensor, last_token_ids.unsqueeze(2))

            # Shape: (batch_size, beam_size)
            backpointer = cast(
                torch.LongTensor,
                torch.divide(beam_indices, sampling_size_per_node, rounding_mode="trunc").long(),
            )
            backpointers.append(backpointer)

            state = state.update(backpointer)
            scorer_state = scorer.update_state(scorer_state, beam_scores, last_token_ids, backpointer)

            if constraint:
                constraint_state = constraint.update_state(constraint_state, last_token_ids, backpointer)

        # Shape: (batch_size, beam_size, max_steps)
        final_token_ids = cast(
            torch.LongTensor, torch.cat(list(reversed(self._reconstruct_sequences(predictions, backpointers))), 2)
        )
        # Shape: (batch_size, beam_size, max_steps)
        final_mask = cast(
            torch.BoolTensor,
            torch.ones_like(final_token_ids, dtype=torch.bool)
            if self._eos_index is None
            else (final_token_ids != self._eos_index),
        )
        # Shape: (batch_size, beam_size)
        final_scores = scorer.finalize(scorer_state, final_token_ids, final_mask)

        sorted_final_scores, sorted_final_indices = final_scores.sort(dim=1, descending=True)
        sorted_final_token_ids = final_token_ids.gather(
            1, sorted_final_indices.unsqueeze(-1).expand_as(final_token_ids)
        )
        sorted_final_mask = final_mask.gather(1, sorted_final_indices.unsqueeze(-1).expand_as(final_mask))

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
