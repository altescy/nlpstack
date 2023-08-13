import dataclasses
from typing import Any, Generic, Optional, TypeVar, Union

import torch

BeamScorerState = TypeVar("BeamScorerState")


class BeamScorer(Generic[BeamScorerState]):
    """
    BeamScorer is a base class for scoring generated sequences in beam search.
    """

    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> BeamScorerState:
        """
        Initialize the state of the beam scorer.

        Args:
            token_ids: Tensor of shape `(batch_size, beam_size, max_length)` representing predicted token indices.
            mask: Tensor of shape `(batch_size, beam_size, max_length)` representing mask for the predicted tokens.
        """
        raise NotImplementedError

    def score(
        self,
        state: BeamScorerState,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return new scores given the current state and log probabilities of the next tokens.

        Args:
            state: State of the beam scorer.
            log_probs: Tensor of shape `(batch_size, beam_size, sampling_size_per_node)`.

        Returns:
            state: Updated state of the beam scorer.
            scores: Tensor of shape `(batch_size, beam_size, sampling_size_per_node)`.
        """
        raise NotImplementedError

    def update_state(
        self,
        state: BeamScorerState,
        last_scores: torch.Tensor,
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> BeamScorerState:
        """
        Update the state of the beam scorer given the last backpointer indices.

        Args:
            state: State of the beam scorer.
            last_scores: Tensor of shape `(batch_size, beam_size)` representing scores at the last step.
            last_token_ids: Tensor of shape `(batch_size, beam_size)` representing token indices at the last step.
            last_backpointer: Tensor of shape `(batch_size, beam_size)` representing backpointer indices
                at the last step.

        Returns:
            state: Updated state of the beam scorer.
        """
        raise NotImplementedError

    def finalize(
        self,
        state: BeamScorerState,
        predictions: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Finalize the scores of generated sequences.

        Args:
            state: State of the beam scorer.
            predictions: Tensor of shape `(batch_size, beam_size, max_length)` representing predicted token indices.
            mask: Tensor of shape `(batch_size, beam_size, max_length)` representing mask for the predicted tokens.

        Returns:
            scores: Tensor of shape `(batch_size, beam_size)`.
        """
        raise NotImplementedError


class SequenceLogProbabilityScorer(BeamScorer["SequenceLogProbabilityScorer.State"]):
    """
    BeamScorer scores generated sequences by cumulating log probabilities for each time step.

    If `length_penalty` is specified, the scores are normalized by the length of the sequences
    to encourage shorter sequences:

    Args:
        length_penalty: Length penalty factor. Defaults to `1.0`.
    """

    @dataclasses.dataclass
    class State:
        """
        Parameters:
            cumulated_log_probs: Tensor of shape `(batch_size, beam_size)`.
            current_length: Tensor of shape `(batch_size, beam_size)`.
        """

        cumulated_log_probs: torch.Tensor
        current_lengths: torch.Tensor

    def __init__(self, length_penalty: Optional[Union[int, float]] = None):
        self._length_penalty = length_penalty
        self._eos_index: Optional[int] = None

    def setup(self, *args: Any, eos_index: Optional[int] = None, **kwarsg: Any) -> None:
        self._eos_index = eos_index

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> "SequenceLogProbabilityScorer.State":
        batch_size, beam_size, _ = token_ids.size()
        return self.State(
            cumulated_log_probs=token_ids.new_zeros(batch_size, beam_size),
            current_lengths=token_ids.new_zeros(batch_size, beam_size),
        )

    def score(
        self,
        state: "SequenceLogProbabilityScorer.State",
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: (batch_size, beam_size, sampling_size_per_node)
        scores = log_probs + state.cumulated_log_probs.unsqueeze(2)
        if self._length_penalty is not None:
            lengths = state.current_lengths.unsqueeze(2) + 1
            scores = scores / lengths**self._length_penalty
        return scores

    def update_state(
        self,
        state: "SequenceLogProbabilityScorer.State",
        last_scores: torch.Tensor,
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> "SequenceLogProbabilityScorer.State":
        cumulated_log_probs = state.cumulated_log_probs.gather(1, last_backpointer)
        current_lengths = state.current_lengths.gather(1, last_backpointer)

        if self._length_penalty is not None:
            # reconstruct log probabilities
            last_log_probs = (last_scores * current_lengths + 1) ** self._length_penalty
        else:
            last_log_probs = last_scores

        if self._eos_index is None:
            is_eos = last_token_ids == last_token_ids.new_zeros(1, dtype=torch.long)
        else:
            is_eos = last_token_ids == self._eos_index

        cumulated_log_probs = torch.where(
            is_eos,
            cumulated_log_probs,
            cumulated_log_probs + last_log_probs,
        )
        current_lengths = torch.where(
            is_eos,
            current_lengths,
            current_lengths + 1,
        )

        return self.State(
            cumulated_log_probs=cumulated_log_probs,
            current_lengths=current_lengths,
        )

    def finalize(
        self,
        state: "SequenceLogProbabilityScorer.State",
        predictions: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        scores = state.cumulated_log_probs
        if self._length_penalty is not None:
            scores = scores / (state.current_lengths**self._length_penalty + 1e-13)
        return scores
