import dataclasses
from collections import defaultdict
from typing import Any, Dict, Generic, List, Optional, Sequence, Set, Tuple, TypeVar, cast

import torch

from nlpstack.data import Vocabulary

ConstraintState = TypeVar("ConstraintState")


class Constraint(Generic[ConstraintState]):
    """
    A constraint that enforces constraints on the output sequence by manipulating
    the log probabilities.
    """

    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> ConstraintState:
        """
        Return the initial state of the constraint.

        Args:
            token_ids: Tensor of shape `(batch_size, beam_size, given_length)`.
            mask: Tensor of shape `(batch_size, beam_size, given_length)`.
        Returns:
            state: Initial state of the constraint.
        """
        raise NotImplementedError

    def apply(
        self,
        state: ConstraintState,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the constraint to the log probabilities.

        Args:
            state: State of the constraint.
            log_probs: Tensor of shape `(batch_size, beam_size, vocab_size)`.
        """
        raise NotImplementedError

    def update_state(
        self,
        state: ConstraintState,
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> ConstraintState:
        """
        Update the state of the constraint.

        Basically, this function should do the following:
        1. Align the constraint state with the beam search state by using the backpointer.
        2. Update the constraint state based on the selected token ids.

        Args:
            state: State of the constraint.
            last_token_ids: Tensor of shape `(batch_size, beam_size)` representing predicted token IDs
                at the last step.
            last_backpointer: Tensor of shape `(batch_size, beam_size)` representing backpointer indices
                at the last step.
        """
        raise NotImplementedError


class MultiConstraint(Constraint[Sequence[Any]]):
    """
    A constraint that applies a set of constraints in a sequential order.

    Args:
        constraints: A list of constraints to apply.
    """

    def __init__(self, constraints: Sequence[Constraint[Any]]):
        self.constraints = constraints

    def setup(self, *args: Any, **kwargs: Any) -> None:
        for constraint in self.constraints:
            constraint.setup(*args, **kwargs)

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> Sequence[Any]:
        return tuple(constraint.init_state(token_ids, mask) for constraint in self.constraints)

    def apply(
        self,
        state: Sequence[Any],
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        for constraint, constraint_state in zip(self.constraints, state):
            log_probs = constraint.apply(constraint_state, log_probs)
        return log_probs

    def update_state(
        self,
        state: Sequence[Any],
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> Sequence[Any]:
        return tuple(
            constraint.update_state(constraint_state, last_token_ids, last_backpointer)
            for constraint, constraint_state in zip(self.constraints, state)
        )


class NoRepeatNgramConstraint(Constraint["NoRepeatNgramConstraint.State"]):
    """
    A constraint that enforces that no n-gram repeats in the output sequence.

    Args:
        ngram_size: Size of the n-gram to check for repeats.
    """

    @dataclasses.dataclass
    class State:
        """
        Parameters:
            seen_ngrams: List of ngrams seen so far. `seen_ngram[batch_index][beam_index]` is a dict
                mapping prefix of the ngram to the last tokens.
            current_prefix: Prefix of the current ngram. `current_prefix[batch_index][beam_index]`
                represents the prefix of the ngram that is currently being constructed.
        """

        seen_ngrams: List[List[Dict[Tuple[int, ...], Set[int]]]]
        current_prefix: List[List[List[int]]]

    def __init__(self, ngram_size: int) -> None:
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")
        self._ngram_size = ngram_size

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> State:
        batch_size, beam_size, _ = token_ids.size()
        state = NoRepeatNgramConstraint.State(
            seen_ngrams=[[defaultdict(set) for _ in range(beam_size)] for _ in range(batch_size)],
            current_prefix=[[[] for _ in range(beam_size)] for _ in range(batch_size)],
        )
        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                tokens = token_ids[batch_index, beam_index][mask[batch_index, beam_index]].tolist()
                state.current_prefix[batch_index][beam_index] = tokens[-(self._ngram_size - 1) :]
                for i in range(len(tokens) - self._ngram_size + 1):
                    prefix = tuple(tokens[i : i + self._ngram_size - 1])
                    last_token = tokens[i + self._ngram_size - 1]
                    state.seen_ngrams[batch_index][beam_index][prefix].add(last_token)
        return state

    def apply(
        self,
        state: State,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, beam_size, vocab_size = log_probs.size()
        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                prefix = tuple(state.current_prefix[batch_index][beam_index])
                for last_token in state.seen_ngrams[batch_index][beam_index][prefix]:
                    log_probs[batch_index, beam_index, last_token] = float("-inf")
        return log_probs

    def update_state(
        self,
        state: State,
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> "NoRepeatNgramConstraint.State":
        batch_size, beam_size = last_token_ids.size()

        # align constraint state
        for batch_index in range(batch_size):
            indices = last_backpointer[batch_index].tolist()
            state.seen_ngrams[batch_index] = [state.seen_ngrams[batch_index][i] for i in indices]
            state.current_prefix[batch_index] = [state.current_prefix[batch_index][i] for i in indices]

        # update constraint state
        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                prefix = state.current_prefix[batch_index][beam_index]
                last_token = int(last_token_ids[batch_index, beam_index].item())

                if len(prefix) == self._ngram_size - 1:
                    state.seen_ngrams[batch_index][beam_index][tuple(prefix)].add(last_token)

                prefix.append(last_token)
                if len(prefix) == self._ngram_size:
                    prefix.pop(0)
        return state


class StopTokenConstraint(Constraint[None]):
    def __init__(
        self,
        stop_tokens: Sequence[str],
        namespace: str = "tokens",
    ) -> None:
        self._namespace = namespace
        self._stop_tokens = set(stop_tokens)
        self._stop_token_ids: Sequence[int] = ()

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        self._stop_token_ids = [vocab.get_index_by_token(self._namespace, token) for token in self._stop_tokens]

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> None:
        return None

    def apply(
        self,
        state: None,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self._stop_token_ids:
            log_probs[:, :, self._stop_token_ids] = float("-inf")
        return log_probs

    def update_state(
        self,
        state: None,
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> None:
        return state


class LengthConstraint(Constraint["LengthConstraint.State"]):
    """
    A constraint that enforces a minimum and maximum length on the sequences produced by beam search.

    Args:
        min_length: The minimum length of the sequences produced by beam search.
        max_length: The maximum length of the sequences produced by beam search.
    """

    @dataclasses.dataclass
    class State:
        """
        Parameters:
            current_lengths: Tensor of shape `(batch_size, beam_size)` representing the current lengths of the
                sequences in the beam.
        """

        current_lengths: torch.LongTensor

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        if min_length is not None and min_length < 0:
            raise ValueError(f"min_length ({min_length}) must be >= 0.")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length ({max_length}) must be >= 0.")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError(f"min_length ({min_length}) must be <= max_length ({max_length}).")

        self._min_length = min_length
        self._max_length = max_length
        self._eos_index: Optional[int] = None

    def setup(
        self,
        *args: Any,
        eos_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if eos_index is None:
            raise ValueError("LengthConstraint requires an EOS index to be set.")
        self._eos_index = eos_index

    def init_state(self, token_ids: torch.LongTensor, mask: torch.BoolTensor) -> State:
        assert self._eos_index is not None, "LengthConstraint requires an EOS index to be set."

        batch_size, beam_size, _ = token_ids.size()
        state = LengthConstraint.State(
            current_lengths=cast(torch.LongTensor, mask.sum(dim=2, dtype=torch.long)),
        )
        return state

    def apply(
        self,
        state: "LengthConstraint.State",
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        assert self._eos_index is not None, "LengthConstraint requires an EOS index to be set."

        # Shape: (batch_size, beam_size, 1)
        current_lengths = state.current_lengths.unsqueeze(2).expand_as(log_probs)
        # Shape: (1, vocab_size)
        eos_mask = (torch.arange(log_probs.size(-1), device=log_probs.device) == self._eos_index).unsqueeze(0)

        if self._min_length is not None:
            min_length_mask = current_lengths < self._min_length
            log_probs = log_probs.masked_fill(min_length_mask * eos_mask, float("-inf"))
        if self._max_length is not None:
            max_length_mask = current_lengths >= self._max_length
            log_probs = log_probs.masked_fill(max_length_mask * eos_mask, 0.0)
            log_probs = log_probs.masked_fill(max_length_mask * ~eos_mask, float("-inf"))

        return log_probs

    def update_state(
        self,
        state: "LengthConstraint.State",
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> "LengthConstraint.State":
        current_lengths = state.current_lengths.gather(dim=1, index=last_backpointer)
        state.current_lengths = cast(torch.LongTensor, current_lengths + 1)
        return state
