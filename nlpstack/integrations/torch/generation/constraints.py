import dataclasses
from collections import defaultdict
from typing import Any, Dict, Generic, List, Literal, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar, cast

import torch

from nlpstack.common import DFA, NFA, DFAState
from nlpstack.data import Tokenizer, Vocabulary

ConstraintState = TypeVar("ConstraintState")


class Constraint(Generic[ConstraintState]):
    """
    A constraint that enforces constraints on the output sequence by manipulating
    the log probabilities.
    """

    def setup(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> ConstraintState:
        """
        Return the initial state of the constraint.

        Args:
            token_ids: Tensor of shape `(batch_size, beam_size, given_length)` representing
                the token IDs of the given sequence truncated to the minimum length in the batch.
            mask: Tensor of shape `(batch_size, beam_size, given_length)` representing
                the mask of the given sequence truncated to the minimum length in the batch.
            rest_token_ids: Tensor of shape `(batch_size, beam_size, rest_length)` representing
                the token IDs of the rest of the given sequence.
            rest_mask: Tensor of shape `(batch_size, beam_size, rest_length)` representing
                the mask of the rest of the given sequence.
        Returns:
            state: Initial state of the constraint.
        """
        raise NotImplementedError

    def apply(
        self,
        state: ConstraintState,
        log_probs: torch.Tensor,
        **kwargs: Any,
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

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> Sequence[Any]:
        return tuple(
            constraint.init_state(
                token_ids,
                mask,
                rest_token_ids,
                rest_mask,
            )
            for constraint in self.constraints
        )

    def apply(
        self,
        state: Sequence[Any],
        log_probs: torch.Tensor,
        **kwargs: Any,
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

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> State:
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
        **kwargs: Any,
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

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> None:
        return None

    def apply(
        self,
        state: None,
        log_probs: torch.Tensor,
        **kwargs: Any,
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

        given_lengths: torch.LongTensor
        current_lengths: torch.LongTensor

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        if min_length is not None and min_length < 0:
            raise ValueError(f"min_length ({min_length}) must be >= 0.")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length ({max_length}) must be >= 0.")
        if max_new_tokens is not None and max_new_tokens < 0:
            raise ValueError(f"max_new_tokens ({max_new_tokens}) must be >= 0.")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError(f"min_length ({min_length}) must be <= max_length ({max_length}).")

        self._min_length = min_length
        self._max_length = max_length
        self._max_new_tokens = max_new_tokens
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

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> State:
        assert self._eos_index is not None, "LengthConstraint requires an EOS index to be set."

        batch_size, beam_size, _ = token_ids.size()
        state = LengthConstraint.State(
            given_lengths=cast(torch.LongTensor, mask.sum(dim=2, dtype=torch.long)),
            current_lengths=cast(torch.LongTensor, mask.sum(dim=2, dtype=torch.long)),
        )
        return state

    def apply(
        self,
        state: "LengthConstraint.State",
        log_probs: torch.Tensor,
        *,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert self._eos_index is not None, "LengthConstraint requires an EOS index to be set."

        min_length = self._min_length if min_length is None else min_length
        max_length = self._max_length if max_length is None else max_length
        max_new_tokens = self._max_new_tokens if max_new_tokens is None else max_new_tokens

        # Shape: (batch_size, beam_size, 1)
        given_lengths = state.given_lengths.unsqueeze(2).expand_as(log_probs)
        # Shape: (batch_size, beam_size, 1)
        current_lengths = state.current_lengths.unsqueeze(2).expand_as(log_probs)
        # Shape: (1, vocab_size)
        eos_mask = (torch.arange(log_probs.size(-1), device=log_probs.device) == self._eos_index).unsqueeze(0)

        if max_new_tokens is not None:
            max_new_tokens_mask = current_lengths >= given_lengths + max_new_tokens
            log_probs = log_probs.masked_fill(max_new_tokens_mask * eos_mask, 0.0)
            log_probs = log_probs.masked_fill(max_new_tokens_mask * ~eos_mask, float("-inf"))
        if min_length is not None:
            min_length_mask = current_lengths < min_length
            log_probs = log_probs.masked_fill(min_length_mask * eos_mask, float("-inf"))
        if self._max_length is not None:
            max_length_mask = current_lengths >= max_length
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


class CandidatePhrasesConstraint(Constraint["CandidatePhrasesConstraint.State"]):
    """
    A constraint that enforces that the generated sequences match exactly a set of candidate
    phrases. This is useful for tasks like classification where the model is required to generate
    a sequence that matches a set of labels.

    Args:
        phrases: A list of phrases to match.
        namespace: The namespace of the generated tokens.
        tokenizer: A tokenizer to use for the generated JSON string. The tokenizer must be
            given at `__init__` or `setup` time.
    """

    @dataclasses.dataclass
    class State:
        dfastate: List[List[DFAState[int]]]
        timestep: int
        timestep_to_start: torch.LongTensor

    def __init__(
        self,
        phrases: Sequence[str],
        namespace: str = "tokens",
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self._phrases = phrases
        self._namespace = namespace
        self._tokenizer = tokenizer
        self._dfa: Optional[DFA[int]] = None

    def setup(
        self,
        *args: Any,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        target_tokenizer: Optional[Tokenizer] = None,
        eos_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        tokenizer = target_tokenizer or tokenizer or self._tokenizer

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided to JsonConstraint")

        self._tokenizer = tokenizer
        self._eos_index = eos_index
        self._dfa = self._nfa_from_phrases(self._phrases, tokenizer, vocab).compile()

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> "CandidatePhrasesConstraint.State":
        assert self._dfa is not None, "CandidatePhrasesConstraint.setup() must be called before init_state()"

        batch_size, beam_size, sequence_length = token_ids.size()
        timestep_to_start = cast(torch.LongTensor, mask.long().sum(-1) + rest_mask.long().sum(-1))
        return CandidatePhrasesConstraint.State(
            dfastate=[[self._dfa.state for _ in range(beam_size)] for _ in range(batch_size)],
            timestep=sequence_length,
            timestep_to_start=timestep_to_start,
        )

    def apply(
        self,
        state: "CandidatePhrasesConstraint.State",
        log_probs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size, beam_size, vocab_size = log_probs.size()

        all_token_ids = set(range(vocab_size))
        # Shape: (vocab_size,)
        eos_mask = (
            (torch.arange(vocab_size, device=log_probs.device) == self._eos_index)
            if self._eos_index is not None
            else None
        )
        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                dfastate = state.dfastate[batch_index][beam_index]
                timestep_to_start = int(state.timestep_to_start[batch_index, beam_index])
                if state.timestep < timestep_to_start:
                    continue
                if dfastate.is_final:
                    if eos_mask is not None:
                        log_probs[batch_index, beam_index, eos_mask] = 0.0
                        log_probs[batch_index, beam_index, ~eos_mask] = -float("inf")
                else:
                    candidates = dfastate.acceptable_symbols()
                    log_probs[batch_index, beam_index, list(all_token_ids - candidates)] = -float("inf")

        return log_probs

    def update_state(
        self,
        state: "CandidatePhrasesConstraint.State",
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> "CandidatePhrasesConstraint.State":
        assert self._dfa is not None, "CandidatePhrasesConstraint.setup() must be called before update_state()"

        batch_size, beam_size = last_backpointer.size()

        dfastates = state.dfastate

        # align constraint state
        for batch_index in range(batch_size):
            indices = last_backpointer[batch_index].tolist()
            dfastates[batch_index] = [state.dfastate[batch_index][index] for index in indices]

        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                dfastate = state.dfastate[batch_index][beam_index]
                if dfastate.is_final:
                    continue
                timestep_to_start = int(state.timestep_to_start[batch_index, beam_index].item())
                if state.timestep < timestep_to_start:
                    continue
                token_id = int(last_token_ids[batch_index, beam_index].item())
                dfastate_or_none = self._dfa.step(dfastate, token_id)
                if dfastate_or_none is None:
                    raise ValueError(f"Invalid token {token_id} at timestep {state.timestep}")
                dfastates[batch_index][beam_index] = dfastate_or_none

        return CandidatePhrasesConstraint.State(
            dfastate=dfastates,
            timestep=state.timestep + 1,
            timestep_to_start=state.timestep_to_start,
        )

    def _nfa_from_phrases(
        self,
        phrases: Sequence[str],
        tokenizer: Tokenizer,
        vocab: Vocabulary,
    ) -> NFA[int]:
        output = NFA[int].from_epsilon()
        for phrase in phrases:
            tokens = tokenizer.tokenize(phrase)
            token_ids = [vocab.get_index_by_token(self._namespace, token.surface) for token in tokens]
            nfa = NFA[int].from_epsilon()
            for token_id in token_ids:
                nfa += NFA.from_symbol(token_id)
            output |= nfa
        return output


class JsonConstraint(Constraint["JsonConstraint.State"]):
    """
    A constraint that ensures that the generated output is a valid JSON string that matches a given
    JSON schema. This constraint is implemented as a finite state machine that reads the output
    token at a time. The state machine is constructed from the given JSON schema.

    For causal language modeling, this constraint does not apply to the given tokens, but only to
    the tokens that are generated by the model. So you can feed any prompt to the model, and the
    constraint will only be applied after the prompt.

    Currently, there are some limitations to the schema and the generated JSON string:

    - Supported types are: "object", "array", "string", "number", "boolean", "null"
    - We assume that `true`, `false`, and `null` tokens are contained in the vocabulary.
    - Numbers are treated as a single token. So the model cannot generate numbers that are not
      already in the vocabulary.
    - The schema must be deterministic. This means that the schema must not contain any "anyOf",
      "oneOf", or "not" keywords.
    - The schema must not contain any default values or constraints, like "default", "pattern", etc.

    Args:
        jsonschema: A JSON schema that the output must match.
        namespace: A namespace of the vocabulary to use for the generated JSON string.
        tokenizer: A tokenizer to use for the generated JSON string. The tokenizer must be
            given at `__init__` or `setup` time.
    """

    class JsonSymbol(NamedTuple):
        name: Literal[
            "OPEN_BRACE",
            "CLOSE_BRACE",
            "OPEN_BRACKET",
            "CLOSE_BRACKET",
            "COLON",
            "COMMA",
            "QUOTE",
            "KEY",
            "STRING",
            "NUMBER",
            "BOOLEAN",
            "NULL",
            "WHITESPACE",
        ]
        value: Optional[str] = None
        last_json_symbol: Optional["JsonConstraint.JsonSymbol"] = None

    @dataclasses.dataclass
    class _BeamState:
        dfastate: DFAState["JsonConstraint.JsonSymbol"]
        last_json_symbol: Optional["JsonConstraint.JsonSymbol"] = None
        buffer: List[int] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class State:
        states: List[List["JsonConstraint._BeamState"]]
        timestep_to_start: torch.LongTensor
        timestep: int = 0

    def __init__(
        self,
        jsonschema: Mapping[str, Any],
        namespace: str = "tokens",
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self._jsonschema = jsonschema
        self._namespace = namespace
        self._tokenizer = tokenizer
        self._eos_index: Optional[int] = None
        self._dfa = self._nfa_from_jsonschema(self._jsonschema).compile()
        self._string_token_ids: Set[int] = set()
        self._number_token_ids: Set[int] = set()
        self._boolean_token_ids: Set[int] = set()
        self._null_token_ids: Set[int] = set()
        self._whitespace_token_ids: Set[int] = set()
        self._open_brace_token_ids: Set[int] = set()
        self._close_brace_token_ids: Set[int] = set()
        self._open_bracket_token_ids: Set[int] = set()
        self._close_bracket_token_ids: Set[int] = set()
        self._colon_token_ids: Set[int] = set()
        self._comma_token_ids: Set[int] = set()
        self._quote_token_ids: Set[int] = set()
        self._key_token_ids: Dict[str, Sequence[int]] = {}

    def setup(
        self,
        *args: Any,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer] = None,
        target_tokenizer: Optional[Tokenizer] = None,
        eos_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        keys = self._extract_keys_from_schema(self._jsonschema)
        index_to_token = vocab.get_index_to_token(self._namespace)
        tokenizer = target_tokenizer or tokenizer or self._tokenizer

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided to JsonConstraint")

        self._tokenizer = tokenizer
        self._eos_index = eos_index
        self._index_to_token = index_to_token
        self._string_token_ids = {index for index, token in index_to_token.items() if self._is_string_token(token)}
        self._number_token_ids = {index for index, token in index_to_token.items() if self._is_number_token(token)}
        self._boolean_token_ids = {index for index, token in index_to_token.items() if self._is_boolean_token(token)}
        self._null_token_ids = {index for index, token in index_to_token.items() if self._is_null_token(token)}
        self._whitespace_token_ids = {
            index for index, token in index_to_token.items() if self._is_whitespace_token(token)
        }
        self._open_brace_token_ids = {
            index for index, token in index_to_token.items() if self._is_open_brace_token(token)
        }
        self._close_brace_token_ids = {
            index for index, token in index_to_token.items() if self._is_close_brace_token(token)
        }
        self._open_bracket_token_ids = {
            index for index, token in index_to_token.items() if self._is_open_bracket_token(token)
        }
        self._close_bracket_token_ids = {
            index for index, token in index_to_token.items() if self._is_close_bracket_token(token)
        }
        self._colon_token_ids = {index for index, token in index_to_token.items() if self._is_colon_token(token)}
        self._comma_token_ids = {index for index, token in index_to_token.items() if self._is_comma_token(token)}
        self._quote_token_ids = {index for index, token in index_to_token.items() if self._is_quote_token(token)}
        self._key_token_ids = {
            key: [vocab.get_index_by_token(self._namespace, token.surface) for token in self._tokenizer.tokenize(key)]
            for *_, key in keys
        }

    def init_state(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        rest_token_ids: torch.LongTensor,
        rest_mask: torch.BoolTensor,
    ) -> "JsonConstraint.State":
        batch_size, beam_size, sequence_length = token_ids.size()
        timestep_to_start = cast(torch.LongTensor, mask.long().sum(-1) + rest_mask.long().sum(-1))
        return JsonConstraint.State(
            states=[
                [JsonConstraint._BeamState(dfastate=self._dfa.state) for _ in range(beam_size)]
                for _ in range(batch_size)
            ],
            timestep=sequence_length,
            timestep_to_start=timestep_to_start,
        )

    def apply(
        self,
        state: "JsonConstraint.State",
        log_probs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size, beam_size, vocab_size = log_probs.size()

        all_token_ids = set(range(vocab_size))
        # Shape: (vocab_size,)
        eos_mask = (
            (torch.arange(vocab_size, device=log_probs.device) == self._eos_index)
            if self._eos_index is not None
            else None
        )
        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                beam_state = state.states[batch_index][beam_index]
                timestep_to_start = int(state.timestep_to_start[batch_index, beam_index])
                if state.timestep < timestep_to_start:
                    continue
                if beam_state.dfastate.is_final:
                    if eos_mask is not None:
                        log_probs[batch_index, beam_index, eos_mask] = 0.0
                        log_probs[batch_index, beam_index, ~eos_mask] = -float("inf")
                else:
                    json_symbols = beam_state.dfastate.acceptable_symbols()
                    candidates = set()
                    for json_symbol in json_symbols:
                        candidates |= self._get_available_token_ids(beam_state, json_symbol)
                    token_ids_to_mask = all_token_ids - candidates
                    log_probs[batch_index, beam_index, list(token_ids_to_mask)] = -float("inf")

        return log_probs

    def update_state(
        self,
        state: "JsonConstraint.State",
        last_token_ids: torch.LongTensor,
        last_backpointer: torch.LongTensor,
    ) -> "JsonConstraint.State":
        batch_size, beam_size = last_backpointer.size()

        # align constraint state
        for batch_index in range(batch_size):
            indices = last_backpointer[batch_index].tolist()
            state.states[batch_index] = [state.states[batch_index][index] for index in indices]

        beam_states = state.states
        timestep = state.timestep
        timestep_to_start = state.timestep_to_start
        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                if timestep < timestep_to_start[batch_index, beam_index].item():
                    continue

                beam_state = beam_states[batch_index][beam_index]
                last_token_id = int(last_token_ids[batch_index, beam_index].item())

                if beam_state.dfastate.is_final:
                    continue

                last_json_symbol = self._inspect_json_symbol(beam_state, last_token_id)
                dfastate = beam_state.dfastate
                buffer = (
                    beam_state.buffer + [last_token_id]
                    if last_json_symbol == beam_state.last_json_symbol
                    else [last_token_id]
                )

                do_update = True
                if last_json_symbol.name == "KEY":
                    assert last_json_symbol.value is not None
                    key_token_ids = self._key_token_ids[last_json_symbol.value]
                    # If key generation is not finished, we should not update the state.
                    do_update = len(buffer) == len(key_token_ids)

                if do_update:
                    dfastate_or_none = self._dfa.step(dfastate, last_json_symbol)
                    if dfastate_or_none is None:
                        raise ValueError(f"Invalid token id: {last_token_id}")
                    dfastate = dfastate_or_none

                beam_states[batch_index][beam_index] = JsonConstraint._BeamState(
                    dfastate=dfastate,
                    buffer=buffer,
                    last_json_symbol=last_json_symbol,
                )

        return JsonConstraint.State(
            states=beam_states,
            timestep=timestep + 1,
            timestep_to_start=timestep_to_start,
        )

    def _get_available_token_ids(
        self,
        state: "JsonConstraint._BeamState",
        json_symbol: "JsonConstraint.JsonSymbol",
    ) -> Set[int]:
        if json_symbol.name == "STRING":
            return self._string_token_ids
        elif json_symbol.name == "NUMBER":
            return self._number_token_ids
        elif json_symbol.name == "BOOLEAN":
            return self._boolean_token_ids
        elif json_symbol.name == "NULL":
            return self._null_token_ids
        elif json_symbol.name == "WHITESPACE":
            return self._whitespace_token_ids
        elif json_symbol.name == "OPEN_BRACE":
            return self._open_brace_token_ids
        elif json_symbol.name == "CLOSE_BRACE":
            return self._close_brace_token_ids
        elif json_symbol.name == "OPEN_BRACKET":
            return self._open_bracket_token_ids
        elif json_symbol.name == "CLOSE_BRACKET":
            return self._close_bracket_token_ids
        elif json_symbol.name == "COLON":
            return self._colon_token_ids
        elif json_symbol.name == "COMMA":
            return self._comma_token_ids
        elif json_symbol.name == "QUOTE":
            return self._quote_token_ids
        elif json_symbol.name == "KEY":
            assert json_symbol.value is not None
            key_token_ids = self._key_token_ids[json_symbol.value]
            if state.last_json_symbol is None or state.last_json_symbol.name != "KEY":
                next_token_id = key_token_ids[0]
            else:
                next_token_id = key_token_ids[len(state.buffer)]
            return {next_token_id}
        raise RuntimeError(f"Unknown symbol: {json_symbol}")

    def _inspect_json_symbol(self, state: "JsonConstraint._BeamState", token_id: int) -> "JsonConstraint.JsonSymbol":
        candidate_json_symbols = state.dfastate.acceptable_symbols()
        for json_symbol in candidate_json_symbols:
            if token_id in self._get_available_token_ids(state, json_symbol):
                return json_symbol
        raise RuntimeError(f"Invalid token_id: {token_id}")

    @staticmethod
    def _nfa_from_jsonschema(jsonschema: Mapping[str, Any]) -> NFA[JsonSymbol]:
        schema_type = jsonschema["type"]
        if schema_type == "object":
            nfa = NFA.from_symbol(JsonConstraint.JsonSymbol("OPEN_BRACE"))
            properties = jsonschema.get("properties", {})
            for index, (key, subschema) in enumerate(properties.items()):
                nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("QUOTE"))
                nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("KEY", key))
                nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("QUOTE"))
                nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("COLON"))
                nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("WHITESPACE"))
                nfa += JsonConstraint._nfa_from_jsonschema(subschema)
                if index != len(properties) - 1:
                    nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("COMMA"))
                    nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("WHITESPACE"))
            nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("CLOSE_BRACE"))
            return nfa
        if schema_type == "array":
            head = NFA.from_symbol(JsonConstraint.JsonSymbol("OPEN_BRACKET"))
            items = JsonConstraint._nfa_from_jsonschema(jsonschema["items"])
            delimiter = NFA.from_symbol(JsonConstraint.JsonSymbol("COMMA")) + NFA.from_symbol(
                JsonConstraint.JsonSymbol("WHITESPACE")
            )
            delimiter.end.is_final = False
            items.end.epsilon_transitions.add(delimiter.start)
            delimiter.end.epsilon_transitions.add(items.start)
            tail = NFA.from_symbol(JsonConstraint.JsonSymbol("CLOSE_BRACKET"))
            head.end.epsilon_transitions.add(tail.start)
            return head + items + tail
        if schema_type == "string":
            nfa = NFA.from_symbol(JsonConstraint.JsonSymbol("QUOTE"))
            nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("STRING")).closure()
            nfa += NFA.from_symbol(JsonConstraint.JsonSymbol("QUOTE"))
            return nfa
        if schema_type == "number":
            return NFA.from_symbol(JsonConstraint.JsonSymbol("NUMBER"))
        if schema_type == "boolean":
            return NFA.from_symbol(JsonConstraint.JsonSymbol("BOOLEAN"))
        if schema_type == "null":
            return NFA.from_symbol(JsonConstraint.JsonSymbol("NULL"))
        raise ValueError(f"Unknown schema type: {schema_type}")

    @staticmethod
    def _extract_keys_from_schema(
        jsonschema: Mapping[str, Any],
        prefix: Tuple[str, ...] = (),
    ) -> Sequence[Tuple[str, ...]]:
        """
        Extracts all keys from a JSON schema.
        Nested keys are represented as tuples.
        """
        keys: List[Tuple[str, ...]] = []
        schema_type = jsonschema["type"]
        if schema_type == "object":
            for key, subschema in jsonschema.get("properties", {}).items():
                keys.append(prefix + (key,))
                keys += JsonConstraint._extract_keys_from_schema(subschema, prefix + (key,))
        if schema_type == "array":
            keys += JsonConstraint._extract_keys_from_schema(jsonschema["items"], prefix + ("__array__",))
        return keys

    @staticmethod
    def _is_string_token(token: str) -> bool:
        return '"' not in token

    @staticmethod
    def _is_number_token(token: str) -> bool:
        token = token.strip(" \n\t")
        if not all(char.isdigit() or char in ".-" for char in token):
            return False
        if "." in token:
            if token.count(".") > 1:
                return False
            if token.startswith(".") or token.endswith("."):
                return False
        if "-" in token:
            if token.count("-") > 1:
                return False
            if not token.startswith("-"):
                return False
        if "-." in token:
            return False
        return True

    @staticmethod
    def _is_boolean_token(token: str) -> bool:
        return token.strip(" \n\t") in {"true", "false"}

    @staticmethod
    def _is_null_token(token: str) -> bool:
        return token.strip(" \n\t") == "null"

    @staticmethod
    def _is_whitespace_token(token: str) -> bool:
        return token.strip(" \n\t") == ""

    @staticmethod
    def _is_open_brace_token(token: str) -> bool:
        return token.strip(" \n\t") == "{"

    @staticmethod
    def _is_close_brace_token(token: str) -> bool:
        return token.strip(" \n\t") == "}"

    @staticmethod
    def _is_open_bracket_token(token: str) -> bool:
        return token.strip(" \n\t") == "["

    @staticmethod
    def _is_close_bracket_token(token: str) -> bool:
        return token.strip(" \n\t") == "]"

    @staticmethod
    def _is_colon_token(token: str) -> bool:
        return token.strip(" \n\t") == ":"

    @staticmethod
    def _is_comma_token(token: str) -> bool:
        return token.strip(" \n\t") == ","

    @staticmethod
    def _is_quote_token(token: str) -> bool:
        return token.strip(" \n\t") == '"'
