"""
This script is based on the original implementation of AllenNLP:
https://github.com/allenai/allennlp/blob/v2.10.1/allennlp/modules/conditional_random_field/conditional_random_field.py
"""

from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import torch

import nlpstack.integrations.torch.util as util
from nlpstack.data.vocabulary import Vocabulary

ViterbiDecoding = Tuple[List[int], float]  # a list of tags, and a viterbi score
ConstraintType = Literal["BIO", "IOB1", "BIOUL", "BMES"]


def allowed_transitions(constraint_type: ConstraintType, labels: Mapping[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Args:
        constraint_type:
            Indicates which constraint to apply. Current choices are
            "BIO", "IOB1", "BIOUL", and "BMES".
        labels:
            A mapping {label_id -> label}. Most commonly this would be the value from
            Vocabulary.get_index_to_token_vocabulary()

    Returns:
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed: List[Tuple[int, int]] = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: ConstraintType,
    from_tag: str,
    from_entity: str,
    to_tag: str,
    to_entity: str,
) -> bool:
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    Args:
        constraint_type:
            Indicates which constraint to apply. Current choices are
            "BIO", "IOB1", "BIOUL", and "BMES".
        from_tag:
            The tag that the transition originates from. For example, if the
            label is `I-PER`, the `from_tag` is `I`.
        from_entity:
            The entity corresponding to the `from_tag`. For example, if the
            label is `I-PER`, the `from_entity` is `PER`.
        to_tag:
            The tag that the transition leads to. For example, if the
            label is `I-PER`, the `to_tag` is `I`.
        to_entity:
            The entity corresponding to the `to_tag`. For example, if the
            label is `I-PER`, the `to_entity` is `PER`.

    Returns:
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Args:
        num_labels:
            The number of labels.
        constraint:
            An optional list of allowed transitions (from_tag_id, to_tag_id).
            These are applied to `viterbi_tags()` but do not affect `forward()`.
            These should be derived from `allowed_transitions` so that the
            start and end transitions are handled correctly for your tag type.
        include_start_end_transitions:
            Whether to include the start and end transition parameters.
    """

    def __init__(
        self,
        num_labels: int,
        constraint: Optional[Sequence[Tuple[int, int]]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.constraint = constraint
        self.include_start_end_transitions = include_start_end_transitions

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.empty(num_labels, num_labels))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        self.constraint_mask = torch.nn.Parameter(
            torch.full((num_labels + 2, num_labels + 2), 1.0), requires_grad=False
        )

        # Also need logits for transitioning from "start" state and to "end" state.
        self.start_transitions: Optional[torch.nn.Parameter] = None
        self.end_transitions: Optional[torch.nn.Parameter] = None
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_labels))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_labels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.transitions)
        if (
            self.include_start_end_transitions
            and self.start_transitions is not None
            and self.end_transitions is not None
        ):
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)
        if self.constraint is not None:
            self.constraint_mask.fill_(0.0)
            for i, j in self.constraint:
                self.constraint_mask[i, j] = 1.0

    def _input_likelihood(
        self,
        logits: torch.Tensor,
        transitions: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Computes the (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood

        This is the sum of the likelihoods across all possible state sequences.

        Args:
            logits (torch.Tensor): a (batch_size, sequence_length num_labels) tensor of
                unnormalized log-probabilities
            transitions (torch.Tensor): a (batch_size, num_labels, num_labels) tensor of transition scores
            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags

        Returns:
            torch.Tensor: (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood
        """
        batch_size, sequence_length, num_labels = logits.size()

        # Transpose batch size and sequence dimensions
        mask = cast(torch.BoolTensor, mask.transpose(0, 1).contiguous())
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_labels) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            assert self.start_transitions is not None
            alpha = self.start_transitions.view(1, num_labels) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_labels, num_labels) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_labels)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = transitions.view(1, num_labels, num_labels)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_labels, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            alpha = util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (~mask[i]).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            assert self.end_transitions is not None
            stops = alpha + self.end_transitions.view(1, num_labels)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_labels dim, result is (batch_size,)
        return util.logsumexp(stops)

    def _joint_likelihood(
        self,
        logits: torch.Tensor,
        transitions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Computes the numerator term for the log-likelihood, which is just score(inputs, tags)

        Args:
            logits (torch.Tensor): a (batch_size, sequence_length num_labels) tensor of unnormalized
                log-probabilities
            transitions (torch.Tensor): a (batch_size, num_labels, num_labels) tensor of transition scores
            tags (torch.Tensor): output tag sequences (batch_size, sequence_length) $y$ for each input sequence
            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags

        Returns:
            torch.Tensor: numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = cast(torch.BoolTensor, mask.transpose(0, 1).contiguous())
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            assert self.start_transitions is not None
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0  # type: ignore[assignment]

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1).long()).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size).long()).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            assert self.end_transitions is not None
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0  # type: ignore[assignment]

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_labels)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1).long())  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Computes the log likelihood for the given batch of input sequences $(x,y)$

        Args:
            inputs: (batch_size, sequence_length, num_labels) tensor of logits for the inputs $x$
            labels: (batch_size, sequence_length) tensor of labels $y$
            mask: (batch_size, sequence_length) tensor of masking flags.
                Defaults to None.

        Returns:
            log likelihoods $log P(y|x)$
        """
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones(*labels.size(), dtype=torch.bool, device=inputs.device))
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = cast(torch.BoolTensor, mask.to(torch.bool))

        log_denominator = self._input_likelihood(inputs, self.transitions, mask)
        log_numerator = self._joint_likelihood(inputs, self.transitions, labels, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_decode(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        *,
        top_k: int = 1,
    ) -> List[List[ViterbiDecoding]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.

        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)
        """
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device))

        _, max_seq_length, num_labels = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data  # type: ignore[assignment]

        # Augment transitions matrix with start and end transitions
        start_tag = num_labels
        end_tag = num_labels + 1
        transitions = torch.full((num_labels + 2, num_labels + 2), -10000.0, device=logits.device)

        # Apply transition constraints
        constrained_transitions = self.transitions * self.constraint_mask[:num_labels, :num_labels] + -10000.0 * (
            1 - self.constraint_mask[:num_labels, :num_labels]
        )
        transitions[:num_labels, :num_labels] = constrained_transitions.data

        if self.include_start_end_transitions:
            assert self.start_transitions is not None and self.end_transitions is not None
            transitions[start_tag, :num_labels] = self.start_transitions.detach() * self.constraint_mask[
                start_tag, :num_labels
            ].data + -10000.0 * (1 - self.constraint_mask[start_tag, :num_labels].detach())
            transitions[:num_labels, end_tag] = self.end_transitions.detach() * self.constraint_mask[
                :num_labels, end_tag
            ].data + -10000.0 * (1 - self.constraint_mask[:num_labels, end_tag].detach())
        else:
            transitions[start_tag, :num_labels] = -10000.0 * (1 - self.constraint_mask[start_tag, :num_labels].detach())
            transitions[:num_labels, end_tag] = -10000.0 * (1 - self.constraint_mask[:num_labels, end_tag].detach())

        best_paths: List[List[ViterbiDecoding]] = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.empty(max_seq_length + 2, num_labels + 2, device=logits.device)

        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.0)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.0
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1 : (sequence_length + 1), :num_labels] = masked_prediction
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            # We pass the tags and the transitions to `viterbi_decode`.
            viterbi_paths, viterbi_scores = util.viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k,
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append.
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)

        return best_paths


class LazyConditionalRandomField(torch.nn.modules.lazy.LazyModuleMixin, ConditionalRandomField):
    cls_to_become = ConditionalRandomField  # type: ignore[assignment]
    transitions: torch.nn.UninitializedParameter  # type: ignore[assignment]
    start_transitions: Optional[torch.nn.UninitializedParameter]  # type: ignore[assignment]
    end_transitions: Optional[torch.nn.UninitializedParameter]  # type: ignore[assignment]
    constraint_mask: torch.nn.UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self,
        constraint: Optional[Sequence[Tuple[int, int]]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__(1, constraint, include_start_end_transitions)

        self.transitions = torch.nn.UninitializedParameter()
        self.constraint_mask = torch.nn.UninitializedParameter(requires_grad=False)
        if include_start_end_transitions:
            self.start_transitions = torch.nn.UninitializedParameter()
            self.end_transitions = torch.nn.UninitializedParameter()

    def initialize_parameters(
        self,
        inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        if self.has_uninitialized_params():
            num_labels = inputs.size(-1)
            self.num_labels = num_labels
            self.transitions.materialize((num_labels, num_labels))
            if self.start_transitions is not None:
                self.start_transitions.materialize((num_labels,))
            if self.end_transitions is not None:
                self.end_transitions.materialize((num_labels,))
            self.constraint_mask.materialize((num_labels + 2, num_labels + 2))
            self.reset_parameters()


class CrfDecoder(torch.nn.Module):
    def __init__(
        self,
        constraint: Optional[Union[ConstraintType, Sequence[Tuple[int, int]]]] = None,
        include_start_end_transitions: bool = True,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__()

        self.constraint = constraint
        self.crf = LazyConditionalRandomField(None, include_start_end_transitions)
        self.label_namespace = label_namespace

    def setup(self, *args: Any, vocab: Vocabulary, **kwargs: Any) -> None:
        if isinstance(self.constraint, str):
            constraint = cast(ConstraintType, self.constraint)
            labels = vocab.get_index_to_token(self.label_namespace)
            self.crf.constraint = allowed_transitions(constraint, labels)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.crf(inputs, labels, mask))

    def viterbi_decode(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        *,
        top_k: int = 1,
    ) -> List[List[ViterbiDecoding]]:
        return self.crf.viterbi_decode(logits, mask, top_k=top_k)
