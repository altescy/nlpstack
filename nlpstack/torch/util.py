import math
from logging import getLogger
from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union, cast, overload

import numpy
import torch

logger = getLogger(__name__)

T = TypeVar("T")


def get_mask_from_text(text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.BoolTensor:
    """
    :param text: Mapping[str, Mapping[str, torch.nn.LongTensor]]
    :return: torch.BoolTensor
    """
    for inputs in text.values():
        if "mask" in inputs:
            return cast(torch.BoolTensor, inputs["mask"].bool())
    raise ValueError("No mask found in text")


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(obj: T, device: Union[int, torch.device]) -> T:
    device = int_to_device(device)

    if isinstance(obj, numpy.ndarray):
        return cast(T, torch.from_numpy(obj).to(device=device))
    if isinstance(obj, torch.Tensor):
        return cast(T, obj if obj.device == device else obj.to(device=device))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return cast(T, obj)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = move_to_device(item, device)
        return cast(T, obj)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return cast(T, obj.__class__(*(move_to_device(item, device) for item in obj)))
    elif isinstance(obj, tuple):
        return cast(T, tuple(move_to_device(item, device) for item in obj))

    return obj


def tensor_to_numpy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = tensor_to_numpy(value)
        return obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = tensor_to_numpy(item)
        return obj
    elif isinstance(obj, tuple):
        return tuple(tensor_to_numpy(item) for item in obj)

    return obj


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return cast(torch.Tensor, max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log())


@overload
def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: int,
    tag_observations: Optional[List[int]] = ...,
    allowed_start_transitions: Optional[torch.Tensor] = ...,
    allowed_end_transitions: Optional[torch.Tensor] = ...,
) -> Tuple[List[List[int]], torch.Tensor]:
    ...


@overload
def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: None = ...,
    tag_observations: Optional[List[int]] = ...,
    allowed_start_transitions: Optional[torch.Tensor] = ...,
    allowed_end_transitions: Optional[torch.Tensor] = ...,
) -> Tuple[List[int], torch.Tensor]:
    ...


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: Optional[int] = None,
    tag_observations: Optional[List[int]] = None,
    allowed_start_transitions: Optional[torch.Tensor] = None,
    allowed_end_transitions: Optional[torch.Tensor] = None,
) -> Union[Tuple[List[int], torch.Tensor], Tuple[List[List[int]], torch.Tensor]]:
    """
    This implementation is originally from AllenNLP:
    https://github.com/allenai/allennlp/blob/v2.10.1/allennlp/nn/util.py#L405

    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.

    Args:
        tag_sequence:
            A tensor of shape (sequence_length, num_tags) representing scores for
            a set of tags over a given sequence.
        transition_matrix:
            A tensor of shape (num_tags, num_tags) representing the binary potentials
            for transitioning between a given pair of tags.
        tag_observations:
            A list of length `sequence_length` containing the class ids of observed
            elements in the sequence, with unobserved elements being set to -1. Note that
            it is possible to provide evidence which results in degenerate labelings if
            the sequences of tags you provide as evidence cannot transition between each
            other, or those transitions are extremely unlikely. In this situation we log a
            warning, but the responsibility for providing self-consistent evidence ultimately
            lies with the user.
        allowed_start_transitions:
            An optional tensor of shape (num_tags,) describing which tags the START token
            may transition *to*. If provided, additional transition constraints will be used for
            determining the start element of the sequence.
        allowed_end_transitions:
            An optional tensor of shape (num_tags,) describing which tags may transition *to* the
            end tag. If provided, additional transition constraints will be used for determining
            the end element of the sequence.
        top_k:
            Optional integer specifying how many of the top paths to return. For top_k>=1, returns
            a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
            tuple with just the top path and its score (not in lists, for backwards compatibility).

    Returns:
        viterbi_path:
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score:
            The score of the viterbi path.
    """
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f"top_k must be either None or an integer >=1. Instead received {top_k}")

    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None

    if has_start_end_restrictions:
        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.

        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ValueError(
                "Observations were provided, but they were not the same length "
                "as the sequence. Found sequence of length: {} and evidence: {}".format(
                    sequence_length, tag_observations
                )
            )
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning(
                    "The pairwise potential between tags you have passed as "
                    "observations is extremely unlikely. Double check your evidence "
                    "or transition potentials!"
                )
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()

        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]

        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)

    if flatten_output:
        return viterbi_paths[0], viterbi_scores[0]

    return viterbi_paths, viterbi_scores
