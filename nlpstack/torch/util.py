import math
import random
from logging import getLogger
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy
import torch

logger = getLogger(__name__)

T = TypeVar("T")
TensorType = TypeVar("TensorType", bound=torch.Tensor)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_mask_from_text(text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.BoolTensor:
    """
    :param text: Mapping[str, Mapping[str, torch.nn.LongTensor]]
    :return: torch.BoolTensor
    """
    for inputs in text.values():
        if "mask" in inputs:
            return cast(torch.BoolTensor, inputs["mask"].bool())
    raise ValueError("No mask found in text")


def get_token_ids_from_text(text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.LongTensor:
    for inputs in text.values():
        if "token_ids" in inputs:
            return cast(torch.LongTensor, inputs["token_ids"].long())
    raise ValueError("No token_ids found in text")


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
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: Union[torch.FloatTensor, torch.BoolTensor],
    average: Literal["token", "batch", "none"] = "batch",
    label_smoothing: Optional[float] = None,
    gamma: Optional[float] = None,
    alpha: Optional[Union[float, List[float], torch.FloatTensor]] = None,
) -> torch.FloatTensor:
    if average not in {"none", "token", "batch"}:
        raise ValueError(f"Got average f{average}, expected one of 'none', 'token', or 'batch'")

    # make sure weights are float
    weights = weights.to(logits.dtype)  # type: ignore[assignment]
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1.0 - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # shape : (2,)
            alpha_factor = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device)

        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)

            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(
                ("alpha must be float, list of float, or torch.FloatTensor, {} provided.").format(type(alpha))
            )
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor  # type: ignore[assignment]

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        smoothed_targets = torch.full_like(log_probs_flat, smoothing_value).scatter_(
            -1, targets_flat, 1.0 - label_smoothing + smoothing_value
        )
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # Shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # Shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # Shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
        return cast(torch.FloatTensor, per_batch_loss.sum() / num_non_empty_sequences)
    elif average == "token":
        return cast(
            torch.FloatTensor,
            negative_log_likelihood.sum()
            / (weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)),
        )
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        return cast(torch.FloatTensor, per_batch_loss)


def replace_masked_values(tensor: TensorType, mask: torch.BoolTensor, replace_with: float) -> TensorType:
    if tensor.dim() != mask.dim():
        raise ValueError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return cast(TensorType, tensor.masked_fill(~mask, replace_with))


def masked_mean(
    vector: TensorType,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> TensorType:
    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return cast(TensorType, value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float)))


def masked_max(
    vector: TensorType,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> TensorType:
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return cast(TensorType, max_value)


def masked_pool(
    inputs: TensorType,
    mask: Optional[torch.BoolTensor] = None,
    method: Literal["mean", "max", "sum", "hier"] = "mean",
    dim: int = 1,
    keepdim: bool = False,
    window_size: Optional[int] = None,
) -> TensorType:
    if mask is None:
        mask = cast(torch.BoolTensor, inputs.new_ones(inputs.size()).bool())

    if method == "mean":
        return masked_mean(inputs, mask, dim=dim, keepdim=keepdim)
    if method == "max":
        return masked_max(inputs, mask, dim=dim, keepdim=keepdim)
    if method == "sum":
        return cast(TensorType, replace_masked_values(inputs, mask, 0.0).sum(dim=dim, keepdim=keepdim))
    if method == "hier":
        if window_size is None:
            raise ValueError("window_size must be specified for hier pooling")
        if inputs.size(1) <= window_size:
            return masked_mean(inputs, mask, dim=dim, keepdim=keepdim)
        inputs = cast(
            TensorType,
            torch.nn.functional.avg_pool1d(inputs.transpose(1, 2), window_size).transpose(1, 2),
        )
        mask = cast(
            torch.BoolTensor,
            torch.nn.functional.max_pool1d(mask.float().transpose(1, 2), window_size).transpose(1, 2).bool(),
        )
        return masked_mean(inputs, mask, dim=dim, keepdim=keepdim)

    raise ValueError(f"Invalid pooling method: {method}")


def masked_softmax(
    vector: TensorType,
    mask: torch.BoolTensor,
    dim: int = -1,
    memory_efficient: bool = False,
) -> TensorType:
    while mask.dim() < vector.dim():
        mask = cast(torch.BoolTensor, mask.unsqueeze(1))
    if not memory_efficient:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype))
    else:
        masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
        result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return cast(TensorType, result)


def weighted_sum(matrix: TensorType, attention: torch.Tensor) -> TensorType:
    if attention.dim() == 2 and matrix.dim() == 3:
        return cast(TensorType, attention.unsqueeze(1).bmm(matrix).squeeze(1))
    if attention.dim() == 3 and matrix.dim() == 3:
        return cast(TensorType, attention.bmm(matrix))
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = cast(TensorType, matrix.unsqueeze(1))
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = cast(TensorType, matrix.expand(*expanded_size))
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return cast(TensorType, intermediate.sum(dim=-2))


def _get_combination(combination: str, tensors: Sequence[TensorType]) -> TensorType:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return cast(TensorType, first_tensor * second_tensor)
        elif operation == "/":
            return cast(TensorType, first_tensor / second_tensor)
        elif operation == "+":
            return cast(TensorType, first_tensor + second_tensor)
        elif operation == "-":
            return cast(TensorType, first_tensor - second_tensor)
        else:
            raise ValueError("Invalid operation: " + operation)


def combine_tensors(combination: str, tensors: Sequence[TensorType]) -> TensorType:
    if len(tensors) > 9:
        raise ValueError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    to_concatenate: List[torch.Tensor] = [_get_combination(piece, tensors) for piece in combination.split(",")]
    return cast(TensorType, torch.cat(to_concatenate, dim=-1))


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    return tensor.get_device()


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return cast(torch.Tensor, torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1)  # type: ignore[attr-defined]
    return torch.arange(0, size, dtype=torch.long)


def flatten_and_batch_shift_indices(indices: TensorType, sequence_length: int) -> TensorType:
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ValueError(f"All elements in indices should be in range (0, {sequence_length - 1})")
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return cast(TensorType, offset_indices)


def batched_index_select(
    target: TensorType,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> TensorType:
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return cast(TensorType, selected_targets)


def batched_span_select(target: TensorType, spans: torch.LongTensor) -> Tuple[TensorType, torch.BoolTensor]:
    # Shape: (batch_size, num_spans, 1)
    # Shape: (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)  # type: ignore[no-untyped-call]

    # Shape: (batch_size, num_spans, 1)
    span_widths = span_ends - span_starts

    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(1, 1, -1)
    # Shape: (batch_size, num_spans, max_batch_span_width)
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def info_value_of_dtype(dtype: torch.dtype) -> Union[torch.finfo, torch.iinfo]:
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype) -> Union[float, int]:
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype) -> Union[float, int]:
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype) -> Union[float, int]:
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in (torch.float, torch.double):
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def add_positional_features(
    tensor: TensorType,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> TensorType:
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.float()
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.float()

    log_timescale_increments = math.log(max_timescale / min_timescale) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # Shape: (timesteps, 2 * num_timescales)
    sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if hidden_dim % 2 != 0:
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return cast(TensorType, tensor + sinusoids.unsqueeze(0))


def fold(tensor: TensorType, max_length: int) -> TensorType:
    assert tensor.dim() >= 2

    batch_size = tensor.size(0)
    original_length = tensor.size(1)
    if original_length <= max_length:
        return tensor

    num_segments, remainder = divmod(original_length, max_length)
    if remainder > 0:
        num_segments += 1
        tensor = cast(TensorType, torch.cat([tensor[:, :-remainder], tensor[:, -max_length:]], dim=1))
    rest_shape = tensor.size()[2:]
    return cast(TensorType, tensor.reshape(batch_size * num_segments, max_length, *rest_shape))


def unfold(tensor: TensorType, original_length: int) -> TensorType:
    assert tensor.dim() >= 2

    folded_length = tensor.size(1)
    if original_length <= folded_length:
        return tensor

    num_segments, remainder = divmod(original_length, folded_length)
    if remainder > 0:
        num_segments += 1
    batch_size = tensor.size(0) // num_segments
    unfolded_length = original_length + (remainder > 0) * (folded_length - remainder)
    rest_shape = tensor.size()[2:]
    x = cast(TensorType, tensor.reshape(batch_size, unfolded_length, *rest_shape))
    if remainder > 0:
        x = cast(TensorType, torch.cat([x[:, :-folded_length], x[:, -remainder:]], dim=1))
    return x


@overload
def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: int,
    tag_observations: Optional[List[int]] = ...,
    allowed_start_transitions: Optional[torch.Tensor] = ...,
    allowed_end_transitions: Optional[torch.Tensor] = ...,
) -> Tuple[List[List[int]], torch.Tensor]: ...


@overload
def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: None = ...,
    tag_observations: Optional[List[int]] = ...,
    allowed_start_transitions: Optional[torch.Tensor] = ...,
    allowed_end_transitions: Optional[torch.Tensor] = ...,
) -> Tuple[List[int], torch.Tensor]: ...


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
        viterbi_path = [int(best_paths[i].item())]
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


def convert_to_toeplitz(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() != 1:
        raise ValueError(f"Number of dimensions of inputs must be equal to 1 (actual={inputs.dim()}).")

    num_elements = inputs.size(0)
    if num_elements % 2 != 1:
        raise ValueError(f"Size of inputs must be an odd number. (actual={num_elements})")

    n = (num_elements + 1) // 2
    r = num_elements // 2

    output = torch.nn.functional.pad(inputs, (0, n))
    output = output.tile(n)
    output = output[:-n]
    output = output.reshape(n, -1)
    output = output[:, r:-r]

    return output
