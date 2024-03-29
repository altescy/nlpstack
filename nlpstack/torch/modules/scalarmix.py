from typing import List, Optional, Sequence

import torch
from torch.nn import Parameter, ParameterList


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.

    Note:
        This script is based on the AllenNLP implementation of ScalarMix:
        https://github.com/allenai/allennlp/blob/v2.10.0/allennlp/modules/scalar_mix.py
    """

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: Optional[Sequence[float]] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ValueError(
                "Length of initial_scalar_parameters {} differs "
                "from mixture_size {}".format(initial_scalar_parameters, mixture_size)
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable)
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors: Sequence[torch.Tensor], mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.
        When `do_layer_norm=False` the `mask` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ValueError(
                "{} tensors were passed, but the module was initialized to "
                "mix {} tensors.".format(len(tensors), self.mixture_size)
            )

        def _do_layer_norm(
            tensor: torch.Tensor,
            broadcast_mask: torch.Tensor,
            num_elements_not_masked: torch.Tensor,
        ) -> torch.Tensor:
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1e-13)

        normed_weights = torch.split(
            torch.nn.functional.softmax(torch.cat([parameter for parameter in self.scalar_parameters]), dim=0),
            split_size_or_sections=1,
        )

        pieces: List[torch.Tensor]
        if not self.do_layer_norm:
            pieces = [weight * tensor for weight, tensor in zip(normed_weights, tensors)]
            return self.gamma * sum(pieces)

        else:
            assert mask is not None
            broadcast_mask = mask.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask) * input_dim

            pieces = [
                weight * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                for weight, tensor in zip(normed_weights, tensors)
            ]
            return self.gamma * sum(pieces)
