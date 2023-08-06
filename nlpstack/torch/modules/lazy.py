from typing import Any, Optional

import torch


class LazyLinearOutput(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Linear):
    """
    A linear layer whose output dimension is not known at initialization.

    Args:
        in_features: The dimension of the input.
        bias: Whether to use bias. Defaults to `True`.
    """

    cls_to_become = torch.nn.Linear  # type: ignore[assignment]
    weight: torch.nn.UninitializedParameter
    bias: torch.nn.UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self,
        in_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=in_features, out_features=0, bias=bias)
        self.weight = torch.nn.UninitializedParameter()
        if bias:
            self.bias = torch.nn.UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.out_features != 0:
            super().reset_parameters()

    def initialize_parameters(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the parameters of the layer.

        Args:
            out_features (int): The dimension of the output.
        """
        if self.has_uninitialized_params():
            num_embeddings = kwargs.get("out_features", 0)
            with torch.no_grad():
                self.out_features = num_embeddings
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


class LazyEmbedding(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Embedding):
    """
    An embedding layer whose number of embeddings and padding index is not known at initialization.

    Args:
        embedding_dim: The size of each embedding vector
        max_norm: If given, each embedding vector with norm larger than `max_norm`` is renormalized to have norm `max_norm`.
        norm_type: The p of the p-norm to compute for the `max_norm` option. Default `2.0`.
        scale_grad_by_freq:  If given, this will scale gradients by the inverse of frequency of the words in the mini-batch.
            Defaults to`False`.
        sparse: If `True`, gradient w.r.t. `weight` matrix will be a sparse tensor.
        freeze: Whether to freeze the embedding. Defaults to `False`.
    """

    cls_to_become = torch.nn.Embedding  # type: ignore[assignment]
    weight: torch.nn.UninitializedParameter

    def __init__(
        self,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__(
            num_embeddings=0,
            embedding_dim=embedding_dim,
            padding_idx=None,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.weight = torch.nn.UninitializedParameter(requires_grad=not freeze)

    def reset_parameters(self, weight: Optional[torch.Tensor] = None) -> None:
        if not self.has_uninitialized_params() and self.num_embeddings != 0:
            if weight is None:
                torch.nn.init.xavier_uniform_(self.weight)
            else:
                self.weight.copy_(weight)
            self._fill_padding_idx_with_zero()

    def initialize_parameters(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.has_uninitialized_params():
            num_embeddings = kwargs.get("num_embeddings", 0)
            padding_idx = kwargs.get("padding_idx", None)
            weight = kwargs.get("weight", None)
            with torch.no_grad():
                self.num_embeddings = num_embeddings
                self.padding_idx = padding_idx
                self.weight.materialize((self.num_embeddings, self.embedding_dim))
                self.reset_parameters(weight)
