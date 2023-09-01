from typing import Literal, Optional, cast

import numpy


def masked_pool(
    embeddings: numpy.ndarray,
    mask: Optional[numpy.ndarray] = None,
    pooling: Literal["mean", "max", "min", "sum", "hier", "first", "last"] = "mean",
    normalize: bool = False,
    window_size: Optional[int] = None,
) -> numpy.ndarray:
    """
    Pool embeddings with a mask.

    Args:
        embeddings: Embeddings to pool of shape (batch_size, sequence_length, embedding_size).
        mask: Mask of shape (batch_size, sequence_length).
        pooling: Pooling method. Defaults to `"mean"`.
        normalize: Whether to normalize the embeddings before pooling. Defaults to `False`.
        window_size: Window size for hierarchical pooling. Defaults to `None`.
    """

    batch_size, sequence_length, embedding_size = embeddings.shape

    if normalize:
        embeddings = embeddings / (numpy.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-13)

    if mask is None:
        mask = numpy.ones((batch_size, sequence_length), dtype=bool)

    if pooling == "mean":
        return cast(numpy.ndarray, embeddings.sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-13))

    if pooling == "max":
        embeddings[~mask] = float("-inf")
        return cast(numpy.ndarray, embeddings.max(axis=1))

    if pooling == "min":
        embeddings[~mask] = float("inf")
        return cast(numpy.ndarray, embeddings.min(axis=1))

    if pooling == "sum":
        return cast(numpy.ndarray, embeddings.sum(axis=1))

    if pooling == "first":
        return embeddings[:, 0, :]

    if pooling == "last":
        batch_indices = numpy.arange(batch_size)
        last_positions = mask.cumsum(axis=1).argmax(axis=1)
        return embeddings[batch_indices, last_positions, :]

    if pooling == "hier":

        def _hierarchical_pooling(vectors: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
            assert window_size is not None
            vectors = vectors[mask]
            if len(vectors) < window_size:
                return cast(numpy.ndarray, vectors.mean(0))
            output = -numpy.inf * numpy.ones(embedding_size)
            for offset in range(len(vectors) - window_size + 1):
                window = vectors[offset : offset + window_size]
                output = numpy.maximum(output, window.mean(0))
            return output

        return numpy.array([_hierarchical_pooling(x, m) for x, m in zip(embeddings, mask)])

    raise ValueError(
        f"pooling must be one of 'mean', 'max', 'min', 'sum', 'hier', 'first', or 'last', but got {pooling}"
    )
