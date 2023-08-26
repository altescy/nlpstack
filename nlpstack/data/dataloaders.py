import math
import random
from typing import Dict, Iterator, List, Sequence

from collatable.collator import Collator
from collatable.instance import Instance
from collatable.typing import DataArray


class BatchSampler:
    """
    A batch sampler is responsible for generating batches of indices from a dataset.
    """

    def get_batch_indices(self, dataset: Sequence[Instance]) -> Iterator[List[int]]:
        """
        Returns an iterator over batches of indices from the dataset.

        Args:
            dataset: The dataset to sample from.

        Returns:
            An iterator over batches of indices from the dataset.
        """
        raise NotImplementedError

    def get_num_batches(self, dataset: Sequence[Instance]) -> int:
        """
        Returns the number of batches in the dataset.

        Args:
            dataset: The dataset to sample from.

        Returns:
            The number of batches in the dataset.
        """
        raise NotImplementedError

    def get_batch_size(self) -> int:
        """
        Returns the batch size.

        Returns:
            The batch size.
        """
        raise NotImplementedError


class BasicBatchSampler(BatchSampler):
    """
    A basic batch sampler that generates batches of indices from a dataset.

    Args:
        batch_size: The batch size.
        shuffle: Whether to shuffle the dataset before sampling.
        drop_last: Whether to drop the last batch if it is smaller than the batch size.
    """

    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def get_batch_indices(self, dataset: Sequence[Instance]) -> Iterator[List[int]]:
        indices = list(range(len(dataset)))

        if self._shuffle:
            random.shuffle(indices)

        batch_indices: List[int] = []
        for index in indices:
            batch_indices.append(index)
            if len(batch_indices) == self._batch_size:
                yield batch_indices
                batch_indices = []

        if batch_indices and not self._drop_last:
            yield batch_indices

    def get_num_batches(self, dataset: Sequence[Instance]) -> int:
        if self._drop_last:
            return len(dataset) // self._batch_size
        return math.ceil(len(dataset) / self._batch_size)

    def get_batch_size(self) -> int:
        return self._batch_size


class BatchIterator:
    def __init__(
        self,
        dataset: Sequence[Instance],
        sampler: BatchSampler,
    ) -> None:
        self._dataset = dataset
        self._sampler = sampler
        self._collator = Collator()
        self._batch_indices = iter(self._sampler.get_batch_indices(self._dataset))

    def __len__(self) -> int:
        return self._sampler.get_num_batches(self._dataset)

    def __next__(self) -> Dict[str, DataArray]:
        batch_indices = next(self._batch_indices)
        return self._collator([self._dataset[i] for i in batch_indices])

    def __iter__(self) -> Iterator[Dict[str, DataArray]]:
        return self


class DataLoader:
    """
    A data loader is responsible for iterating over batches of instances from a dataset.

    Args:
        sampler: The batch sampler to use.
    """

    def __init__(
        self,
        sampler: BatchSampler,
    ) -> None:
        self._sampler = sampler

    def __call__(self, dataset: Sequence[Instance]) -> BatchIterator:
        """
        Returns an iterator over batches of instances from the dataset.

        Args:
            dataset: The dataset to iterate over.

        Returns:
            An iterator over batches of instances from the dataset.
        """
        return BatchIterator(dataset, self._sampler)

    def get_batch_size(self) -> int:
        """
        Returns the batch size.

        Returns:
            The batch size.
        """
        return self._sampler.get_batch_size()
