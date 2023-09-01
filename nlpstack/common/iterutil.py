import itertools
import math
from collections import abc
from typing import Any, Callable, Generic, Iterable, Iterator, List, TypeVar

T = TypeVar("T")


class SizedIterator(Generic[T]):
    """
    A wrapper for an iterator that knows its size.

    Args:
        iterator: The iterator.
        size: The size of the iterator.
    """

    def __init__(self, iterator: Iterator[T], size: int):
        self.iterator = iterator
        self.size = size

    def __iter__(self) -> Iterator[T]:
        return self.iterator

    def __next__(self) -> T:
        return next(self.iterator)

    def __len__(self) -> int:
        return self.size


def batched(iterable: Iterable[T], batch_size: int, drop_last: bool = False) -> Iterator[List[T]]:
    """
    Batch an iterable into lists of the given size.

    Args:
        iterable: The iterable.
        batch_size: The size of each batch.
        drop_last: Whether to drop the last batch if it is smaller than the given size.

    Returns:
        An iterator over batches.
    """

    def iterator() -> Iterator[List[T]]:
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch and not drop_last:
            yield batch

    if isinstance(iterable, abc.Sized):
        num_batches = math.ceil(len(iterable) / batch_size)
        return SizedIterator(iterator(), num_batches)

    return iterator()


def batched_iterator(iterable: Iterable[T], batch_size: int) -> Iterator[Iterator[T]]:
    """
    Batch an iterable into iterators of the given size.

    Args:
        iterable: The iterable.
        batch_size: The size of each batch.

    Returns:
        An iterator over batches.
    """

    def iterator() -> Iterator[Iterator[T]]:
        stop = False
        batch_progress = 0

        def iterator_wrapper() -> Iterator[T]:
            nonlocal batch_progress
            for item in iterable:
                yield item
                batch_progress += 1

        iterator = iterator_wrapper()

        def consume(n: int) -> Iterator[T]:
            for _ in range(n):
                try:
                    yield next(iterator)
                except StopIteration:
                    nonlocal stop
                    stop = True
                    break

        while not stop:
            try:
                batch_progress = 0
                yield itertools.chain([next(iterator)], consume(batch_size - 1))
                for _ in range(batch_size - batch_progress):
                    next(iterator)
            except StopIteration:
                break

    if isinstance(iterable, abc.Sized):
        num_batches = math.ceil(len(iterable) / batch_size)
        return SizedIterator(iterator(), num_batches)

    return iterator()


def iter_with_callback(
    iterable: Iterable[T],
    callback: Callable[[T], Any],
) -> Iterator[T]:
    """
    Iterate over an iterable and call a callback for each item.

    Args:
        iterable: The iterable.
        callback: The callback to call for each item.

    Returns:
        An iterator over the iterable.
    """

    def iterator() -> Iterator[T]:
        for item in iterable:
            yield item
            callback(item)

    if isinstance(iterable, abc.Sized):
        return SizedIterator(iterator(), len(iterable))

    return iterator()
