import itertools
import math
from collections import abc
from typing import Any, Callable, Generic, Iterable, Iterator, List, TypeVar

T = TypeVar("T")


class SizedIterator(Generic[T]):
    def __init__(self, it: Iterator[T], size: int):
        self.it = it
        self.size = size

    def __iter__(self) -> Iterator[T]:
        return self.it

    def __next__(self) -> T:
        return next(self.it)

    def __len__(self) -> int:
        return self.size


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    def iterator() -> Iterator[List[T]]:
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    if isinstance(iterable, abc.Sized):
        num_batches = math.ceil(len(iterable) / batch_size)
        return SizedIterator(iterator(), num_batches)

    return iterator()


def batched_iterator(iterable: Iterable[T], batch_size: int) -> Iterator[Iterator[T]]:
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
    def iterator() -> Iterator[T]:
        for item in iterable:
            yield item
            callback(item)

    if isinstance(iterable, abc.Sized):
        return SizedIterator(iterator(), len(iterable))

    return iterator()
