import itertools
from typing import Callable, Iterable, Iterator, List, TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batched_iterator(
    iterable: Iterable[T],
    batch_size: int,
) -> Iterator[Iterator[T]]:
    iterator = iter(iterable)
    stop = False

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
            yield itertools.chain([next(iterator)], consume(batch_size - 1))
        except StopIteration:
            break


def iter_with_callback(
    iterable: Iterable[T],
    callback: Callable[[T], None],
) -> Iterator[T]:
    for item in iterable:
        yield item
        callback(item)
