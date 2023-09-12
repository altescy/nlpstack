"""
A pipeline is a sequence of processing steps that can be applied to a sequence
of inputs. Each step in the pipeline is a `Pipeline` object. The pipeline
itself is also a `Pipeline` object, so pipelines can be composed together.
"""

from collections import abc
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Iterable, Iterator, List, Sequence, TypeVar

from nlpstack.common.iterutil import SizedIterator, batched

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")


class Pipeline(Generic[S, T]):
    """
    A base class for pipelines.

    To define a pipeline, inherit this class and implement the `apply` method, which
    applies the pipeline to a single input. The `apply_batch` method is implemented
    in terms of `apply`, and can be overridden for efficiency.

    Here is an example of pipeline that tokenizes a string by splitting on spaces:

    Example:
        >>> class MyPipeline(Pipeline[str, List[str]]):
        >>>     def apply(self, input: str) -> List[str]:
        >>>         return input.split()

    To apply the pipeline to a sequence of inputs, call the pipeline object as a
    function. The `batch_size` argument controls the number of inputs to process
    at a time. The `max_workers` argument controls the number of threads to use
    for multi-thread processing. For example, to apply the pipeline to a sequence
    of inputs, 100 at a time, using 4 threads, do the following:

    Example:
        >>> pipeline = MyPipeline()
        >>> inputs = ["This is a test.", "This is another test.", ...]
        >>> outputs = pipeline(inputs, batch_size=100, max_workers=4)

    Chaining pipelines together is done using the `|` operator. For example, to
    create a pipeline consisting of three steps, `first_step`, `second_step`, and
    `third_step`, do the following:

    Example:
        >>> first_step = FirstPipeline()
        >>> second_step = SecondPipeline()
        >>> third_step = ThirdPipeline()
        >>> pipeline = first_step | second_step | third_step
    """

    def apply(self, input: S) -> T:
        """
        Apply the pipeline to a single input.

        Args:
            input: The input.

        Returns:
            The output.
        """

        raise NotImplementedError

    def apply_batch(self, batch: Sequence[S]) -> List[T]:
        """
        Apply the pipeline to a batch of inputs.

        Args:
            batch: The batch of inputs.

        Returns:
            The batch of outputs.
        """

        return list(map(self.apply, batch))

    def __call__(
        self,
        inputs: Iterable[S],
        *,
        batch_size: int = 1,
        max_workers: int = 1,
    ) -> Iterator[T]:
        """
        Apply the pipeline to a sequence of inputs.

        Args:
            inputs: The sequence of inputs.
            batch_size: The batch size. Defaults to `1`.
            max_workers: The maximum number of workers to use for multi-thread
                processing. Defaults to `1`.

        Returns:
            An iterator over the outputs.
        """

        def iterator() -> Iterator[T]:
            if max_workers < 2:
                for batch in batched(inputs, batch_size):
                    yield from self.apply_batch(batch)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for results in executor.map(self.apply_batch, batched(inputs, batch_size)):
                        yield from results

        if isinstance(inputs, abc.Sized):
            return SizedIterator(iterator(), len(inputs))

        return iterator()

    def __or__(self, other: "Pipeline[T, U]") -> "Pipeline[S, U]":
        return ComposePipeline(self, other)


class ComposePipeline(Pipeline[S, U]):
    """
    A pipeline that is the composition of two pipelines.

    Args:
        first: The first pipeline.
        second: The second pipeline.
    """

    def __init__(self, first: Pipeline[S, T], second: Pipeline[T, U]):
        self.first = first
        self.second = second

    def apply(self, input: S) -> U:
        return self.second.apply(self.first.apply(input))

    def apply_batch(self, batch: Sequence[S]) -> List[U]:
        return self.second.apply_batch(self.first.apply_batch(batch))
