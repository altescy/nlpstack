"""
A pipeline is a sequence of processing steps that can be applied to a sequence
of inputs. Each step in the pipeline is a `Pipeline` object. The pipeline
itself is also a `Pipeline` object, so pipelines can be composed together.
"""

from collections import abc
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generic, Iterable, Iterator, List, Optional, Sequence, TypeVar

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

    Args:
        batch_size: The batch size. Defaults to `1`.
        max_workers: The maximum number of workers to use for multi-thread
            processing. Defaults to `1`.
    """

    def __init__(
        self,
        *,
        batch_size: int = 1,
        max_workers: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self._batch_size = batch_size
        self._max_workers = max_workers

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
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
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

        batch_size = batch_size or self._batch_size
        max_workers = max_workers or self._max_workers

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

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

    @classmethod
    def from_callable(cls, func: Callable[[S], T]) -> "Pipeline[S, T]":
        """
        Create a pipeline from a callable.

        Args:
            func: The callable.

        Returns:
            The pipeline.
        """

        return CallablePipeline(func)


class ComposePipeline(Pipeline[S, U]):
    """
    A pipeline that is the composition of two pipelines.

    Args:
        first: The first pipeline.
        second: The second pipeline.
        batch_size: The batch size. Defaults to `1`.
        max_workers: The maximum number of workers to use for multi-thread
            processing. Defaults to `1`.
    """

    def __init__(
        self,
        first: Pipeline[S, T],
        second: Pipeline[T, U],
        *,
        batch_size: int = 1,
        max_workers: int = 1,
    ):
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self.first = first
        self.second = second

    def apply(self, input: S) -> U:
        return self.second.apply(self.first.apply(input))

    def apply_batch(self, batch: Sequence[S]) -> List[U]:
        return self.second.apply_batch(self.first.apply_batch(batch))


class CallablePipeline(Pipeline[S, T]):
    """
    A pipeline that can be created from a callable.

    Args:
        func: The callable.
        batch_size: The batch size. Defaults to `1`.
        max_workers: The maximum number of workers to use for multi-thread
            processing. Defaults to `1`.
    """

    def __init__(
        self,
        func: Callable[[S], T],
        *,
        batch_size: int = 1,
        max_workers: int = 1,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._func = func

    def apply(self, input: S) -> T:
        return self._func(input)


class ChainPipeline(Pipeline[S, S]):
    """
    A pipeline that is the composition of multiple pipelines.
    Note that each pipeline must have the same input and output types.

    Args:
        steps: The pipelines.
        batch_size: The batch size. Defaults to `1`.
        max_workers: The maximum number of workers to use for multi-thread
    """

    def __init__(
        self,
        *steps: Pipeline[S, S],
        batch_size: int = 1,
        max_workers: int = 1,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._steps = steps

    def apply(self, input: S) -> S:
        output = input
        for step in self._steps:
            output = step.apply(output)
        return output

    def apply_batch(self, batch: Sequence[S]) -> List[S]:
        output = list(batch)
        for step in self._steps:
            output = step.apply_batch(output)
        return output


class PassThroughPipeline(Pipeline[S, S]):
    """
    A pipeline that does nothing.
    """

    def apply(self, input: S) -> S:
        return input
