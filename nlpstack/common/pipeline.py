"""
A pipeline is a sequence of processing steps that can be applied to a sequence
of inputs. Each step in the pipeline is a `Pipeline` object. The pipeline
itself is also a `Pipeline` object, so pipelines can be composed together.
"""

from collections import abc
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

from mpire import WorkerPool

from nlpstack.common.iterutil import SizedIterator, batched

# inputs/outputs
S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")
# fixtures
E = TypeVar("E")
F = TypeVar("F")
# runtime params
P = TypeVar("P")
Q = TypeVar("Q")


class Pipeline(Generic[S, T, F, P]):
    """
    A base class for pipelines.

    To define a pipeline, inherit this class and implement the `apply` method, which
    applies the pipeline to a single input. The `apply_batch` method is implemented
    in terms of `apply`, and can be overridden for efficiency.

    Here is an example of pipeline that tokenizes a string by splitting on spaces:

    Example:
        >>> class MyPipeline(Pipeline[str, List[str], None, None]):
        >>>     fixtures = None
        >>>
        >>>     def apply_batch(self, inputs: List[str], fixtures: None, params: None) -> List[str]:
        >>>         return [x.split() for x in inputs]

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

    @property
    def fixtures(self) -> F:
        raise NotImplementedError

    def apply_batch(
        self,
        batch: Sequence[S],
        fixtures: F,
        params: Optional[P] = None,
    ) -> List[T]:
        """
        Apply the pipeline to a batch of inputs.

        Args:
            batch: The batch of inputs.
            fixtures: The fixtures to use.
            params: The runtime parameters.

        Returns:
            The batch of outputs.
        """

        raise NotImplementedError

    def __call__(
        self,
        inputs: Iterable[S],
        params: Optional[P] = None,
        *,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> Iterator[T]:
        """
        Apply the pipeline to a sequence of inputs.

        Args:
            inputs: The sequence of inputs.
            params: The runtime parameters.
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
                fixtures = self.fixtures
                for batch in batched(inputs, batch_size):
                    yield from self.apply_batch(batch, fixtures, params)
            else:

                def apply_batch(shared_objects: Tuple[F, Optional[P]], *batch: S) -> List[T]:
                    fixtures, params = shared_objects
                    return self.apply_batch(batch, fixtures, params)

                with WorkerPool(
                    n_jobs=max_workers,
                    shared_objects=(self.fixtures, params),
                ) as pool:
                    for results in pool.imap(
                        apply_batch,
                        batched(inputs, batch_size),
                    ):
                        yield from results

        if isinstance(inputs, abc.Sized):
            return SizedIterator(iterator(), len(inputs))

        return iterator()

    def __or__(self, other: "Pipeline[T, U, E, Q]") -> "Pipeline[S, U, Tuple[F, E], Tuple[P, Q]]":
        return ComposePipeline(self, other)

    @classmethod
    def from_callable(
        cls,
        func: Callable[[Sequence[S], F, Optional[Q]], List[T]],
        fixtures: F,
        **kwargs: Any,
    ) -> "Pipeline[S, T, F, Q]":
        """
        Create a pipeline from a callable.

        Args:
            func: The callable.
            fixtures: The fixtures to use.

        Returns:
            The pipeline.
        """

        return CallablePipeline(func, fixtures, **kwargs)


class ComposePipeline(Pipeline[S, U, Tuple[E, F], Tuple[P, Q]]):
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
        first: Pipeline[S, T, E, P],
        second: Pipeline[T, U, F, Q],
        *,
        batch_size: int = 1,
        max_workers: int = 1,
    ):
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self.first = first
        self.second = second

    def apply_batch(
        self,
        batch: Sequence[S],
        fixtures: Tuple[E, F],
        params: Optional[Tuple[Optional[P], Optional[Q]]] = None,
    ) -> List[U]:
        params = params or (None, None)
        return self.second.apply_batch(self.first.apply_batch(batch, fixtures[0], params[0]), fixtures[1], params[1])

    def __call__(
        self,
        inputs: Iterable[S],
        params: Optional[Tuple[Optional[P], Optional[Q]]] = None,
        *,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> Iterator[U]:
        params = params or (None, None)
        return self.second(
            self.first(
                inputs,
                params[0],
                batch_size=batch_size,
                max_workers=max_workers,
            ),
            params[1],
            batch_size=batch_size,
            max_workers=max_workers,
        )


class CallablePipeline(Pipeline[S, T, F, P]):
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
        func: Callable[[Sequence[S], F, Optional[P]], List[T]],
        fixtures: F,
        *,
        batch_size: int = 1,
        max_workers: int = 1,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._func = func
        self._fixtures = fixtures

    @property
    def fixtures(self) -> F:
        return self._fixtures

    def apply_batch(
        self,
        batch: Sequence[S],
        fixtures: F,
        params: Optional[P] = None,
    ) -> List[T]:
        return self._func(batch, fixtures, params)


class ChainPipeline(Pipeline[S, S, List[Any], List[Any]]):
    """
    A pipeline that is the composition of multiple pipelines.
    Note that each pipeline must have the same input and output types.

    Args:
        steps: The sequence of pipelines.
        batch_size: The batch size. Defaults to `1`.
        max_workers: The maximum number of workers to use for multi-thread
    """

    def __init__(
        self,
        steps: Sequence[Pipeline[S, S, Any, Any]],
        batch_size: int = 1,
        max_workers: int = 1,
    ) -> None:
        super().__init__(batch_size=batch_size, max_workers=max_workers)
        self._steps = steps or [PassThroughPipeline()]

    @property
    def fixtures(self) -> List[Any]:
        return [step.fixtures() for step in self._steps]

    def apply_batch(
        self,
        batch: Sequence[S],
        fixtures: List[Any],
        params: Optional[List[Any]] = None,
    ) -> List[S]:
        if params is not None and len(params) != len(self._steps):
            raise ValueError("params must have the same length as steps")
        params = params or [None] * len(self._steps)
        for s, f, p in zip(self._steps, fixtures, params):
            batch = s.apply_batch(batch, f, p)
        return list(batch)

    def __call__(
        self,
        inputs: Iterable[S],
        params: Optional[List[Any]] = None,
        *,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> Iterator[S]:
        if params is not None and len(params) != len(self._steps):
            raise ValueError("params must have the same length as steps")
        params = params or [None] * len(self._steps)
        output = self._steps[0](inputs, params[0], batch_size=batch_size, max_workers=max_workers)
        for step, param in zip(self._steps[1:], params[1:]):
            output = step(output, param, batch_size=batch_size, max_workers=max_workers)
        return output


class PassThroughPipeline(Pipeline[S, S, None, None]):
    """
    A pipeline that does nothing.
    """

    fixtures = None

    def apply_batch(
        self,
        batch: Sequence[S],
        fixtures: None,
        params: None = None,
    ) -> List[S]:
        return list(batch)
