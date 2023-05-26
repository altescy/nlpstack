from __future__ import annotations

from typing import Generic, Mapping, Sequence, TypeVar

Inference = TypeVar("Inference")


class Metric(Generic[Inference]):
    def update(self, inference: Inference) -> None:
        raise NotImplementedError

    def compute(self) -> Mapping[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class MultiMetrics(Metric[Inference]):
    def __init__(
        self,
        metrics: Sequence[Metric[Inference]],
    ) -> None:
        self.metrics = metrics

        # check metric name conflicts
        names: set[str] = set()
        for metric in metrics:
            names_ = set(metric.compute())
            if names_ & names:
                raise ValueError(f"metric name conflict: {names_ & names}")
            names |= names_

    def update(self, inference: Inference) -> None:
        for metric in self.metrics:
            metric.update(inference)

    def compute(self) -> Mapping[str, float]:
        metrics: dict[str, float] = {}
        for metric in self.metrics:
            metrics.update(metric.compute())
        return metrics

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()


class EmptyMetric(Metric[Inference]):
    def update(self, inference: Inference) -> None:
        pass

    def compute(self) -> Mapping[str, float]:
        return {}

    def reset(self) -> None:
        pass
