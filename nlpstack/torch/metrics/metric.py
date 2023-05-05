from __future__ import annotations

from typing import Any


class Metric:
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
