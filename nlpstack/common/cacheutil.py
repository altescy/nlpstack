import functools
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def cached_property(func: Callable[[Any], T], **kwargs: Any) -> property:
    """
    This is a decorator to cache the result of a property.
    Unlike functools.cached_property, this decorator do not modify `__dict__` of the class.
    """
    cached_func = functools.lru_cache(**kwargs)(func)
    return property(functools.update_wrapper(cached_func, func))
