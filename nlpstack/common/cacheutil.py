import functools
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def cached_property(func: Callable[[Any], T], **kwargs: Any) -> property:
    """
    A decorator that converts a method into a lazy property. The method wrapped
    is called the first time to retrieve the result and then that calculated result
    is used the next time you access the value. The result is cached at the method level
    not at the instance level as in `functools.cached_property`.

    This implementation mimics a part of `functools.cached_property` but without caching
    at the instance level, which might be useful in certain scenarios where you don't
    want cache to be tied with instance lifecycle.

    Args:
        func: The function to be decorated.

    Returns:
        property: A property that represents `func` as a cached property.

    """
    cached_func = functools.lru_cache(**kwargs)(func)
    return property(functools.update_wrapper(cached_func, func))
