from typing import Any, Mapping, TypeVar, Union, cast

import numpy
import torch

T = TypeVar("T")


def get_mask_from_text(text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.BoolTensor:
    """
    :param text: Mapping[str, Mapping[str, torch.nn.LongTensor]]
    :return: torch.BoolTensor
    """
    for inputs in text.values():
        if "mask" in inputs:
            return cast(torch.BoolTensor, inputs["mask"].bool())
    raise ValueError("No mask found in text")


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(obj: T, device: Union[int, torch.device]) -> T:
    device = int_to_device(device)

    if isinstance(obj, numpy.ndarray):
        return cast(T, torch.from_numpy(obj).to(device=device))
    if isinstance(obj, torch.Tensor):
        return cast(T, obj if obj.device == device else obj.to(device=device))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return cast(T, obj)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = move_to_device(item, device)
        return cast(T, obj)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return cast(T, obj.__class__(*(move_to_device(item, device) for item in obj)))
    elif isinstance(obj, tuple):
        return cast(T, tuple(move_to_device(item, device) for item in obj))

    return obj


def tensor_to_numpy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = tensor_to_numpy(value)
        return obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = tensor_to_numpy(item)
        return obj
    elif isinstance(obj, tuple):
        return tuple(tensor_to_numpy(item) for item in obj)

    return obj
