from __future__ import annotations

import tempfile
from typing import Any, ClassVar

import torch


class TorchPicklable:  # type: ignore[misc]
    cuda_dependent_attributes: ClassVar[list[str]] = []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        cuda_attrs: dict[str, Any] = {}
        for attr in self.cuda_dependent_attributes:
            if attr in state:
                cuda_attrs[attr] = state.pop(attr)

        with tempfile.SpooledTemporaryFile() as f:
            torch.save(cuda_attrs, f)
            f.seek(0)
            state["__cuda_dependent_attributes__"] = f.read()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        with tempfile.SpooledTemporaryFile() as f:
            f.write(state.pop("__cuda_dependent_attributes__"))
            f.seek(0)
            if not torch.cuda.is_available():
                cuda_attrs = torch.load(f, map_location=torch.device("cpu"))
            else:
                cuda_attrs = torch.load(f)

        state.update(cuda_attrs)
        self.__dict__.update(state)
