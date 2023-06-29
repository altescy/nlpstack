import tempfile
from typing import Any, ClassVar, Dict, Sequence

import torch


class TorchPicklable:  # type: ignore[misc]
    cuda_dependent_attributes: ClassVar[Sequence[str]] = []

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        cuda_attrs: Dict[str, Any] = {}
        for attr in self.cuda_dependent_attributes:
            if attr in state:
                cuda_attrs[attr] = state.pop(attr)

        with tempfile.SpooledTemporaryFile() as f:
            torch.save(cuda_attrs, f)
            f.seek(0)
            state["__cuda_dependent_attributes__"] = f.read()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        with tempfile.SpooledTemporaryFile() as f:
            f.write(state.pop("__cuda_dependent_attributes__"))
            f.seek(0)
            cuda_attrs = torch.load(f, map_location=torch.device("cpu"))

        state.update(cuda_attrs)
        self.__dict__.update(state)
