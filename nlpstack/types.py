from dataclasses import Field
from typing import ClassVar, Dict, Protocol, TypeVar


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field]]


T_Dataclass = TypeVar("T_Dataclass", bound=DataclassInstance)

Example = TypeVar("Example")
Inference = TypeVar("Inference")
Prediction = TypeVar("Prediction")
