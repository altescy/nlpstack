from typing import Any, Generic, Iterable, Iterator, Literal, Mapping, Optional, Sequence, TypeVar

Self = TypeVar("Self", bound="Rune")

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")
SetupMode = Literal["training", "prediction", "evaluation"]


class Rune(Generic[Example, Prediction]):
    def train(
        self: Self,
        train_dataset: Sequence[Example],
        valid_dataset: Optional[Sequence[Example]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def setup(self, mode: SetupMode, **kwargs: Any) -> None:
        pass

    def predict(self, dataset: Iterable[Example], **kwargs: Any) -> Iterator[Prediction]:
        raise NotImplementedError

    def evaluate(self, dataset: Iterable[Example], **kwargs: Any) -> Mapping[str, float]:
        raise NotImplementedError
