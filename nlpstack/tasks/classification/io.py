import json
from typing import Iterator, Optional, Sequence

import minato

from .data import ClassificationExample, ClassificationPrediction


class JsonlReader:
    def __init__(
        self,
        text_field: str = "text",
        label_field: str = "label",
    ) -> None:
        self.text_field = text_field
        self.label_field = label_field

    def __call__(self, filename: str) -> Iterator[ClassificationExample]:
        with minato.open(filename) as jsonlfile:
            for line in jsonlfile:
                example = json.loads(line)
                yield ClassificationExample(
                    text=example[self.text_field],
                    label=example[self.label_field],
                    metadata=example,
                )


class JsonlWriter:
    def __init__(
        self,
        additional_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self.additional_fields = additional_fields or []

    def __call__(
        self,
        output_filename: str,
        predictions: Iterator[ClassificationPrediction],
    ) -> None:
        with minato.open(output_filename, "w") as jsonlfile:
            for prediction in predictions:
                output = {"label": prediction.label}
                for field in self.additional_fields:
                    if not prediction.metadata:
                        raise ValueError("metadata is not available")
                    if field not in prediction.metadata:
                        raise ValueError(f"metadata does not have {field} field")
                    output[field] = prediction.metadata[field]
                jsonlfile.write(json.dumps(output, ensure_ascii=False) + "\n")
