import json
from typing import Iterator, Optional

import minato

from .data import ClassificationExample, ClassificationPrediction


class JsonlReader:
    def __init__(
        self,
        train_filename: str,
        valid_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        *,
        text_field: str = "text",
        label_field: str = "label",
    ) -> None:
        self.train_filename = train_filename
        self.valid_filename = valid_filename
        self.test_filename = test_filename

    def __call__(self, subset: str) -> Iterator[ClassificationExample]:
        filename: Optional[str]
        if subset == "train":
            filename = self.train_filename
        elif subset == "valid":
            filename = self.valid_filename
        elif subset == "test":
            filename = self.test_filename
        else:
            filename = subset

        if filename is None:
            return iter([])

        with minato.open(filename) as jsonlfile:
            for line in jsonlfile:
                example = json.loads(line)
                yield ClassificationExample(
                    text=example["text"],
                    label=example["label"],
                    metadata=example,
                )


class JsonlWriter:
    def __call__(
        self,
        output_filename: str,
        predictions: Iterator[ClassificationPrediction],
    ) -> None:
        with minato.open(output_filename, "w") as jsonlfile:
            for prediction in predictions:
                jsonlfile.write(json.dumps({"label": prediction.label}, ensure_ascii=False) + "\n")
