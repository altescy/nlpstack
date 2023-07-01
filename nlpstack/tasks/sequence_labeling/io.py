import itertools
import json
from typing import Iterable, Iterator, Literal, Optional, Sequence

import minato

from nlpstack.data import Token

from .types import SequenceLabelingExample, SequenceLabelingPrediction
from .util import LabelEncoding, convert_encoding


class JsonlReader:
    def __init__(
        self,
        text_field: str = "text",
        label_field: str = "labels",
    ) -> None:
        self.text_field = text_field
        self.label_field = label_field

    def __call_(self, fielname: str) -> Iterator[SequenceLabelingExample]:
        with minato.open(fielname) as jsonlfile:
            for line in jsonlfile:
                example = json.loads(line)
                yield SequenceLabelingExample(
                    text=example[self.text_field],
                    labels=example.get(self.label_field),
                )


class JsonlWriter:
    def __init__(
        self,
        additional_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self.additional_fields = additional_fields or []

    def __call__(
        self,
        filename: str,
        predictions: Iterable[SequenceLabelingPrediction],
    ) -> None:
        with minato.open(filename, "w") as jsonlfile:
            for prediction in predictions:
                output = {"tokens": prediction.tokens, "labels": prediction.labels}
                for field in self.additional_fields:
                    if prediction.metadata is None:
                        raise ValueError("metadata is not available")
                    if field not in prediction.metadata:
                        raise ValueError(f"metadata does not have {field} field")
                    output[field] = prediction.metadata[field]
                jsonlfile.write(json.dumps(output, ensure_ascii=False) + "\n")


class Conll2003Reader:
    def __init__(
        self,
        label_field: Literal["ner", "pos", "chunk"],
        label_encoding: LabelEncoding = "IOB1",
        convert_to_label_encoding: Optional[LabelEncoding] = None,
        use_postag_as_feature: bool = False,
    ) -> None:
        self.label_field = label_field
        self.label_encoding = label_encoding
        self.convert_to_label_encoding = convert_to_label_encoding
        self.use_postag_as_feature = use_postag_as_feature

    @staticmethod
    def _is_divider(line: str) -> bool:
        empty_line = line.strip() == ""
        if empty_line:
            return True
        else:
            first_token = line.split()[0]
            if first_token == "-DOCSTART-":
                return True
            else:
                return False

    def __call__(self, filename: str) -> Iterator[SequenceLabelingExample]:
        with minato.open(filename) as conllfile:
            line_chunks = (
                line for is_divider, line in itertools.groupby(conllfile, self._is_divider) if not is_divider
            )
            for lines in line_chunks:
                fields = [list(field) for field in zip(*(line.strip().split() for line in lines))]
                _tokens, pos_tags, chunk_tags, ner_tags = fields

                if self.use_postag_as_feature:
                    tokens = list(itertools.starmap(Token, zip(_tokens, pos_tags)))
                else:
                    tokens = [Token(surface) for surface in _tokens]

                if self.label_field == "ner":
                    labels = ner_tags
                elif self.label_field == "pos":
                    labels = pos_tags
                elif self.label_field == "chunk":
                    labels = chunk_tags
                else:
                    raise ValueError(f"Unknown label field: {self.label_field}")

                if self.convert_to_label_encoding is not None:
                    labels = convert_encoding(
                        labels,
                        source_encoding=self.label_encoding,
                        target_encoding=self.convert_to_label_encoding,
                    )

                yield SequenceLabelingExample(
                    text=tokens,
                    labels=labels,
                )
