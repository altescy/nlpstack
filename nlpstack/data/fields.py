import functools
from typing import Any, Dict, Iterator, Mapping, Sequence, Union, cast

import numpy
from collatable.fields.adjacency_field import AdjacencyField  # noqa: F401
from collatable.fields.field import Field  # noqa: F401
from collatable.fields.index_field import IndexField  # noqa: F401
from collatable.fields.label_field import LabelField  # noqa: F401
from collatable.fields.list_field import ListField  # noqa: F401
from collatable.fields.metadata_field import MetadataField  # noqa: F401
from collatable.fields.scalar_field import ScalarField  # noqa: F401
from collatable.fields.sequence_field import SequenceField  # noqa: F401
from collatable.fields.sequence_label_field import SequenceLabelField  # noqa: F401
from collatable.fields.span_field import SpanField  # noqa: F401
from collatable.fields.tensor_field import TensorField  # noqa: F401
from collatable.fields.text_field import TextField as SingleTextField  # noqa: F401
from collatable.typing import Tensor

from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Token
from nlpstack.data.vocabulary import Vocabulary


class MappingField(Field[Dict[str, Any]]):
    def __init__(self, mapping: Mapping[str, Field]) -> None:
        super().__init__()
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __getitem__(self, key: str) -> Field:
        return self._mapping[key]

    def __str__(self) -> str:
        return str(self._mapping)

    def __repr__(self) -> str:
        return f"MappingField({self._mapping})"

    def as_array(self) -> Dict[str, Any]:
        return {key: field.as_array() for key, field in self._mapping.items()}

    def collate(  # type: ignore[override]
        self,
        arrays: Sequence,
    ) -> Dict[str, Any]:
        if not isinstance(arrays[0], MappingField):
            return super().collate(arrays)  # type: ignore[no-any-return]
        arrays = [x.as_array() for x in arrays]
        return {key: field.collate([x[key] for x in arrays]) for key, field in self._mapping.items()}


class TextField(SequenceField[Dict[str, Any]]):
    __slots__ = ["_tokens", "_text_fields"]

    def __init__(
        self,
        tokens: Sequence[Token],
        vocab: Vocabulary,
        indexers: Mapping[str, TokenIndexer],
    ) -> None:
        super().__init__()

        self._tokens = tokens
        self._text_fields = {
            key: SingleTextField(
                tokens,
                indexer=functools.partial(indexer, vocab=vocab),
                padding_value=indexer.get_pad_index(vocab),
            )
            for key, indexer in indexers.items()
        }

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(token) for token in self._tokens)}]"

    def __repr__(self) -> str:
        return f"TextField(tokens={self._tokens}, padding_value={self._padding_value})"

    @property
    def tokens(self) -> Sequence[Token]:
        return self._tokens

    def as_array(self) -> Dict[str, Any]:
        return {key: field.as_array() for key, field in self._text_fields.items()}

    def collate(  # type: ignore[override]
        self,
        arrays: Sequence,
    ) -> Dict[str, Any]:
        collate_fn = super().collate
        arrays = [x.as_array() for x in arrays]
        return {key: collate_fn([x[key] for x in arrays]) for key in arrays[0]}


class MultiLabelField(Field[Tensor]):
    __slots__ = ["_labels", "_label_indices", "_num_labels"]

    def __init__(
        self,
        labels: Union[Sequence[Sequence[int]], Sequence[Sequence[str]]],
        vocab: Mapping[str, int],
    ) -> None:
        super().__init__()
        self._labels = labels
        self._num_labels = len(vocab)

        self._label_indices: Sequence[int]
        if all(isinstance(label, int) for label in labels):
            self._label_indices = cast(Sequence[int], labels)
        elif all(isinstance(label, str) for label in labels):
            self._label_indices = [vocab[label] for label in cast(Sequence[str], labels)]
        else:
            raise ValueError("labels must be either all int or all str")

    def __str__(self) -> str:
        return f"[{', '.join(str(label) for label in self._labels)}]"

    def __repr__(self) -> str:
        return f"MultiLabelField(labels={self._labels})"

    @property
    def labels(self) -> Union[Sequence[Sequence[int]], Sequence[Sequence[str]]]:
        return self._labels

    def as_array(self) -> Tensor:
        return numpy.bincount(self._label_indices, minlength=self._num_labels)
