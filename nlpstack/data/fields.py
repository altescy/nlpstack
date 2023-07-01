import functools
from typing import Any, Dict, Iterator, Mapping, Sequence

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
from collatable.fields.text_field import TextField as SingleTextField
from collatable.typing import DataArray

from nlpstack.data.indexers import TokenIndexer
from nlpstack.data.tokenizers import Token
from nlpstack.data.vocabulary import Vocabulary


class MappingField(Field):
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
    ) -> DataArray:
        if not isinstance(arrays[0], MappingField):
            return super().collate(arrays)  # type: ignore[no-any-return]
        arrays = [x.as_array() for x in arrays]
        return {key: field.collate([x[key] for x in arrays]) for key, field in self._mapping.items()}


class TextField(MappingField):
    def __init__(
        self,
        text: Sequence[Token],
        vocab: Vocabulary,
        indexers: Mapping[str, TokenIndexer],
    ) -> None:
        super().__init__(
            {
                key: SingleTextField(
                    text,
                    indexer=functools.partial(indexer, vocab=vocab),
                    padding_value=indexer.get_pad_index(vocab),
                )
                for key, indexer in indexers.items()
            }
        )
