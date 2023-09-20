import dataclasses
import fcntl
import json
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T")


class Metadata(TypedDict):
    pagesize: int


@dataclasses.dataclass(frozen=True)
class Key(Generic[K]):
    value: K

    def __str__(self) -> str:
        return str(self.value)

    def to_bytes(self) -> bytes:
        value = pickle.dumps(self.value)
        length = len(value).to_bytes(4, "little")
        return length + value

    @classmethod
    def from_binaryio(cls, f: BinaryIO) -> "Key":
        length = int.from_bytes(f.read(4), "little")
        return cls(pickle.loads(f.read(length)))


@dataclasses.dataclass(frozen=True)
class Index:
    page: int
    offset: int
    length: int

    def to_bytes(self) -> bytes:
        return self.page.to_bytes(4, "little") + self.offset.to_bytes(4, "little") + self.length.to_bytes(4, "little")

    @classmethod
    def from_binaryio(cls, f: BinaryIO) -> "Index":
        return cls(
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
        )


class FileBackendMapping(Mapping[K, V]):
    def __init__(
        self,
        path: Optional[Union[str, PathLike]] = None,
        pagesize: Optional[int] = None,
    ) -> None:
        self._delete_on_exit = path is None
        self._path = Path(path or tempfile.TemporaryDirectory().name)
        self._pagesize = pagesize
        self._indices: Dict[Key, Index] = {}
        self._pageios: Dict[int, BinaryIO] = {}

        self._restore()

    @property
    def path(self) -> Path:
        return self._path

    def _restore(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        index_filename = self._get_index_filename()
        if not index_filename.exists():
            index_filename.touch()

        metadata_filename = self._get_metadata_filename()
        if metadata_filename.exists():
            self._load_metadata()
        else:
            self._save_metadata()

        self._indexio: BinaryIO = index_filename.open("rb+")
        if self._indexio.seek(0, 2) > 0:
            self._load_indices()

        for page, page_filename in self._iter_page_filenames():
            self._pageios[page] = page_filename.open("rb+")

    @contextmanager
    def lock(self) -> Iterator[None]:
        lockfile = self._get_lock_filename().open("w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
            lockfile.close()

    def _get_lock_filename(self) -> Path:
        return self._path / "storage.lock"

    def _get_metadata_filename(self) -> Path:
        return self._path / "metadata.json"

    def _get_index_filename(self) -> Path:
        return self._path / "index.bin"

    def _get_page_filename(self, page: int) -> Path:
        return self._path / f"page_{page:08d}"

    def _iter_page_filenames(self) -> Iterator[Tuple[int, Path]]:
        for page_filename in self._path.glob("page_*"):
            page = int(page_filename.stem.split("_", 1)[1])
            yield page, page_filename

    def _load_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        if not metadata_filename.exists():
            raise FileNotFoundError(metadata_filename)
        with metadata_filename.open("r") as f:
            metadata = json.load(f)
        self._pagesize = metadata["pagesize"]

    def _save_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        with metadata_filename.open("w") as f:
            json.dump({"pagesize": self._pagesize}, f)

    def _load_indices(self) -> None:
        eof = self._indexio.seek(0, 2)
        self._indexio.seek(0)
        while self._indexio.seek(0, 1) < eof:
            key = Key.from_binaryio(self._indexio)
            index = Index.from_binaryio(self._indexio)
            self._indices[key] = index

    def _add_index(self, key: Key, index: Index) -> None:
        self._indices[key] = index
        key_value = key.to_bytes()
        index_value = index.to_bytes()
        self._indexio.seek(0, 2)
        self._indexio.write(key_value)
        self._indexio.write(index_value)

    def _encode(self, value: V) -> bytes:
        buffer = BytesIO()
        pickle.dump(value, buffer)
        return buffer.getvalue()

    def _decode(self, value_bytes: bytes) -> V:
        return cast(V, pickle.loads(value_bytes))

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            key = Key(key)
        if not isinstance(key, Key):
            return False
        return key in self._indices

    def __getitem__(self, key: Union[K, Key[K]]) -> V:
        if not isinstance(key, Key):
            key = Key(key)
        if key not in self._indices:
            raise KeyError(key)
        index = self._indices[key]
        pageio = self._pageios[index.page]
        pageio.seek(index.offset)
        return cast(V, pickle.loads(pageio.read(index.length)))

    def __setitem__(self, key: Union[K, Key[K]], value: V) -> None:
        if not isinstance(key, Key):
            key = Key(key)

        if key in self._indices:
            raise KeyError(f"Key {key.value} already exists")

        encoded_value = self._encode(value)
        length = len(encoded_value)

        pageio: BinaryIO
        if not self._pageios:
            page = 0
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio
        else:
            page = len(self._pageios) - 1
            pageio = self._pageios[page]

        offset = pageio.seek(0, 2)
        if self._pagesize is not None and offset + length > self._pagesize:
            page += 1
            offset = 0
            pageio = open(self._get_page_filename(page), "wb+")
            self._pageios[page] = pageio

        pageio.write(encoded_value)
        self._add_index(key, Index(page, offset, length))

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[K]:
        for key in self._indices:
            yield key.value

    def flush(self) -> None:
        self._indexio.flush()
        for pageio in self._pageios.values():
            pageio.flush()

    def close(self) -> None:
        self._indexio.close()
        for pageio in self._pageios.values():
            pageio.close()

    def __del__(self) -> None:
        self.flush()
        self.close()
        if self._delete_on_exit and self._path.exists():
            shutil.rmtree(self._path)

    def __getstate__(self) -> Dict[str, Any]:
        return {"path": self._path}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._path = state["path"]
        self._pagesize = None
        self._indices = {}
        self._pageios = {}
        self._restore()


class FileBackendSequence(Sequence[T]):
    def __init__(
        self,
        path: Optional[Union[str, PathLike]] = None,
        pagesize: Optional[int] = None,
    ) -> None:
        self._delete_on_exit = path is None
        self._path = Path(path or tempfile.TemporaryDirectory().name)
        self._pagesize = pagesize
        self._indices: List[Index] = []
        self._pageios: Dict[int, BinaryIO] = {}

        self._path.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        index_filename = self._get_index_filename()
        if not index_filename.exists():
            index_filename.touch()

        metadata_filename = self._get_metadata_filename()
        if metadata_filename.exists():
            self._load_metadata()
        else:
            self._save_metadata()

        self._indexio: BinaryIO = index_filename.open("rb+")
        if self._indexio.seek(0, 2) > 0:
            self._load_indices()

        for page, page_filename in self._iter_page_filenames():
            self._pageios[page] = page_filename.open("rb+")

    def __del__(self) -> None:
        self.flush()
        self.close()
        if self._delete_on_exit:
            shutil.rmtree(self._path)

    @staticmethod
    def _encode(obj: T) -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def _decode(data: bytes) -> T:
        return cast(T, pickle.loads(data))

    @property
    def path(self) -> Path:
        return self._path

    def _get_index_filename(self) -> Path:
        return self._path / "index.bin"

    def _get_metadata_filename(self) -> Path:
        return self._path / "metadata.json"

    def _get_lock_filename(self) -> Path:
        return self._path / "lock"

    def _get_page_filename(self, page: int) -> Path:
        return self._path / f"page_{page:08d}"

    def _iter_page_filenames(self) -> Iterable[Tuple[int, Path]]:
        for page_filename in self._path.glob("page_*"):
            page = int(page_filename.stem.split("_", 1)[1])
            yield page, page_filename

    def _add_index(self, index: Index) -> None:
        self._indices.append(index)
        self._indexio.seek(0, 2)
        self._indexio.write(index.to_bytes())

    def _load_indices(self) -> None:
        if self._indices:
            raise RuntimeError("indices already loaded")
        eof = self._indexio.seek(0, 2)
        self._indexio.seek(0)
        while self._indexio.tell() < eof:
            self._indices.append(Index.from_binaryio(self._indexio))

    def _load_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        with metadata_filename.open("r") as f:
            metadata = json.load(f)
        self._pagesize = metadata["pagesize"]

    def _save_metadata(self) -> None:
        metadata_filename = self._get_metadata_filename()
        with metadata_filename.open("w") as f:
            json.dump({"pagesize": self._pagesize}, f)

    def append(self, obj: T) -> None:
        binary = self._encode(obj)

        pageio: BinaryIO
        if not self._pageios:
            page = 0
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio
        else:
            page = len(self._pageios) - 1
            pageio = self._pageios[page]

        offset = pageio.seek(0, 2)
        if self._pagesize is not None and offset + len(binary) > self._pagesize:
            page += 1
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio

        pageio.write(binary)
        self._add_index(Index(page, offset, len(binary)))

    def flush(self) -> None:
        for pageio in self._pageios.values():
            pageio.flush()
        self._indexio.flush()

    def close(self) -> None:
        for pageio in self._pageios.values():
            pageio.close()
        self._indexio.close()

    @contextmanager
    def lock(self) -> Iterator[None]:
        import fcntl

        lockfile = self._get_lock_filename().open("w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
            lockfile.close()

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> "List[T]":
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, List[T]]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            index = self._indices[key]
            pageio = self._pageios[index.page]
            pageio.seek(index.offset)
            return self._decode(pageio.read(index.length))
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

    def __getstate__(self) -> Dict[str, Any]:
        if self._delete_on_exit:
            raise RuntimeError("cannot pickle a temporary database")
        return {"path": self._path}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._path = state["path"]
        self._delete_on_exit = False
        self._pagesize = 1024 * 1024 * 1024
        self._indices = []
        self._pageios = {}
        self._restore()

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[T],
        path: Optional[Union[str, PathLike]] = None,
        pagesize: int = 1024 * 1024 * 1024,
    ) -> "FileBackendSequence[T]":
        dataset = cls(path, pagesize)
        for obj in iterable:
            dataset.append(obj)
        dataset.flush()
        return dataset
