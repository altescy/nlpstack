import dataclasses
import datetime
import json
import pickle
import tarfile
import tempfile
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import ClassVar, Generic, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from .base import Rune

logger = getLogger(__name__)

Self = TypeVar("Self", bound="RuneArchive")
RuneType = TypeVar("RuneType", bound=Rune)


def get_pip_packages() -> Optional[List[Tuple[str, str]]]:
    try:
        import pkg_resources

        return sorted([(d.key, d.version) for d in iter(pkg_resources.working_set)])
    except Exception:
        logger.error("Error saving pip packages")
    return None


@dataclasses.dataclass(frozen=True)
class RuneArchive(Generic[RuneType]):
    rune: RuneType
    metadata: Optional[Mapping[str, str]] = None
    packages: Optional[Sequence[Tuple[str, str]]] = dataclasses.field(default_factory=get_pip_packages)
    archived_at: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.utcnow)

    _RUNE_FILENAME: ClassVar[str] = "rune.pkl"
    _METADATA_FILENAME: ClassVar[str] = "metadata.json"

    def save(self, filename: Union[str, PathLike]) -> None:
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)

            rune_filename = tmpdir / self._RUNE_FILENAME
            with rune_filename.open("wb") as f:
                pickle.dump(self.rune, f)

            metadata_filename = tmpdir / self._METADATA_FILENAME
            with metadata_filename.open("w") as f:
                json.dump(
                    {
                        "metadata": self.metadata,
                        "packages": self.packages,
                        "archived_at": self.archived_at.isoformat(),
                    },
                    f,
                )

            with tarfile.open(filename, "w:gz") as tar:
                tar.add(rune_filename, arcname=self._RUNE_FILENAME)
                tar.add(metadata_filename, arcname=self._METADATA_FILENAME)

    @classmethod
    def load(cls: Type[Self], filename: Union[str, PathLike]) -> Self:
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)

            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(tmpdir)

            rune_filename = tmpdir / cls._RUNE_FILENAME
            with rune_filename.open("rb") as f:
                rune = pickle.load(f)

            metadata_filename = tmpdir / cls._METADATA_FILENAME
            with metadata_filename.open("r") as f:
                metadata = json.load(f)

        return cls(
            rune=rune,
            metadata=metadata["metadata"],
            packages=metadata["packages"],
            archived_at=datetime.datetime.fromisoformat(metadata["archived_at"]),
        )
