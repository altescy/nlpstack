import dataclasses
import importlib
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple


def get_installed_packages() -> List[Tuple[str, str]]:
    distributions = importlib.metadata.distributions()
    return sorted([(d.metadata["Name"], d.version) for d in distributions])


@dataclasses.dataclass(frozen=True)
class PlatformInfo:
    python: str = dataclasses.field(default_factory=lambda: platform.python_version())
    packages: Optional[Sequence[Tuple[str, str]]] = dataclasses.field(default_factory=get_installed_packages)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, json: Dict[str, Any]) -> "PlatformInfo":
        return cls(
            python=json["python"],
            packages=[(n, v) for n, v in json["packages"]],
        )
