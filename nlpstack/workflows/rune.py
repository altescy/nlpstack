import dataclasses
import json
import pickle
from logging import getLogger
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar

import colt
import minato
from rjsonnet import evaluate_file

from nlpstack.data import Dataset
from nlpstack.rune import Rune

from .workflow import Workflow

logger = getLogger(__name__)

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")

coltbuilder = colt.ColtBuilder(typekey="type")


@dataclasses.dataclass
class RuneConfig(Generic[Example, Prediction]):
    rune: Rune[Example, Prediction]
    reader: Optional[Callable[[str], Iterator[Example]]] = None
    writer: Optional[Callable[[str, Iterable[Prediction]], None]] = None

    @classmethod
    def from_file(cls, filename: str) -> "RuneConfig":
        config = json.loads(evaluate_file(filename))
        return coltbuilder(config, RuneConfig)


class RuneWorkflow(Workflow):
    def train(
        self,
        config_filename: str,
        archive_filename: str,
    ) -> None:
        logger.info("Loading config from %s", config_filename)
        rune_config = RuneConfig.from_file(config_filename)

        if rune_config.reader is None:
            print("No reader given.")
            exit(1)

        train_examples = Dataset.from_iterable(rune_config.reader("train"))
        valid_examples = Dataset.from_iterable(rune_config.reader("valid")) or None

        rune = rune_config.rune
        rune.train(train_examples, valid_examples)

        logger.info("Saving archive to %s", archive_filename)
        with minato.open(archive_filename, "wb") as pklfile:
            pickle.dump(rune, pklfile)

    def predict(
        self,
        config_filename: str,
        archive_filename: str,
        output_filename: str,
        *,
        dataset: str = "valid",
    ) -> None:
        rune_config = RuneConfig.from_file(config_filename)

        if rune_config.reader is None:
            print("No reader given.")
            exit(1)
        if rune_config.writer is None:
            print("No writer given.")
            exit(1)

        with minato.open(archive_filename, "rb") as pklfile:
            rune = pickle.load(pklfile)

        if not isinstance(rune, Rune):
            print("Given archive is not a Rune.")
            exit(1)

        predictions = rune.predict(rune_config.reader(dataset))
        rune_config.writer(output_filename, predictions)
