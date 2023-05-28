import dataclasses
import json
import pickle
from logging import getLogger
from typing import Callable, Generic, Iterable, Iterator, Optional, Sequence, TypeVar

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
    rune: Optional[Rune[Example, Prediction]] = None
    reader: Optional[Callable[[str], Iterator[Example]]] = None
    writer: Optional[Callable[[str, Iterable[Prediction]], None]] = None
    train_dataset_filename: Optional[str] = None
    valid_dataset_filename: Optional[str] = None
    test_dataset_filename: Optional[str] = None

    @classmethod
    def from_file(cls, filename: str) -> "RuneConfig":
        config = json.loads(evaluate_file(filename))
        return coltbuilder(config, RuneConfig)


@Workflow.register("rune")
class RuneWorkflow(Workflow):
    def train(
        self,
        config_filename: str,
        archive_filename: str,
    ) -> None:
        logger.info("Loading config from %s", config_filename)
        rune_config = RuneConfig.from_file(config_filename)

        if rune_config.rune is None:
            print("No rune given.")
            exit(1)
        if rune_config.reader is None:
            print("No reader given.")
            exit(1)
        if rune_config.train_dataset_filename is None:
            print("No train dataset filename given.")
            exit(1)

        train_examples: Sequence = Dataset.from_iterable(rune_config.reader(rune_config.train_dataset_filename))
        valid_examples: Optional[Sequence] = None
        if rune_config.valid_dataset_filename is not None:
            valid_examples = Dataset.from_iterable(rune_config.reader(rune_config.valid_dataset_filename))

        rune = rune_config.rune
        rune.train(train_examples, valid_examples)

        logger.info("Saving archive to %s", archive_filename)
        with minato.open(archive_filename, "wb") as pklfile:
            pickle.dump(rune, pklfile)

    def predict(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        input_filename: str,
        output_filename: str,
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

        predictions = rune.predict(rune_config.reader(input_filename))
        rune_config.writer(output_filename, predictions)

    def evaluate(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        input_filename: str,
        output_filename: Optional[str] = None,
    ) -> None:
        rune_config = RuneConfig.from_file(config_filename)

        if rune_config.reader is None:
            print("No reader given.")
            exit(1)

        with minato.open(archive_filename, "rb") as pklfile:
            rune = pickle.load(pklfile)

        if not isinstance(rune, Rune):
            print("Given archive is not a Rune.")
            exit(1)

        metrics = rune.evaluate(rune_config.reader(input_filename))

        if output_filename is None:
            print(json.dumps(metrics, indent=2))
        else:
            with minato.open(output_filename, "w") as jsonfile:
                json.dump(metrics, jsonfile, ensure_ascii=False)
