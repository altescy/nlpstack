import dataclasses
import json
import shutil
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Optional, Sequence, TypeVar

import colt
import minato

from nlpstack.common import load_jsonnet
from nlpstack.data import Dataset
from nlpstack.rune import Rune, RuneArchive

from .workflow import Workflow

logger = getLogger(__name__)

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")

coltbuilder = colt.ColtBuilder(typekey="type")


@dataclasses.dataclass
class RuneConfig(Generic[Example, Prediction]):
    model: Optional[Rune[Example, Prediction]] = None
    reader: Optional[Callable[[str], Iterator[Example]]] = None
    writer: Optional[Callable[[str, Iterable[Prediction]], None]] = None
    train_dataset_filename: Optional[str] = None
    valid_dataset_filename: Optional[str] = None
    test_dataset_filename: Optional[str] = None


@Workflow.register("rune")
class RuneWorkflow(Workflow):
    def train(
        self,
        config_filename: str,
        archive_filename: str,
    ) -> None:
        """train a model and save archive"""

        logger.info("Loading config from %s", config_filename)
        config = load_jsonnet(minato.cached_path(config_filename))
        rune_config = coltbuilder(config, RuneConfig)

        if rune_config.model is None:
            print("No model given.")
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

        model = rune_config.model
        model.train(train_examples, valid_examples)

        logger.info("Saving archive to %s", archive_filename)
        archive = RuneArchive(model, metadata={"config": config})
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)
            _archive_filename = tmpdir / "archive.tar.gz"
            archive.save(_archive_filename)
            with minato.open(archive_filename, "wb") as pklfile, _archive_filename.open("rb") as acvfile:
                shutil.copyfileobj(acvfile, pklfile)  # type: ignore[misc]

    def predict(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        input_filename: str,
        output_filename: str,
    ) -> None:
        """predict with a model and output results into a file"""

        config = load_jsonnet(minato.cached_path(config_filename))
        rune_config = coltbuilder(config, RuneConfig)

        if rune_config.reader is None:
            print("No reader given.")
            exit(1)
        if rune_config.writer is None:
            print("No writer given.")
            exit(1)

        archive = RuneArchive.load(minato.cached_path(archive_filename))  # type: ignore[var-annotated]
        model = archive.rune

        if not isinstance(model, Rune):
            print("Given model is not a Rune.")
            exit(1)

        predictions = model.predict(rune_config.reader(input_filename))
        rune_config.writer(output_filename, predictions)

    def evaluate(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        input_filename: str,
        output_filename: Optional[str] = None,
    ) -> None:
        """evaluate a model and output metrics"""

        config = load_jsonnet(minato.cached_path(config_filename))
        rune_config = coltbuilder(config, RuneConfig)

        if rune_config.reader is None:
            print("No reader given.")
            exit(1)

        archive = RuneArchive.load(minato.cached_path(archive_filename))  # type: ignore[var-annotated]
        model = archive.rune

        if not isinstance(model, Rune):
            print(f"Given model is not a Rune: {type(model)}")
            print(archive)
            exit(1)

        metrics = model.evaluate(rune_config.reader(input_filename))

        if output_filename is None:
            print(json.dumps(metrics, indent=2))
        else:
            with minato.open(output_filename, "w") as jsonfile:
                json.dump(metrics, jsonfile, ensure_ascii=False)
