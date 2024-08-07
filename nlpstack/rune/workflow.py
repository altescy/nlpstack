import functools
import json
import shutil
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Sequence

import minato

from nlpstack.common import FileBackendSequence
from nlpstack.workflow import Workflow

from .archive import RuneArchive
from .base import Rune
from .config import RuneConfig

logger = getLogger(__name__)


@Workflow.register("rune")
class RuneWorkflow(Workflow):
    def train(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        overrides: Optional[str] = None,
    ) -> None:
        """train a model and save archive"""

        logger.info("Loading config from %s", config_filename)
        rune_config = RuneConfig[Any, Any].from_jsonnet(minato.cached_path(config_filename), overrides=overrides)

        if rune_config.model is None:
            print("No model given.")
            exit(1)
        if rune_config.reader is None:
            print("No reader given.")
            exit(1)
        if rune_config.train_dataset_filename is None:
            print("No train dataset filename given.")
            exit(1)

        train_examples: Sequence = FileBackendSequence.from_iterable(
            rune_config.reader(rune_config.train_dataset_filename)
        )
        valid_examples: Optional[Sequence] = None
        if rune_config.valid_dataset_filename is not None:
            valid_examples = FileBackendSequence.from_iterable(rune_config.reader(rune_config.valid_dataset_filename))

        model = rune_config.model

        logger.info("Setting up model...")
        model.setup("training", rune_config.get_setup_params(model.SetupParams))

        logger.info("Start training...")
        model.train(train_examples, valid_examples)

        logger.info("Saving archive to %s", archive_filename)
        archive = RuneArchive(model, metadata={"config": rune_config.to_json()})
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
        overrides: Optional[str] = None,
    ) -> None:
        """predict with a model and output results into a file"""

        rune_config = RuneConfig[Any, Any].from_jsonnet(minato.cached_path(config_filename), overrides=overrides)

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

        logger.info("Setting up model...")
        model.setup("prediction", rune_config.get_setup_params(model.SetupParams))

        params = rune_config.get_prediction_params(model.PredictionParams)

        predictions = model.predict(rune_config.reader(input_filename), params)
        rune_config.writer(output_filename, predictions)

    def evaluate(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        input_filename: str,
        output_filename: Optional[str] = None,
        overrides: Optional[str] = None,
    ) -> None:
        """evaluate a model and output metrics"""

        rune_config = RuneConfig[Any, Any].from_jsonnet(minato.cached_path(config_filename), overrides=overrides)

        if rune_config.reader is None:
            print("No reader given.")
            exit(1)

        archive = RuneArchive.load(minato.cached_path(archive_filename))  # type: ignore[var-annotated]
        model = archive.rune

        if not isinstance(model, Rune):
            print(f"Given model is not a Rune: {type(model)}")
            print(archive)
            exit(1)

        logger.info("Setting up model...")
        model.setup("evaluation", rune_config.get_setup_params(model.EvaluationParams))

        params = rune_config.get_evaluation_params(model.EvaluationParams)

        metrics = model.evaluate(rune_config.reader(input_filename), params)

        if output_filename is None:
            print(json.dumps(metrics, indent=2))
        else:
            with minato.open(output_filename, "w") as jsonfile:
                json.dump(metrics, jsonfile, ensure_ascii=False)

    def serve(
        self,
        archive_filename: str,
        *,
        host: str = "localhost",
        port: int = 8080,
        config_filename: Optional[str] = None,
        overrides: Optional[str] = None,
    ) -> None:
        from http.server import HTTPServer

        from nlpstack.server.handler import RuneHandler

        archive = RuneArchive.load(minato.cached_path(archive_filename))  # type: ignore[var-annotated]
        model = archive.rune

        if not isinstance(model, Rune):
            print("Given file is not a rune archive")
            exit(1)

        rune_config: Optional[RuneConfig] = None
        if config_filename is not None:
            rune_config = RuneConfig.from_jsonnet(minato.cached_path(config_filename), overrides=overrides)

            logger.info("Setting up model...")
            model.setup("prediction", rune_config.get_setup_params(model.SetupParams))

        server = HTTPServer(
            (host, port),
            functools.partial(RuneHandler, rune=model),
        )
        logger.info("Listening on %s:%d", host, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            server.shutdown()
            logger.info("Done")
