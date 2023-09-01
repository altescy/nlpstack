import dataclasses
import functools
import json
import os
import shutil
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, Literal, Mapping, Optional, Sequence, TypeVar

import colt
import minato

from nlpstack.common import FileBackendSequence, load_jsonnet
from nlpstack.mlflow.util import flatten_dict_for_mlflow_log
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
    predictor: Optional[Mapping[str, Any]] = None
    evaluator: Optional[Mapping[str, Any]] = None
    train_dataset_filename: Optional[str] = None
    valid_dataset_filename: Optional[str] = None
    test_dataset_filename: Optional[str] = None


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
        config = load_jsonnet(minato.cached_path(config_filename), overrides=overrides)
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

        train_examples: Sequence = FileBackendSequence.from_iterable(
            rune_config.reader(rune_config.train_dataset_filename)
        )
        valid_examples: Optional[Sequence] = None
        if rune_config.valid_dataset_filename is not None:
            valid_examples = FileBackendSequence.from_iterable(rune_config.reader(rune_config.valid_dataset_filename))

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
        overrides: Optional[str] = None,
    ) -> None:
        """predict with a model and output results into a file"""

        config = load_jsonnet(minato.cached_path(config_filename), overrides=overrides)
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

        model.setup("prediction", **coltbuilder(rune_config.predictor or {}))

        predictions = model.predict(rune_config.reader(input_filename))
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

        config = load_jsonnet(minato.cached_path(config_filename), overrides=overrides)
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

        model.setup("evaluation", **coltbuilder(rune_config.evaluator or {}))

        metrics = model.evaluate(rune_config.reader(input_filename))

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
            config = load_jsonnet(minato.cached_path(config_filename), overrides=overrides)
            rune_config = coltbuilder(config, RuneConfig)

        predicotr_config: Optional[Mapping[str, Any]] = None
        if rune_config is not None:
            predicotr_config = rune_config.predictor

        model.setup("prediction", **coltbuilder(predicotr_config or {}))

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


@Workflow.register("rune-mlflow")
class RuneMlflowWorkflow(Workflow):
    def train(
        self,
        config_filename: str,
        *,
        run_name: Optional[str] = None,
        overrides: Optional[str] = None,
    ) -> None:
        """train a model and save archive with mlflow"""

        import mlflow

        logger.info("Loading config from %s", config_filename)
        config = load_jsonnet(minato.cached_path(config_filename), overrides=overrides)
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

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(flatten_dict_for_mlflow_log(config))

            train_examples: Sequence = FileBackendSequence.from_iterable(
                rune_config.reader(rune_config.train_dataset_filename)
            )
            valid_examples: Optional[Sequence] = None
            if rune_config.valid_dataset_filename is not None:
                valid_examples = FileBackendSequence.from_iterable(
                    rune_config.reader(rune_config.valid_dataset_filename)
                )

            logger.info("Start training...")
            model = rune_config.model
            model.train(train_examples, valid_examples)

            logger.info("Saving artifacts...")
            archive = RuneArchive(model, metadata={"config": config})
            with tempfile.TemporaryDirectory() as _tmpdir:
                tmpdir = Path(_tmpdir)

                Path(tmpdir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
                archive.save(tmpdir / "archive.tar.gz")

                mlflow.log_artifacts(tmpdir)

            if valid_examples is not None:
                logger.info("Start evaluation with valid dataset...")
                metrics = model.evaluate(valid_examples)
                mlflow.log_metrics(metrics)

            if rune_config.test_dataset_filename is not None:
                logger.info("Start evaluation with test dataset...")
                test_examples = FileBackendSequence.from_iterable(rune_config.reader(rune_config.test_dataset_filename))
                metrics = model.evaluate(test_examples)
                mlflow.log_metrics(metrics)

            logger.info("Done")

    def tune(
        self,
        config_filename: str,
        params_filename: str,
        *,
        study_name: Optional[str] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        resume: bool = False,
        optuna_config_filename: str = "optuna.jsonnet",
        overrides: Optional[str] = None,
    ) -> None:
        """tune hyper-parameters with optuna and save archive with mlflow"""

        import mlflow
        import optuna
        from optuna.pruners import BasePruner
        from optuna.samplers import BaseSampler

        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default")

        def get_previous_mlflow_run() -> Optional[mlflow.entities.Run]:
            mlflow_experiment = mlflow_client.get_experiment_by_name(mlflow_experiment_name)
            if mlflow_experiment is None:
                return None

            mlflow_runs = mlflow_client.search_runs(
                experiment_ids=[mlflow_experiment.experiment_id],
                filter_string=f"attributes.run_name = '{study_name}'",
            )
            if not mlflow_runs:
                return None
            if len(mlflow_runs) > 1:
                print("Found multiple runs with the same name.")
                exit(1)
            return mlflow_runs[0]

        @dataclasses.dataclass
        class OptunaSettings:
            metric: str
            direction: Literal["minimize", "maximize"] = "minimize"
            sampler: Optional[BaseSampler] = None
            pruner: Optional[BasePruner] = None

        def _objective(
            trial: optuna.trial.Trial,
            metric: str,
            experiment: str,
            hparam_path: str,
            parent_run: mlflow.entities.Run,
        ) -> float:
            ext_vars: Dict[str, Any] = {}
            hparams = load_jsonnet(hparam_path)
            for key, params in hparams.items():
                value_type = params.pop("type")
                suggest = getattr(trial, f"suggest_{value_type}")
                value = suggest(key, **params)
                ext_vars[key] = str(value)

            config = load_jsonnet(minato.cached_path(config_filename), ext_vars=ext_vars, overrides=overrides)
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

            parent_run_name = parent_run.info.run_name
            run_name = f"{parent_run_name}_trial_{trial.number}"

            with mlflow.start_run(run_name=run_name, nested=True) as mlflow_run:
                mlflow.log_params(flatten_dict_for_mlflow_log(config))

                train_examples: Sequence = FileBackendSequence.from_iterable(
                    rune_config.reader(rune_config.train_dataset_filename)
                )
                valid_examples: Optional[Sequence] = None
                if rune_config.valid_dataset_filename is not None:
                    valid_examples = FileBackendSequence.from_iterable(
                        rune_config.reader(rune_config.valid_dataset_filename)
                    )

                logger.info("Start training...")
                model = rune_config.model
                model.train(train_examples, valid_examples)

                logger.info("Saving artifacts...")
                archive = RuneArchive(model, metadata={"config": config})
                with tempfile.TemporaryDirectory() as _tmpdir:
                    tmpdir = Path(_tmpdir)

                    Path(tmpdir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
                    archive.save(tmpdir / "archive.tar.gz")

                    mlflow.log_artifacts(tmpdir)

                metric_history = mlflow_client.get_metric_history(mlflow_run.info.run_id, metric)
                if not metric_history:
                    raise RuntimeError(f"No metric history found for {metric}.")

                latest_metric = metric_history[-1]

                mlflow_client.log_metric(
                    parent_run.info.run_id,
                    metric,
                    latest_metric.value,
                    step=trial.number,
                )

                return float(latest_metric.value)

        optuna_config = load_jsonnet(minato.cached_path(optuna_config_filename))
        optuna_settings = coltbuilder(optuna_config, OptunaSettings)

        previous_run = get_previous_mlflow_run()

        with tempfile.TemporaryDirectory() as _workdir:
            workdir = Path(_workdir)

            optuna_storage_path = workdir / "optuna.db"

            if resume and previous_run is not None:
                logger.info(f"Resuming run {study_name}.")
                mlflow.artifacts.download_artifacts(
                    run_id=previous_run.info.run_id,
                    artifact_path="optuna.db",
                    dst_path=workdir,
                )
                assert optuna_storage_path.exists()
            elif not resume and previous_run is not None:
                print(f"Run {study_name} already exists. If you want to resume, please use --resume option.")
                exit(1)
            elif resume and previous_run is None:
                print(f"Resume requested but run {study_name} does not exist.")
                exit(1)

            with mlflow.start_run(
                run_name=study_name,
                run_id=previous_run.info.run_id if (resume and previous_run is not None) else None,
            ) as mlflow_run:
                study = optuna.create_study(
                    study_name=mlflow_run.info.run_name,
                    direction=optuna_settings.direction,
                    sampler=optuna_settings.sampler,
                    pruner=optuna_settings.pruner,
                    storage=f"sqlite:///{optuna_storage_path}",
                    load_if_exists=resume,
                )

                mlflow.log_params(flatten_dict_for_mlflow_log(optuna_config))

                mlflow.log_artifact(config_filename)
                mlflow.log_artifact(params_filename)

                objective = functools.partial(
                    _objective,
                    metric=optuna_settings.metric,
                    experiment=mlflow.active_run().info.experiment_id,
                    hparam_path=params_filename,
                    parent_run=mlflow_run,
                )

                try:
                    study.optimize(
                        objective,
                        n_trials=n_trials,
                        timeout=timeout,
                    )
                    best_result = {
                        "trial": study.best_trial.number,
                        "params": study.best_trial.params,
                        "metric": study.best_trial.value,
                    }
                    best_result_filename = workdir / "best.json"
                    best_result_filename.write_text(json.dumps(best_result, indent=2, ensure_ascii=False))
                    mlflow.log_artifact(best_result_filename)
                finally:
                    logger.info("Saving artifacts...")
                    mlflow.log_artifact(optuna_storage_path)

            logger.info("Done")

    def evaluate(
        self,
        config_filename: str,
        archive_filename: str,
        *,
        input_filename: str,
        metric_prefix: Optional[str] = None,
        run_id: Optional[str] = None,
        overrides: Optional[str] = None,
    ) -> None:
        """evaluate a model and output metrics"""
        import mlflow

        config = load_jsonnet(minato.cached_path(config_filename), overrides=overrides)
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

        model.setup("evaluation", **coltbuilder(rune_config.evaluator or {}))

        with mlflow.start_run(run_id=run_id):
            logger.info("Start evaluation...")
            metrics = model.evaluate(rune_config.reader(input_filename))
            if metric_prefix is not None:
                metrics = {f"{metric_prefix}{key}": value for key, value in metrics.items()}

            logger.info("Metrics: %s", json.dumps(metrics))
            mlflow.log_metrics(metrics)
