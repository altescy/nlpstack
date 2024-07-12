import functools
import json
import os
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import minato

from nlpstack.common import FileBackendSequence, load_jsonnet
from nlpstack.integrations.mlflow.util import flatten_dict_for_mlflow_log
from nlpstack.rune import Rune, RuneArchive, RuneConfig
from nlpstack.workflow import Workflow

logger = getLogger(__name__)


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

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(flatten_dict_for_mlflow_log(rune_config.to_json()))

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
            archive = RuneArchive(model, metadata={"config": rune_config.to_json()})
            with tempfile.TemporaryDirectory() as _tmpdir:
                tmpdir = Path(_tmpdir)

                Path(tmpdir / "config.json").write_text(json.dumps(rune_config.to_json(), indent=2, ensure_ascii=False))
                archive.save(tmpdir / "archive.tar.gz")

                mlflow.log_artifacts(tmpdir)

            if valid_examples is not None:
                logger.info("Start evaluation with valid dataset...")
                metrics = model.evaluate(valid_examples)
                metrics = {f"valid_{key}": value for key, value in metrics.items()}
                mlflow.log_metrics(metrics)

            if rune_config.test_dataset_filename is not None:
                logger.info("Start evaluation with test dataset...")
                test_examples = FileBackendSequence.from_iterable(rune_config.reader(rune_config.test_dataset_filename))
                metrics = model.evaluate(test_examples)
                metrics = {f"test_{key}": value for key, value in metrics.items()}
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

        from nlpstack.integrations.optuna import OptunaConfig

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

            rune_config = RuneConfig[Any, Any].from_jsonnet(
                minato.cached_path(config_filename), ext_vars=ext_vars, overrides=overrides
            )

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
                mlflow.log_params(flatten_dict_for_mlflow_log(rune_config.to_json()))

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

                if valid_examples is not None:
                    logger.info("Start evaluation with valid dataset...")
                    metrics = model.evaluate(valid_examples)
                    metrics = {f"valid_{key}": value for key, value in metrics.items()}
                    mlflow.log_metrics(metrics)

                logger.info("Saving artifacts...")
                archive = RuneArchive(model, metadata={"config": rune_config.to_json()})
                with tempfile.TemporaryDirectory() as _tmpdir:
                    tmpdir = Path(_tmpdir)

                    Path(tmpdir / "config.json").write_text(
                        json.dumps(rune_config.to_json(), indent=2, ensure_ascii=False)
                    )
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

        optuna_config = OptunaConfig.from_jsonnet(minato.cached_path(optuna_config_filename))

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
                    direction=optuna_config.direction,
                    sampler=optuna_config.sampler,
                    pruner=optuna_config.pruner,
                    storage=f"sqlite:///{optuna_storage_path}",
                    load_if_exists=resume,
                )

                mlflow.log_params(flatten_dict_for_mlflow_log(optuna_config.to_json()))

                mlflow.log_artifact(config_filename)
                mlflow.log_artifact(params_filename)

                objective = functools.partial(
                    _objective,
                    metric=optuna_config.metric,
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

        model.setup("evaluation", **(rune_config.evaluator or {}))

        with mlflow.start_run(run_id=run_id):
            logger.info("Start evaluation...")
            metrics = model.evaluate(rune_config.reader(input_filename))
            if metric_prefix is not None:
                metrics = {f"{metric_prefix}{key}": value for key, value in metrics.items()}

            logger.info("Metrics: %s", json.dumps(metrics))
            mlflow.log_metrics(metrics)
