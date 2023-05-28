from pathlib import Path

from nlpstack.workflows import RuneWorkflow


def test_run_workflow_train_and_predict(tmp_path: Path) -> None:
    archive_filename = tmp_path / "archive.pkl"
    RuneWorkflow.run(
        [
            "train",
            "tests/fixtures/configs/rune_workflow.jsonnet",
            f"{archive_filename}",
        ]
    )
    assert archive_filename.is_file()


def test_run_workflow_predict(tmp_path: Path) -> None:
    output_filename = tmp_path / "output.jsonl"
    RuneWorkflow.run(
        [
            "predict",
            "tests/fixtures/configs/rune_workflow.jsonnet",
            "tests/fixtures/archives/classifier.pkl",
            "--input-filename",
            "./tests/fixtures/data/classification.jsonl",
            "--output-filename",
            f"{output_filename}",
        ]
    )
    assert output_filename.is_file()


def test_run_workflow_evaluate(tmp_path: Path) -> None:
    metrics_filename = tmp_path / "metrics.json"
    RuneWorkflow.run(
        [
            "evaluate",
            "tests/fixtures/configs/rune_workflow.jsonnet",
            "tests/fixtures/archives/classifier.pkl",
            "--input-filename",
            "tests/fixtures/data/classification.jsonl",
            "--output-filename",
            f"{metrics_filename}",
        ]
    )
    assert metrics_filename.is_file()
