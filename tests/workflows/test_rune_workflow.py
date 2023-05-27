from pathlib import Path

from nlpstack.workflows import RuneWorkflow


def test_run_workflow_train_and_predict(tmp_path: Path) -> None:
    archive_filename = tmp_path / "archive.pkl"
    RuneWorkflow.run(
        [
            "train",
            "tests/fixtures/configs/run_workflow.jsonnet",
            f"{archive_filename}",
        ]
    )
    assert archive_filename.is_file()

    output_filename = tmp_path / "output.jsonl"
    RuneWorkflow.run(
        [
            "predict",
            "tests/fixtures/configs/run_workflow.jsonnet",
            f"{archive_filename}",
            f"{output_filename}",
            "--dataset",
            "./tests/fixtures/data/classification.jsonl",
        ]
    )
    assert output_filename.is_file()
