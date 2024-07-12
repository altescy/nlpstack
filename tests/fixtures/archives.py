from pathlib import Path

import pytest

from nlpstack.rune import RuneWorkflow


@pytest.fixture(scope="session")
def classifier_archive_filename(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("archives")
    filename = tmp_path / "archive.tar.gz"
    RuneWorkflow.run(
        [
            "train",
            "tests/fixtures/configs/rune_workflow.jsonnet",
            f"{filename}",
        ]
    )
    return filename


@pytest.fixture()
def archive_filename(classifier_archive_filename: Path) -> Path:
    return classifier_archive_filename
