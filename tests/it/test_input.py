import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import mlflow
import mlflow.tracking

from modelplane.runways.data import (
    _MLFLOW_REQUIRED_ERROR_MESSAGE,
    LocalInput,
    DVCInput,
    MLFlowArtifactInput,
    build_and_log_input,
)

LOCAL_FILE_PATH = "tests/data/prompts.csv"
LOCAL_FILE_NAME = "prompts.csv"
ARTIFACT_PATH = "tests/data/prompts-responses.csv"
ARTIFACT_NAME = "prompts-responses.csv"


@pytest.fixture(scope="module")
def mlflow_tmpdir():
    """Fixture to setup the temporary directory for MLflow."""
    tmpdir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{tmpdir}")
    return tmpdir


@pytest.fixture(scope="module")
def mlflow_experiment_id(mlflow_tmpdir):
    """Fixture to provide the MLflow experiment ID."""
    return mlflow.create_experiment(name="test-exp")


@pytest.fixture
def run_id_local_input(mlflow_experiment_id):
    """Run ID + LocalInput for a run that logged a local input."""
    with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
        yield (
            run.info.run_id,
            build_and_log_input(path=LOCAL_FILE_PATH),
        )


class TestLocalInput:
    def test_local_path(self, run_id_local_input):
        _, local_input = run_id_local_input
        path = local_input.local_path()
        assert isinstance(path, Path)
        assert str(path) == LOCAL_FILE_PATH

    def test_input_logging(self, run_id_local_input):
        """MLFlow integration test."""
        run_id, _ = run_id_local_input
        client = mlflow.tracking.MlflowClient()

        artifacts = client.list_artifacts(run_id)

        assert len(artifacts) == 1
        artifact = artifacts[0]

        assert artifact.path == Path(LOCAL_FILE_PATH).name


class TestDVCInput:
    @pytest.fixture
    @patch("modelplane.runways.data.dvc.api")
    def dvc_input(self, mock_dvc, tmpdir):
        repo = "https://github.com/fake-org/fake-repo.git"
        # Mock url following google cloud storage schema. Used to get the md5 hash.
        mock_dvc.get_url.return_value = "gs://repo/files/md5/01/abcdef1234"

        # Don't actually download anything. Just point to pre-existing local file.
        with patch.object(DVCInput, "_download_dvc_file", return_value=LOCAL_FILE_PATH):
            dvc_input = DVCInput(LOCAL_FILE_PATH, repo, tmpdir)
        return dvc_input

    def test_local_path(self, dvc_input):
        local_path = dvc_input.local_path()

        assert isinstance(local_path, Path)
        assert str(local_path) == LOCAL_FILE_PATH

    def test_input_logging(self, dvc_input, mlflow_experiment_id):
        """MLFlow integration test."""
        with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
            dvc_input = build_and_log_input(
                input_object=dvc_input,
            )
        client = mlflow.tracking.MlflowClient()

        artifacts = client.list_artifacts(run.info.run_id)

        assert len(artifacts) == 1
        artifact = artifacts[0]

        assert artifact.path == Path(LOCAL_FILE_PATH).name


class TestMLFlowArtifactInput:
    def test_local_path(self, run_id_local_input, tmpdir):
        expected_download_path = os.path.join(tmpdir, LOCAL_FILE_NAME)
        run_id, _ = run_id_local_input

        mlflow_input = MLFlowArtifactInput(run_id, LOCAL_FILE_NAME, tmpdir)
        path = mlflow_input.local_path()

        assert isinstance(path, Path)
        assert str(path) == expected_download_path

    def test_download_artifact(self, run_id_local_input, tmpdir):
        """Integration test that downloads a real MLflow artifact from a run with different input types."""
        # Get the actual run_id from the fixture
        run_id, _ = run_id_local_input
        mlflow.log_artifact(ARTIFACT_PATH, run_id=run_id)

        mlflow_input = MLFlowArtifactInput(run_id, ARTIFACT_NAME, tmpdir)

        # Verify the file was actually downloaded
        downloaded_file_path = mlflow_input.local_path()
        expected_path = Path(tmpdir) / ARTIFACT_NAME
        assert downloaded_file_path == expected_path
        assert downloaded_file_path.exists()

        # Verify the content matches the original
        with open(ARTIFACT_PATH, "r") as original_file:
            original_content = original_file.read()

        with open(downloaded_file_path, "r") as downloaded_file:
            downloaded_content = downloaded_file.read()

        assert original_content == downloaded_content


class TestBuildAndLogInput:
    def test_build_local_input(self, run_id_local_input):
        """No run_id nor dvc_repo should result in LocalInput."""
        run_id, _ = run_id_local_input
        inp = build_and_log_input(path=LOCAL_FILE_PATH)
        assert isinstance(inp, LocalInput)

    def test_build_local_input_ignores_dest_dir(self, run_id_local_input):
        run_id, _ = run_id_local_input
        inp = build_and_log_input(path=LOCAL_FILE_PATH, dest_dir="fake_dir")
        assert isinstance(inp, LocalInput)

    @patch("modelplane.runways.data.dvc.api")
    def test_build_dvc_input(self, mock_dvc, run_id_local_input):
        mock_dvc.get_url.return_value = "url"
        run_id, _ = run_id_local_input
        with patch.object(DVCInput, "_download_dvc_file", return_value=LOCAL_FILE_PATH):
            inp = build_and_log_input(
                path=LOCAL_FILE_PATH, dvc_repo="some-repo", dest_dir="fake_dir"
            )
        assert isinstance(inp, DVCInput)

    def test_build_dvc_input_no_path_raises_error(self, run_id_local_input):
        with pytest.raises(ValueError, match="Path must be provided"):
            build_and_log_input(dvc_repo="some-repo", dest_dir="fake_dir")

    def test_build_mlf_input(self, run_id_local_input):
        run_id, _ = run_id_local_input
        inp = build_and_log_input(
            run_id=run_id, artifact_path=LOCAL_FILE_NAME, dest_dir="fake_dir"
        )
        assert isinstance(inp, MLFlowArtifactInput)

    def test_build_mlf_input_no_artifact_path_raises_error(self, run_id_local_input):
        run_id, _ = run_id_local_input
        with pytest.raises(ValueError, match="Artifact path must be provided"):
            build_and_log_input(run_id=run_id, dest_dir="fake_dir")

    def test_run_id_and_path_error(self, run_id_local_input):
        run_id, _ = run_id_local_input
        with pytest.raises(ValueError, match="Cannot provide both path and run_id"):
            build_and_log_input(
                run_id=run_id, path=LOCAL_FILE_PATH, dest_dir="fake_dir"
            )

    def test_run_id_and_repo_error(self, run_id_local_input):
        run_id, _ = run_id_local_input
        with pytest.raises(ValueError, match="Cannot provide both run_id and dvc_repo"):
            build_and_log_input(
                run_id=run_id,
                path=LOCAL_FILE_PATH,
                dvc_repo="some_repo",
                dest_dir="fake_dir",
            )

    def test_no_args_error(self, run_id_local_input):
        with pytest.raises(ValueError, match="Either path or run_id must be provided"):
            build_and_log_input()

    def test_no_current_mlflow_run(self):
        with pytest.raises(RuntimeError, match=_MLFLOW_REQUIRED_ERROR_MESSAGE):
            build_and_log_input()
