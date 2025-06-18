import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import mlflow

from modelplane.utils.input import (
    BaseInput,
    LocalInput,
    DVCInput,
    MLFlowArtifactInput,
    build_input,
)

LOCAL_FILE_PATH = "tests/data/prompts.csv"
HTTTP_PATH = LOCAL_FILE_PATH
ARTIFACT_PATH = "tests/data/prompts-responses.csv"
ARTIFACT_NAME = "prompts-responses.csv"
HTTP_ARTIFACT_DIGEST = "abc123def456"


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
def local_input():
    return LocalInput(LOCAL_FILE_PATH)


@pytest.fixture
def run_id_local_input(mlflow_experiment_id, local_input):
    """Run ID for a run that logged a local input."""
    with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
        local_input.log_input()
        # Log output artifact.
        mlflow.log_artifact(ARTIFACT_PATH)
        run_id = run.info.run_id
    return run_id


@pytest.fixture
def run_id_http_input(mlflow_experiment_id):
    """Run ID for a run that logged an HTTP dataset input."""
    with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
        # Create an HTTP dataset source
        http_url = "https://example.com/data.csv"
        http_dataset = mlflow.data.meta_dataset.MetaDataset(
            source=mlflow.data.http_dataset_source.HTTPDatasetSource(http_url),
            name=HTTTP_PATH,
            digest=HTTP_ARTIFACT_DIGEST,
        )
        mlflow.log_input(http_dataset)

        # Log output artifact.
        mlflow.log_artifact(ARTIFACT_PATH)
        run_id = run.info.run_id
    return run_id


class TestLocalInput:
    def test_local_path(self, local_input):
        path = local_input.local_path()
        assert isinstance(path, Path)
        assert str(path) == LOCAL_FILE_PATH

    def test_log_input(self, local_input, run_id_local_input):
        """MLFlow integration test."""
        client = mlflow.tracking.MlflowClient()
        inputs = client.get_run(run_id_local_input).inputs

        assert len(inputs.dataset_inputs) == 1
        dataset = inputs.dataset_inputs[0].dataset

        assert dataset.name == LOCAL_FILE_PATH
        assert dataset.source_type == "local"


class TestDVCInput:
    @pytest.fixture
    @patch("modelplane.utils.input.dvc.api")
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

    def test_digest(self, dvc_input):
        """Test digest method parses MD5 hash from URL."""
        digest = dvc_input.digest()
        assert digest == "01abcdef1234"

    def test_log_input(self, dvc_input, mlflow_experiment_id):
        """MLFlow integration test."""
        with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
            dvc_input.log_input()
            run_id = run.info.run_id
        client = mlflow.tracking.MlflowClient()
        inputs = client.get_run(run_id).inputs

        assert len(inputs.dataset_inputs) == 1
        dataset = inputs.dataset_inputs[0].dataset

        assert dataset.name == LOCAL_FILE_PATH
        assert dataset.source_type == "http"
        assert dataset.digest == "01abcdef1234"


class TestMLFlowArtifactInput:
    @patch("modelplane.utils.input.mlflow.artifacts")
    def test_local_path(self, mock_artifact, tmpdir):
        fake_run_id = "test_run_123"
        expected_download_path = os.path.join(tmpdir, ARTIFACT_NAME)

        mlflow_input = MLFlowArtifactInput(fake_run_id, ARTIFACT_NAME, tmpdir)
        path = mlflow_input.local_path()

        assert isinstance(path, Path)
        assert str(path) == expected_download_path

    @pytest.mark.parametrize(
        "run_id_fixture", ["run_id_local_input", "run_id_http_input"]
    )
    def test_downloads_artifact(self, request, run_id_fixture, tmpdir):
        """Integration test that downloads a real MLflow artifact from a run with different input types."""
        # Get the actual run_id from the fixture
        run_id = request.getfixturevalue(run_id_fixture)

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

    def test_log_input_local_source(
        self, mlflow_experiment_id, run_id_local_input, tmpdir
    ):
        """Test log_input for a local mlflow artifact with real MLflow integration."""
        mlflow_input = MLFlowArtifactInput(run_id_local_input, ARTIFACT_NAME, tmpdir)

        with mlflow.start_run(experiment_id=mlflow_experiment_id) as new_run:
            mlflow_input.log_input()

            # Verify the input was logged
            client = mlflow.tracking.MlflowClient()
            inputs = client.get_run(new_run.info.run_id).inputs

            # TODO: Shouldn't the actual artifact from the previous run also get logged?
            assert len(inputs.dataset_inputs) == 1
            dataset = inputs.dataset_inputs[0].dataset
            assert dataset.name == LOCAL_FILE_PATH
            assert dataset.source_type == "local"

    def test_log_input_http_source(
        self, mlflow_experiment_id, run_id_http_input, tmpdir
    ):
        """Test log_input for an HTTP source mlflow artifact with real MLflow integration."""
        mlflow_input = MLFlowArtifactInput(run_id_http_input, ARTIFACT_NAME, tmpdir)

        with mlflow.start_run(experiment_id=mlflow_experiment_id) as new_run:
            mlflow_input.log_input()

            # Verify the input was logged
            client = mlflow.tracking.MlflowClient()
            inputs = client.get_run(new_run.info.run_id).inputs

            # TODO: Shouldn't the actual artifact from the previous run also get logged?
            assert len(inputs.dataset_inputs) == 1
            dataset = inputs.dataset_inputs[0].dataset
            assert dataset.name == HTTTP_PATH
            assert dataset.source_type == "http"


class TestBuildInput:
    def test_build_local_input(self):
        """No run_id nor dvc_repo should result in LocalInput."""
        inp = build_input(path=LOCAL_FILE_PATH)
        assert isinstance(inp, LocalInput)

    def test_build_local_input_ignores_dest_dir(self):
        inp = build_input(path=LOCAL_FILE_PATH, dest_dir="fake_dir")
        assert isinstance(inp, LocalInput)

    @patch("modelplane.utils.input.dvc.api")
    def test_build_dvc_input(self, mock_dvc):
        mock_dvc.get_url.return_value = "url"
        with patch.object(DVCInput, "_download_dvc_file", return_value=LOCAL_FILE_PATH):
            inp = build_input(
                path=LOCAL_FILE_PATH, dvc_repo="some-repo", dest_dir="fake_dir"
            )
        assert isinstance(inp, DVCInput)

    def test_build_dvc_input_no_path_raises_error(self):
        with pytest.raises(ValueError, match="Path must be provided"):
            build_input(dvc_repo="some-repo", dest_dir="fake_dir")

    @patch("modelplane.utils.input.mlflow.artifacts")
    def test_build_mlf_input(self, run_id_local_input):
        inp = build_input(
            run_id=run_id_local_input, artifact_path=ARTIFACT_PATH, dest_dir="fake_dir"
        )
        assert isinstance(inp, MLFlowArtifactInput)

    def test_build_mlf_input_no_artifact_path_raises_error(self, run_id_local_input):
        with pytest.raises(ValueError, match="Artifact path must be provided"):
            build_input(run_id=run_id_local_input, dest_dir="fake_dir")

    def test_run_id_and_path_error(self, run_id_local_input):
        with pytest.raises(ValueError, match="Cannot provide both path and run_id"):
            build_input(
                run_id=run_id_local_input, path=LOCAL_FILE_PATH, dest_dir="fake_dir"
            )

    def test_run_id_and_repo_error(self):
        with pytest.raises(ValueError, match="Cannot provide both run_id and dvc_repo"):
            build_input(
                run_id=run_id_local_input,
                path=LOCAL_FILE_PATH,
                dvc_repo="some_repo",
                dest_dir="fake_dir",
            )

    def test_no_args_error(self):
        with pytest.raises(ValueError, match="Either path or run_id must be provided"):
            build_input()
