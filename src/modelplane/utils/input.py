import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import dvc.api
import mlflow

from modelplane.mlflow.datasets import LocalDatasetSource


class BaseInput(ABC):
    """Base class for input datasets."""

    @abstractmethod
    def log_input(self):
        """Log the dataset to MLflow as input. This method should only be called inside an active MLflow run."""
        pass

    @abstractmethod
    def local_path(self) -> Path:
        pass


class LocalInput(BaseInput):
    """A dataset that is stored locally."""

    def __init__(self, path: str):
        self.path = path

    def log_input(self):
        mlf_dataset = mlflow.data.meta_dataset.MetaDataset(
            source=LocalDatasetSource(uri=self.path),
            name=self.path,
        )
        mlflow.log_input(mlf_dataset)

    def local_path(self) -> Path:
        return Path(self.path)


class DVCInput(BaseInput):
    """A dataset from a DVC remote."""

    def __init__(self, path: str, repo: str, dest_dir: str):
        self.path = path
        self.rev = "main"
        self.url = dvc.api.get_url(path, repo=repo, rev=self.rev)  # For logging.
        self._local_path = self._download_dvc_file(path, repo, dest_dir)

    def _download_dvc_file(self, path: str, repo: str, dest_dir: str) -> str:
        local_path = os.path.join(dest_dir, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with dvc.api.open(path=path, repo=repo, rev=self.rev, mode="rb") as source_file:
            with open(local_path, "wb") as dest_file:
                shutil.copyfileobj(source_file, dest_file)

        return local_path

    def digest(self) -> str:
        """Return the md5 hash of the dvc file."""
        # TODO: Check if this works with other storage options (besides google cloud)
        segments = self.url.split("/")
        i = segments.index("md5")
        digest = "".join(segments[i + 1 :])
        return digest

    def log_input(self):
        dataset = mlflow.data.meta_dataset.MetaDataset(
            source=mlflow.data.http_dataset_source.HTTPDatasetSource(self.url),
            name=self.path,
            digest=self.digest(),
        )
        mlflow.log_input(dataset)

    def local_path(self) -> Path:
        return Path(self._local_path)


class MLFlowArtifactInput(BaseInput):
    """A dataset artifact from a previous MLFlow run."""

    def __init__(self, run_id: str, artifact_path: str, dest_dir: str):
        self.run_id = run_id
        self._local_path = self._download_artifacts(run_id, artifact_path, dest_dir)

    def _download_artifacts(
        self, run_id: str, artifact_path: str, dest_dir: str
    ) -> str:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dest_dir,
        )
        return os.path.join(dest_dir, artifact_path)

    def log_input(self):
        run = mlflow.get_run(self.run_id)
        for input in run.inputs.dataset_inputs:
            ds = input.dataset
            source_dict = json.loads(ds.source)
            if ds.source_type == "http":
                source = mlflow.data.http_dataset_source.HTTPDatasetSource(
                    source_dict["url"]
                )
            else:
                source = LocalDatasetSource.from_dict(source_dict)
            dataset = mlflow.data.dataset.Dataset(
                source=source, name=ds.name, digest=ds.digest
            )
            mlflow.log_input(dataset)

    def local_path(self) -> Path:
        return Path(self._local_path)


def build_input(
    path: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dvc_repo: Optional[str] = None,
    dest_dir: Optional[str] = None,
) -> BaseInput:
    if dvc_repo is not None:
        if path is None:
            raise ValueError("Path must be provided when dvc_repo is provided.")
        return DVCInput(path=path, repo=dvc_repo, dest_dir=dest_dir)
    elif path is not None:
        if run_id is not None:
            raise ValueError("Cannot provide both path and run_id.")
        return LocalInput(path)
    elif run_id is not None:
        if artifact_path is None:
            raise ValueError("Artifact path must be provided when run_id is provided.")
        return MLFlowArtifactInput(run_id, artifact_path, dest_dir)
    else:
        raise ValueError("Either path or run_id must be provided to build an input.")
