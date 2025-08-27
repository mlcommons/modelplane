from dataclasses import dataclass
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import dvc.api
import mlflow
import mlflow.artifacts
import pandas as pd

_MLFLOW_REQUIRED_ERROR_MESSAGE = (
    "An active MLflow run is required to log input artifacts."
)


class Artifact:

    def __init__(self, experiment_id: str, run_id: str, name: str):
        self.name = name
        tracking_uri = mlflow.get_tracking_uri()
        self._mlflow_link = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}/artifacts/{name}"
        self._download_link = f"{tracking_uri}/get-artifact?run_id={run_id}&path={name}"

    @property
    def mlflow_link(self) -> str:
        return self._mlflow_link

    @property
    def download_link(self) -> str:
        return self._download_link


@dataclass
class RunArtifacts:
    run_id: str
    artifacts: dict[str, Artifact | None]


class BaseInput(ABC):
    """Base class for input datasets."""

    input_type: str

    def __init__(self):
        self.input_run_id = None
        self._artifact = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "input_type"):
            raise TypeError(f"{cls.__name__} must define class attribute 'input_type'")

    def log_artifact(self):
        """Log the dataset to MLflow as an artifact to the current run."""
        if self.input_run_id is not None:
            raise ValueError(
                f"Input has already been logged with an input_run_id: {self.input_run_id}"
            )
        current_run = mlflow.active_run()
        if current_run is None:
            raise ValueError("An active MLflow run is required to log input artifacts.")
        local = self.local_path()
        mlflow.log_artifact(str(local))
        mlflow.set_tags(self.input_tags())
        self._artifact = Artifact(
            experiment_id=current_run.info.experiment_id,
            run_id=current_run.info.run_id,
            name=local.name,
        )

    @property
    def artifact(self) -> Artifact | None:
        return self._artifact

    @abstractmethod
    def local_path(self) -> Path:
        pass

    def input_tags(self) -> dict:
        tags = {"input_type": self.input_type}
        tags.update(self.tags_for_input_type)
        return tags

    @property
    @abstractmethod
    def tags_for_input_type(self) -> dict:
        pass


class LocalInput(BaseInput):
    """A dataset that is stored locally."""

    input_type = "local"

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def local_path(self) -> Path:
        return Path(self.path)

    @property
    def tags_for_input_type(self) -> dict:
        return {"input_path": self.path}


class DataframeInput(BaseInput):
    """A dataset that is represented as a Pandas DataFrame."""

    input_type = "dataframe"
    _INPUT_FILE_NAME = "input.csv"

    def __init__(self, df: pd.DataFrame, dest_dir: str):
        super().__init__()
        self._local_path = Path(dest_dir) / self._INPUT_FILE_NAME
        self.df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame):
        self._df = df
        self._update_local_file()

    def _update_local_file(self):
        self.df.to_csv(self._local_path, index=False)

    def local_path(self) -> Path:
        return self._local_path

    @property
    def tags_for_input_type(self) -> dict:
        return {}


class DVCInput(BaseInput):
    """A dataset from a DVC remote."""

    input_type = "dvc"

    def __init__(self, path: str, repo: str, dest_dir: str):
        super().__init__()
        repo_path = repo.split("#")
        if len(repo_path) == 2:
            repo, self.rev = repo_path
        else:
            self.rev = "main"
        self._local_path = self._download_dvc_file(path, repo, dest_dir)
        self._tags = {"input_repo": repo, "input_rev": self.rev, "input_path": path}

    def _download_dvc_file(self, path: str, repo: str, dest_dir: str) -> str:
        local_path = os.path.join(dest_dir, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with dvc.api.open(path=path, repo=repo, rev=self.rev, mode="rb") as source_file:
            with open(local_path, "wb") as dest_file:
                shutil.copyfileobj(source_file, dest_file)

        return local_path

    def local_path(self) -> Path:
        return Path(self._local_path)

    @property
    def tags_for_input_type(self) -> dict:
        return self._tags


class MLFlowArtifactInput(BaseInput):
    """A dataset artifact from a previous MLFlow run."""

    input_type = "artifact"

    def __init__(self, run_id: str, artifact_path: str, dest_dir: str):
        super().__init__()
        self.run_id = run_id
        self._local_path = self._download_artifacts(run_id, artifact_path, dest_dir)
        self._tags = {"input_run_id": run_id, "input_artifact_path": artifact_path}

    def _download_artifacts(
        self, run_id: str, artifact_path: str, dest_dir: str
    ) -> str:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dest_dir,
        )
        return os.path.join(dest_dir, artifact_path)

    def local_path(self) -> Path:
        return Path(self._local_path)

    @property
    def tags_for_input_type(self) -> dict:
        return self._tags


def build_and_log_input(
    input_object: Optional[BaseInput] = None,
    path: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dvc_repo: Optional[str] = None,
    dest_dir: str = "",
    df: Optional[pd.DataFrame] = None,
) -> BaseInput:
    if mlflow.active_run() is None:
        raise RuntimeError(_MLFLOW_REQUIRED_ERROR_MESSAGE)
    inp = build_input(
        input_object=input_object,
        path=path,
        run_id=run_id,
        artifact_path=artifact_path,
        dvc_repo=dvc_repo,
        dest_dir=dest_dir,
        df=df,
    )
    inp.log_artifact()
    return inp


def build_input(
    input_object: Optional[BaseInput] = None,
    path: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dvc_repo: Optional[str] = None,
    dest_dir: str = "",
    df: Optional[pd.DataFrame] = None,
) -> BaseInput:
    # Direct input
    if input_object is not None:
        return input_object
    # DF case
    elif df is not None:
        return DataframeInput(df, dest_dir=dest_dir)
    # DVC case
    elif dvc_repo is not None:
        if path is None:
            raise ValueError("Path must be provided when dvc_repo is provided.")
        if run_id is not None:
            raise ValueError(
                "Cannot provide both run_id and dvc_repo to build an input."
            )
        return DVCInput(path=path, repo=dvc_repo, dest_dir=dest_dir)
    # Local case
    elif path is not None:
        if run_id is not None:
            raise ValueError("Cannot provide both path and run_id.")
        return LocalInput(path)
    # MLFlow artifact case
    elif run_id is not None:
        if artifact_path is None:
            raise ValueError("Artifact path must be provided when run_id is provided.")
        return MLFlowArtifactInput(run_id, artifact_path, dest_dir)
    raise ValueError("Either path or run_id must be provided to build an input.")
