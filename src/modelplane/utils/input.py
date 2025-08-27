import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import dvc.api
import mlflow
import mlflow.artifacts
import pandas as pd


class BaseInput(ABC):
    """Base class for input datasets."""

    input_type: str

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "input_type"):
            raise TypeError(f"{cls.__name__} must define class attribute 'input_type'")

    def log_artifact(self):
        """Log the dataset to MLflow as an artifact to the current run."""
        mlflow.log_artifact(str(self.local_path()))
        mlflow.set_tags(self.input_tags())

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
    input_obj: Optional[BaseInput] = None,
    path: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dvc_repo: Optional[str] = None,
    dest_dir: str = "",
    df: Optional[pd.DataFrame] = None,
) -> BaseInput:
    # Direct input
    if input_obj is not None:
        inp = input_obj
    # DF case
    elif df is not None:
        inp = DataframeInput(df, dest_dir=dest_dir)
    # DVC case
    elif dvc_repo is not None:
        if path is None:
            raise ValueError("Path must be provided when dvc_repo is provided.")
        if run_id is not None:
            raise ValueError(
                "Cannot provide both run_id and dvc_repo to build an input."
            )
        inp = DVCInput(path=path, repo=dvc_repo, dest_dir=dest_dir)
    # Local case
    elif path is not None:
        if run_id is not None:
            raise ValueError("Cannot provide both path and run_id.")
        inp = LocalInput(path)
    # MLFlow artifact case
    elif run_id is not None:
        if artifact_path is None:
            raise ValueError("Artifact path must be provided when run_id is provided.")
        inp = MLFlowArtifactInput(run_id, artifact_path, dest_dir)
    else:
        raise ValueError("Either path or run_id must be provided to build an input.")
    inp.log_artifact()
    return inp
