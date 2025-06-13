import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import mlflow

from modelplane.mlflow.datasets import LocalDatasetSource, get_mlflow_dataset


class BaseInput(ABC):
    """Base class for input datasets."""

    @abstractmethod
    def log_input(self):
        """Log the dataset to MLflow as input."""
        pass

    @abstractmethod
    def local_path(self) -> str:
        pass


class LocalInput(BaseInput):
    """A dataset that is stored locally."""

    def __init__(self, path: str):
        self.path = path

    def log_input(self):
        mlf_dataset = get_mlflow_dataset(self.path, source_type="local")
        mlflow.log_input(mlf_dataset)

    def local_path(self) -> Path:
        return Path(self.path)


class MLFlowArtifactInput(BaseInput):
    """A dataset artifact from a previous MLFlow run."""

    def __init__(self, run_id: str, artifact_path: str, dest_dir: str):
        self.run_id = run_id
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dest_dir,
        )
        self.path = os.path.join(dest_dir, artifact_path)

    def log_input(self):
        run = mlflow.get_run(self.run_id)
        for input in run.inputs.dataset_inputs:
            ds = input.dataset
            source_dict = json.loads(ds.source)
            # TODO: Shouldn't source be mlFLOW?
            source = LocalDatasetSource.from_dict(source_dict)
            dataset = mlflow.data.dataset.Dataset(
                source=source, name=ds.name, digest=ds.digest
            )
            mlflow.log_input(dataset)

    def local_path(self) -> Path:
        return Path(self.path)
