from pathlib import Path
from typing import Any

import mlflow.data.dataset
import mlflow.data.dataset_source
import mlflow.data.meta_dataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.utils.uri import is_local_uri


class LocalDatasetSource(FileSystemDatasetSource):
    """This tries to follow the same pattern as the ArtifactRepoSource class
    implementation of the abstract class FileSystemDatasetSource. That one
    doesn't directly address what we need, including ensuring the metadata
    includes something more specific in the version than just the configuration.
    In particular, we inject the timestamp of the file, and in the future
    will inject the DVC version for files that are tracked by DVC.
    """

    LOCAL_SOURCE_TYPE = "local"

    def __init__(self, uri: str):
        self._uri = uri

    @property
    def uri(self):  # type: ignore
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return LocalDatasetSource.LOCAL_SOURCE_TYPE

    def load(self, dst_path=None) -> str:
        raise NotImplementedError("LocalSource does not support loading.")

    @staticmethod
    def _can_resolve(raw_source: Any):
        if not isinstance(raw_source, str) and not isinstance(raw_source, Path):
            return False

        try:
            return is_local_uri(str(raw_source), is_tracking_or_registry_uri=False)
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> "LocalDatasetSource":
        return cls(str(raw_source))

    def to_dict(self) -> dict[Any, Any]:
        # Add timestamp so the digest computation uses the file timestamp.
        return {
            "uri": self.uri,
            "timestamp": Path(self.uri).stat().st_mtime,
        }

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> "LocalDatasetSource":
        uri = source_dict.get("uri")
        if not isinstance(uri, str):
            raise ValueError(
                "The 'uri' field must be present and of type str in source_dict."
            )
        return cls(uri=uri)


def get_mlflow_dataset(
    path: str, source_type: str = "local"
) -> mlflow.data.dataset.Dataset:
    """Get the MLflow dataset from the input file path."""
    if source_type == "local":
        return mlflow.data.meta_dataset.MetaDataset(
            source=LocalDatasetSource(uri=path),
            name=path,
        )
    else:
        raise ValueError(f"Unsupported source type: {source_type}.")


def get_dataset_source_cls(
    source_type: str,
) -> type[mlflow.data.dataset_source.DatasetSource]:
    if source_type == "local":
        return LocalDatasetSource
    else:
        raise ValueError(f"Unsupported source type: {source_type}.")
