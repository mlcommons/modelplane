import json
import mlflow
import mlflow.data.dataset

from modelplane.mlflow.datasets import get_dataset_source_cls, get_mlflow_dataset
from modelplane.runways.utils import RUN_TYPE_TAG_NAME


def log_input(
    run_id: str | None = None, path: str | None = None, source_type: str = "local"
) -> None:
    if path is not None:
        dataset = get_mlflow_dataset(path, source_type=source_type)
        mlflow.log_input(dataset)
    elif run_id is not None:
        dataset_source_cls = get_dataset_source_cls(source_type)
        run = mlflow.get_run(run_id)
        for input in run.inputs.dataset_inputs:
            ds = input.dataset
            source_dict = json.loads(ds.source)
            source = dataset_source_cls.from_dict(source_dict)
            dataset = mlflow.data.dataset.Dataset(
                source=source, name=ds.name, digest=ds.digest
            )
            mlflow.log_input(dataset)
    else:
        raise ValueError("Exactly one of run_id or path must be provided.")


def log_tags(run_id: str) -> None:
    """
    Re-logs user tags from a prior run to current run.
    """
    run = mlflow.get_run(run_id)
    mlflow.set_tags(
        {
            k: v
            for k, v in run.data.tags.items()
            if not k.startswith("mlflow.") and k != RUN_TYPE_TAG_NAME
        }
    )
