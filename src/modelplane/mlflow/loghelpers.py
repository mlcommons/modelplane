import mlflow

from modelplane.runways.utils import RUN_TYPE_TAG_NAME


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
