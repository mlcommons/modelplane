import os
from typing import List

import mlflow

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import (
    SECRETS_PATH,
    load_secrets_from_config,
    raise_if_missing_from_config,
)
from modelgauge.secret_values import RawSecrets
from modelgauge.sut_registry import SUTS

# Path to the secrets toml file
SECRETS_PATH_ENV = "MODEL_SECRETS_PATH"
DEBUG_MODE_ENV = "MODELPLANE_DEBUG_MODE"
PROMPT_RESPONSE_ARTIFACT_NAME = "prompt-responses.csv"
ANNOTATION_RESPONSE_ARTIFACT_NAME = "annotations.csv"
RUN_TYPE_TAG_NAME = "type"
RUN_TYPE_RESPONDER = "get-sut-responses"
RUN_TYPE_ANNOTATOR = "annotate"
RUN_TYPE_SCORER = "score"


def is_debug_mode() -> bool:
    """
    Check if the debug mode is enabled.
    """
    return os.getenv(DEBUG_MODE_ENV, "false").lower() == "true"


def setup_sut_credentials(uid: str) -> RawSecrets:
    missing_secrets = []
    secrets = safe_load_secrets_from_config()
    missing_secrets.extend(SUTS.get_missing_dependencies(uid, secrets=secrets))
    raise_if_missing_from_config(missing_secrets)
    return secrets


def setup_annotator_credentials(uids: List[str]) -> RawSecrets:
    missing_secrets = []
    secrets = safe_load_secrets_from_config()
    for uid in uids:
        missing_secrets.extend(
            ANNOTATORS.get_missing_dependencies(uid, secrets=secrets)
        )
    raise_if_missing_from_config(missing_secrets)
    return secrets


def safe_load_secrets_from_config() -> RawSecrets:
    path = os.getenv(SECRETS_PATH_ENV, SECRETS_PATH)
    if os.path.exists(path):
        return load_secrets_from_config(path=path)
    return {}


def get_experiment_id(experiment_name: str) -> str:
    """
    Get the experiment ID from MLflow. If the experiment does not exist, create it.
    """
    # check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(name=experiment_name)
    elif experiment is not None and experiment.lifecycle_stage != "active":
        raise ValueError(
            f"Experiment '{experiment_name}' exists but is not active. "
            "Please delete it or create a new experiment with a different name."
        )
    else:
        return experiment.experiment_id
