import os

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
ANNOTATION_RESPONSE_ARTIFACT_NAME = "annotations.jsonl"


def is_debug_mode() -> bool:
    """
    Check if the debug mode is enabled.
    """
    return os.getenv(DEBUG_MODE_ENV, "false").lower() == "true"


def setup_sut_credentials(uid: str) -> RawSecrets:
    """Load secrets from the config file and check for missing secrets."""
    missing_secrets = []
    secrets_path = os.getenv(SECRETS_PATH_ENV, SECRETS_PATH)
    secrets = {}
    if os.path.exists(secrets_path):
        secrets = load_secrets_from_config(path=secrets_path)
    missing_secrets.extend(SUTS.get_missing_dependencies(uid, secrets=secrets))
    raise_if_missing_from_config(missing_secrets)
    return secrets


def setup_annotator_credentials(uid: str) -> RawSecrets:
    """Load secrets from the config file and check for missing secrets."""
    secrets = load_secrets_from_config(path=os.getenv(SECRETS_PATH_ENV, SECRETS_PATH))
    missing_secrets = []
    missing_secrets.extend(ANNOTATORS.get_missing_dependencies(uid, secrets=secrets))
    raise_if_missing_from_config(missing_secrets)
    return secrets


def get_experiment_id(experiment_name: str) -> str:
    """
    Get the experiment ID from MLflow. If the experiment does not exist, create it.
    """
    # check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id
