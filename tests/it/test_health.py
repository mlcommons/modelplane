# Ensures the mlflow tracking server is live.

import mlflow
from modelplane.mlflow.health import tracking_server_is_live


def test_tracking_server_is_live():
    """Test if the MLflow tracking server is live."""
    # TODO: remove this
    print("Checking if MLflow tracking server is live...")
    print("Current tracking URI:", mlflow.get_tracking_uri())
    assert tracking_server_is_live(), "MLflow tracking server should be live"
