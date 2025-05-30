# Ensures the mlflow tracking server is live.

from modelplane.mlflow.health import tracking_server_is_live


def test_tracking_server_is_live():
    """Test if the MLflow tracking server is live."""
    assert tracking_server_is_live(), "MLflow tracking server should be live"
