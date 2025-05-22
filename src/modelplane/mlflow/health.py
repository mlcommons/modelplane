import mlflow
import requests


def tracking_server_is_live() -> bool:
    """Check if the tracking server is live."""
    try:
        uri = mlflow.get_tracking_uri()
        health_uri = f"{uri}/health"
        response = requests.get(health_uri)
        response.raise_for_status()
        return True
    except Exception:
        return False
