./start_services.sh --no-jupyter -d
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run pytest
docker compose down
