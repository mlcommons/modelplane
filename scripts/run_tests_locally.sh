# starts services required for unit tests
./start_services.sh --no-jupyter -d
# run unit tests
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run pytest
# stop services
docker compose down
