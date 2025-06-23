#!/bin/bash

# load .env
set -a
source .env
set +a

# if USE_JUPYTER, set MLFLOW_TRACKING_URI appropriately
if [ "$USE_JUPYTER" = "true" ]; then
  docker compose down && docker compose build --ssh default && MLFLOW_TRACKING_URI="http://mlflow:8080" docker compose up
else
  docker compose down && docker compose build --ssh default && MLFLOW_TRACKING_URI="http://localhost:8080" docker compose up mlflow
fi
