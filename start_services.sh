#!/bin/bash

# load .env
set -a
source .env
set +a

# Default to starting Jupyter unless explicitly disabled
if [ "$1" = "no-jupyter" ]; then
  docker compose down && docker compose build --ssh default && MLFLOW_TRACKING_URI="http://localhost:8080" docker compose up -d mlflow
else
  docker compose down && docker compose build --ssh default && MLFLOW_TRACKING_URI="http://mlflow:8080" docker compose up -d
fi
