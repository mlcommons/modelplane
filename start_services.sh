#!/bin/bash

# load .env
set -a
source .env
set +a

# Check if SSH agent is running
if ssh-add -l >/dev/null 2>&1; then
  SSH_FLAG="--ssh default"
else
  echo "SSH agent not running. Proceeding without SSH."
  SSH_FLAG=""
fi

# Default values
USE_JUPYTER=true
DETACHED=""

# Parse arguments
for arg in "$@"; do
  case $arg in
    no-jupyter)
      USE_JUPYTER=false
      ;;
    -d)
      DETACHED="-d"
      ;;
  esac
done

# Start services based on the options
if [ "$USE_JUPYTER" = "true" ]; then
  docker compose down && docker compose build $SSH_FLAG mlflow && MLFLOW_TRACKING_URI="http://mlflow:8080" docker compose up $DETACHED
else
  docker compose down && docker compose build $SSH_FLAG && MLFLOW_TRACKING_URI="http://localhost:8080" docker compose up $DETACHED mlflow
fi