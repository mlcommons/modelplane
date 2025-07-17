#!/bin/bash
set -e

ENV_FILE=".env"

USE_JUPYTER=true
DETACHED=""
SERVICES=("mlflow")

for arg in "$@"; do
  case $arg in
    --no-jupyter)
      USE_JUPYTER=false
      ENV_FILE=".env.nojupyter"
      ;;
    -d)
      DETACHED="-d"
      ;;
    --vllm)
      SERVICES+=("vllm")
      ;;
  esac
done

if [ "$USE_JUPYTER" = "true" ]; then
  SERVICES+=("jupyter")
fi

# Load specified env file
set -a
source "$ENV_FILE"
set +a

if [ -n "$SSH_AUTH_SOCK" ] && ssh-add -l >/dev/null 2>&1; then
  SSH_FLAG="--ssh default"
else
  echo "SSH agent not running. Proceeding without SSH."
  SSH_FLAG=""
fi

docker compose down "${SERVICES[@]}"
docker compose build $SSH_FLAG "${SERVICES[@]}"
docker compose up $DETACHED "${SERVICES[@]}"
