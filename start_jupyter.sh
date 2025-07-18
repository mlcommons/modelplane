#!/bin/bash
set -e

ENV_FILE=".env.jupyteronly"

DETACHED=""

for arg in "$@"; do
  case $arg in
    -d)
      DETACHED="-d"
      ;;
  esac
done

# Load specified env file
set -a
source "$ENV_FILE"
set +a

if [ -n "$SSH_AUTH_SOCK" ] && ssh-add -l >/dev/null 2>&1; then
  SSH_FLAG="--ssh default"
else
  SSH_FLAG=""
fi

docker compose down jupyter
docker compose build $SSH_FLAG jupyter
docker compose up $DETACHED jupyter
