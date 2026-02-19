#!/usr/bin/env bash
set -uo pipefail

uv lock --check
uv sync --group test

services_started=false
cleanup() {
	local exit_code=$?
	if [[ "${services_started}" == "true" ]]; then
		docker compose down
	fi
	exit ${exit_code}
}
trap cleanup EXIT

if ./start_services.sh --no-jupyter -d; then
	services_started=true
else
	exit 1
fi

MLFLOW_TRACKING_URI=http://localhost:8080 uv run pytest
