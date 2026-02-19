#!/usr/bin/env bash
set -uo pipefail

uv lock --check
uv sync

services_started=false
cleanup() {
	local exit_code=$?
	if [[ "${services_started}" == "true" ]]; then
		docker compose down
	fi
	exit ${exit_code}
}
trap cleanup EXIT

if ./start_services.sh -d --vllm; then
	services_started=true
else
	exit 1
fi

docker cp tests/notebooks/run.py modelplane-jupyter-1:/app/test_notebooks.py
docker exec modelplane-jupyter-1 uv run python /app/test_notebooks.py
