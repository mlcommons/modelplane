./start_services.sh -d --vllm
docker cp tests/notebooks/run.py modelplane-jupyter-1:/app/test_notebooks.py
docker exec modelplane-jupyter-1 poetry run python /app/test_notebooks.py
docker compose down
