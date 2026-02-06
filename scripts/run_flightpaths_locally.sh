# starts services required for unit tests
./start_services.sh -d --vllm
# copy test script
docker cp tests/notebooks/run.py modelplane-jupyter-1:/app/test_notebooks.py
# run tests
docker exec modelplane-jupyter-1 poetry run python /app/test_notebooks.py
# stop services
docker compose down
