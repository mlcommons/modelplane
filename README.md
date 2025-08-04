# modelplane - an AI evaluator development platform

## ⚠️ Content warning

The sample datasets provided in the [`flightpaths/data`](https://github.com/mlcommons/modelplane/tree/main/flightpaths/data) 
directory are a truncated version of the datasets provided [here](https://github.com/mlcommons/ailuminate).
These data come with the following warning:

>This dataset was created to elicit hazardous responses. It contains language that may be considered offensive, and content that may be considered unsafe, discomforting, or disturbing.
>Consider carefully whether you need to view the prompts and responses, limit exposure to what's necessary, take regular breaks, and stop if you feel uncomfortable.
>For more information on the risks, see [this literature review](https://www.zevohealth.com/wp-content/uploads/2024/07/lit_review_IN-1.pdf) on vicarious trauma.

## Quickstart

You must have a docker engine installed on your system. The given
docker-compose.yaml file has definitions for running the following services
locally:

* mlflow tracking server + postgres
* jupyter

First, clone this repo:
```bash
git clone https://github.com/mlcommons/modelplane.git
cd modelplane
```

If you plan to share notebooks, clone 
[modelplane-flights](https://github.com/mlcommons/modelplane-flights) as well. Both `modelplane`
and `modelplane-flights` should be in the same directory.

Finally, set up secrets for accessing SUTs, as needed in 
`modelplane/flightpaths/config/secrets.toml`. See [modelbench](https://github.com/mlcommons/modelbench) for more details.


### Running jupyter locally against the MLCommons mlflow server.

1. Ensure you have access to the MLCommons mlflow tracking 
and artifact server. If not, email 
[airr-engineering@mlcommons.org](mailto:airr-engineering@mlcommons.org)
for access.
1. Modify `.env.jupyteronly` to include  your credentials for the
MLFlow server (`MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD`).
    * Alternatively, put the credentials in `~/.mlflow/credentials` as described [here](https://mlflow.org/docs/latest/ml/auth/#credentials-file).
1. To access `modelbench-private` code (assuming you have 
access), you must also set `USE_MODELBENCH_PRIVATE=true` in `.env.jupyteronly`. This will forward your ssh agent to the container
allowing it to load the private repository to build the image.
1. Start jupyter with `./start_jupyter.sh`. (You can add the
`-d` flag to start in the background.)

### Running jupyter and mlflow locally.

1. Adjust the `.env` file as needed. The committed `.env` / 
`docker-compose.yaml` will bring up mlflow, postgres, jupyter, and set up mlflow to use a local disk for artifact storage.
1. Start services with `./start_services.sh`. (You can add the
`-d` flag to start in the background.)

    * If you are using the cli only, and not using jupyter, you must pass the `--no-jupyter` option: 
    `./start_services.sh -d`

## Getting started in JupyterLab.

1. Visit the [Jupyter Server](http://localhost:8888/lab?token=changeme). The
   token is configured in the .env file. You shouldn't need to enter it 
   more than once (until the server is restarted). You can get started with
   the template notebook or create a new one.
1. You should see the `flights` directory, which leads to the
`modelplane-flights` repository. Create a user directory
for yourself (`flights/users/{username}`) and either
copy an existing flightpath there or create a notebook from
scratch.
    * You can manage branches and commits for 
    `modelplane-flights` directly from jupyter.

## Caching

Annotator and SUT responses will be cached (locally) unless you pass the
`disable_cache` flag to the appropriate calls.

## CLI

You can also interact with modelplane via CLI. Run `poetry run modelplane --help`
for more details.

*Important:* You must set the `MLFLOW_TRACKING_URI` environmental variable.
For example, if you've brought up MLFlow using the fully local docker compose process above,
you could run:
```
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run modelplane get-sut-responses --sut_id {sut_id} --prompts tests/data/prompts.csv --experiment expname
```
After running the command, you'd see the `run_id` in the output from mlflow, 
or you can get the `run_id` via the MLFlow UI.

### Basic Annotations
```
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run modelplane annotate --annotator_id {annotator_id} --experiment expname --response_run_id {run_id}
```

### Custom Ensembles
```
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run modelplane annotate --annotator_id {annotator_id1} --annotator_id {annotator_id2} --ensemble_strategy {ensemble_strategy} --experiment expname --response_file path/to/response.csv
```

### Private Ensemble
If you have access to the private ensemble, you can install with the needed extras
```
poetry install --extras modelbench-private
```
And then run annotations with:
```
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run modelplane annotate --ensemble_id official --experiment expname --response_run_id {run_id}
```
