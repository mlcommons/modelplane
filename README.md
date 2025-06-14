# ModelPlane

Develop new evaluators / annotators.

## Get Started

You must have docker installed on your system. The
given docker-compose.yaml file will start up:

* mlflow tracking server + postgres
* jupyter

1.  Clone the repository:
    ```bash
    git clone https://github.com/mlcommons/modelplane.git
    cd modelplane
    ```
1. Environment:
    1. Adjust the .env file as needed. The committed .env / 
    docker-compose.yaml will bring up mlflow, postgres, jupyter, set up
    mlflow to use a local disk for artifact storage.
    1. Set up secrets for accessing SUTs, as needed in 
    `modelplane/flightpaths/config/secrets.toml`. See [modelbench](https://github.com/mlcommons/modelbench) for more details.
    1. Stage your input data in `modelplane/flightpaths/data`. You can get a
    sample input file [here](https://github.com/mlcommons/ailuminate/tree/main).
1. Bring up the services:
    ```bash
    docker compose up -d
    ```
    Or if you are running mlflow somewhere else, you can bring up just jupyter with:
    ```bash
    docker compose up -d jupyter
    ```
1. Visit the [Jupyter Server](http://localhost:8888/?token=changeme). The
   token is configured in the .env file. You shouldn't need to enter it 
   more than once (until the server is restarted). You can get started with
   the template notebook or create a new one.
1. The runs can be monitored in MLFlow wherever you have that set up. If
   local with the default setup, http://localhost:8080.

## CLI

You can also interact with modelplane via CLI. Run `poetry run modelplane --help`
for more details.

*Important:* You must set the `MLFLOW_TRACKING_URI` environmental variable.
For example, if you've brought up MLFlow using the docker compose process above,
you could run:
```
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run modelplane get-sut-responses --sut_id {sut_id} --prompts tests/data/prompts.csv --experiment expname
```
After running the command, you'd see the `run_id` in the output from mlflow, 
or you can get the `run_id` via the MLFlow UI.

Then you can run annotations with:
```
MLFLOW_TRACKING_URI=http://localhost:8080 poetry run modelplane annotate --annotator_id {annotator_id} --experiment expname --response_run_id {run_id}
```

## TODO

- [ ] Scoring against ground truth (measurement runner functionality)
- [ ] Support ensemble option
- [ ] Support multiple annotators in single run
- [ ] Confirm this works with cloud storage
- [ ] Add test coverage
- [ ] Support for data via remote DVC repo
- [ ] Template with annotator that's served elsewhere
- [ ] Missing safety runner functionality
- [ ] Automated experiment names
- [ ] `annotate` should add sut_id tag to its runs
- [ ] Better handling of jupyter token
