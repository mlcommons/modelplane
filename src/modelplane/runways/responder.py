"""Runway for getting responses from SUTs."""

import pathlib
import tempfile

import click
import mlflow

from modelgauge.load_plugins import load_plugins
from modelgauge.pipeline_runner import PromptRunner
from modelgauge.sut_registry import SUTS

from modelplane.mlflow.datasets import get_mlflow_dataset
from modelplane.runways.utils import (
    get_experiment_id,
    is_debug_mode,
    setup_sut_credentials,
)
from modelplane.utils.env import load_from_dotenv

load_plugins(disable_progress_bar=True)


@click.command(name="get-sut-responses")
@click.option(
    "--sut_id",
    type=str,
    required=True,
    help="The SUT UID to use.",
)
@click.option(
    "--prompts",
    type=str,
    required=True,
    help="The path to the input prompts file.",
)
@click.option(
    "--experiment",
    type=str,
    required=True,
    help="The experiment name to use. If the experiment does not exist, it will be created.",
)
@click.option(
    "--cache_dir",
    type=str,
    default=None,
    help="The cache directory. Defaults to None. Local directory used to cache LLM responses.",
)
@click.option(
    "--n_jobs",
    type=int,
    default=1,
    help="The number of jobs to run in parallel. Defaults to 1.",
)
@load_from_dotenv
def get_sut_responses(
    sut_id: str,
    prompts: str,
    experiment: str,
    cache_dir: str | None = None,
    n_jobs: int = 1,
):
    """
    Run the pipeline to get responses from SUTs.
    """
    return respond(
        sut_id=sut_id,
        prompts=prompts,
        experiment=experiment,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
    )


def respond(
    sut_id: str,
    prompts: str,
    experiment: str,
    cache_dir: str | None = None,
    n_jobs: int = 1,
) -> str:
    secrets = setup_sut_credentials(sut_id)
    sut = SUTS.make_instance(uid=sut_id, secrets=secrets)
    params = {
        "cache_dir": cache_dir,
        "n_jobs": n_jobs,
    }
    tags = {"sut_id": sut_id}
    dataset = get_mlflow_dataset(prompts)

    experiment_id = get_experiment_id(experiment)

    with mlflow.start_run(experiment_id=experiment_id, tags=tags):
        mlflow.log_params(params)
        mlflow.log_input(dataset)

        # Use temporary file as mlflow will log this into the artifact store
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_runner = PromptRunner(
                num_workers=n_jobs,
                input_path=pathlib.Path(prompts),
                output_dir=pathlib.Path(tmp),
                cache_dir=cache_dir,
                suts={sut_id: sut},
            )

            pipeline_runner.run(
                progress_callback=mlflow.log_metrics, debug=is_debug_mode()
            )

            # log the output to mlflow's artifact store
            mlflow.log_artifact(
                local_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
            )
        return mlflow.active_run().info.run_id  # type: ignore
