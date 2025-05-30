"""Runway for getting responses from SUTs."""

import pathlib
import tempfile

import mlflow

from modelgauge.pipeline_runner import PromptRunner
from modelgauge.sut_registry import SUTS

from modelplane.mlflow.datasets import get_mlflow_dataset
from modelplane.runways.utils import (
    get_experiment_id,
    is_debug_mode,
    setup_sut_credentials,
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
