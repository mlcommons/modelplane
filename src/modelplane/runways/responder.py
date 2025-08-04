"""Runway for getting responses from SUTs."""

import pathlib
import tempfile

import mlflow

from modelgauge.pipeline_runner import PromptRunner
from modelgauge.sut_registry import SUTS

from modelplane.runways.utils import (
    MODELGAUGE_RUN_TAG_NAME,
    RUN_TYPE_RESPONDER,
    RUN_TYPE_TAG_NAME,
    get_experiment_id,
    is_debug_mode,
    setup_sut_credentials,
)
from modelplane.utils.input import build_input


def respond(
    sut_id: str,
    prompts: str,
    experiment: str,
    dvc_repo: str | None = None,
    cache_dir: str | None = None,
    num_workers: int = 1,
) -> str:
    secrets = setup_sut_credentials(sut_id)
    sut = SUTS.make_instance(uid=sut_id, secrets=secrets)
    params = {
        "cache_dir": cache_dir,
        "num_workers": num_workers,
    }
    tags = {"sut_id": sut_id, RUN_TYPE_TAG_NAME: RUN_TYPE_RESPONDER}

    experiment_id = get_experiment_id(experiment)

    with mlflow.start_run(experiment_id=experiment_id, tags=tags) as run:
        mlflow.log_params(params)
        # Use temporary file as mlflow will log this into the artifact store
        with tempfile.TemporaryDirectory() as tmp:
            input_data = build_input(path=prompts, dvc_repo=dvc_repo, dest_dir=tmp)
            input_data.log_input()
            pipeline_runner = PromptRunner(
                num_workers=num_workers,
                input_path=input_data.local_path(),
                output_dir=pathlib.Path(tmp),
                cache_dir=cache_dir,
                suts={sut_id: sut},
            )

            pipeline_runner.run(
                progress_callback=mlflow.log_metrics, debug=is_debug_mode()
            )
            mlflow.set_tag(MODELGAUGE_RUN_TAG_NAME, pipeline_runner.run_id)

            # log the output to mlflow's artifact store
            mlflow.log_artifact(
                local_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
            )
        return run.info.run_id
