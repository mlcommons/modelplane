"""Runway for getting responses from SUTs."""

import pathlib
import tempfile

import mlflow
from modelgauge.pipeline_runner import build_runner
from modelgauge.sut_factory import SUT_FACTORY

from modelplane.runways.data import (
    Artifact,
    BaseInput,
    RunArtifacts,
    build_and_log_input,
)
from modelplane.runways.utils import (
    CACHE_DIR,
    MODELGAUGE_RUN_TAG_NAME,
    RUN_TYPE_RESPONDER,
    RUN_TYPE_TAG_NAME,
    get_experiment_id,
    is_debug_mode,
    setup_sut_credentials,
)


def respond(
    sut_id: str,
    experiment: str,
    prompts: str | None = None,
    input_object: BaseInput | None = None,
    dvc_repo: str | None = None,
    disable_cache: bool = False,
    num_workers: int = 1,
    prompt_uid_col=None,
    prompt_text_col=None,
) -> RunArtifacts:
    secrets = setup_sut_credentials(sut_id)
    sut = SUT_FACTORY.make_instance(uid=sut_id, secrets=secrets)
    params = {"num_workers": num_workers}
    tags = {"sut_id": sut_id, RUN_TYPE_TAG_NAME: RUN_TYPE_RESPONDER}

    experiment_id = get_experiment_id(experiment)

    with mlflow.start_run(experiment_id=experiment_id, tags=tags) as run:
        mlflow.log_params(params)
        # Use temporary file as mlflow will log this into the artifact store
        with tempfile.TemporaryDirectory() as tmp:
            input_data = build_and_log_input(
                input_object=input_object,
                path=prompts,
                dvc_repo=dvc_repo,
                dest_dir=tmp,
            )
            pipeline_runner = build_runner(
                num_workers=num_workers,
                input_path=input_data.local_path(),
                output_dir=pathlib.Path(tmp),
                cache_dir=None if disable_cache else CACHE_DIR,
                suts={sut.uid: sut},
                prompt_uid_col=prompt_uid_col,
                prompt_text_col=prompt_text_col,
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
            artifacts = {
                input_data.local_path().name: input_data.artifact,
                pipeline_runner.output_file_name: Artifact(
                    experiment_id=run.info.experiment_id,
                    run_id=run.info.run_id,
                    name=pipeline_runner.output_file_name,
                ),
            }

        return RunArtifacts(run_id=run.info.run_id, artifacts=artifacts)
