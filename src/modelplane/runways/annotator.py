"""Runway for annotating responses from SUTs."""

import collections
import os
import pathlib
import tempfile
from typing import Any, Dict, List

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from modelgauge.annotator import Annotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.dataset import AnnotationDataset
from modelgauge.ensemble_annotator import EnsembleAnnotator
from modelgauge.ensemble_strategies import ENSEMBLE_STRATEGIES
from modelgauge.pipeline_runner import build_runner

from modelplane.mlflow.loghelpers import log_tags
from modelplane.runways.data import (
    Artifact,
    BaseInput,
    RunArtifacts,
    build_and_log_input,
)
from modelplane.runways.utils import (
    CACHE_DIR,
    MODELGAUGE_RUN_TAG_NAME,
    PROMPT_RESPONSE_ARTIFACT_NAME,
    RUN_TYPE_ANNOTATOR,
    RUN_TYPE_TAG_NAME,
    get_experiment_id,
    is_debug_mode,
    setup_annotator_credentials,
)


DEFAULT_ENSEMBLE_ANNOTATOR_UID = "ensemble"


def annotate(
    experiment: str,
    annotator_ids: List[str],
    input_object: BaseInput | None = None,
    dvc_repo: str | None = None,
    response_file: str | None = None,
    response_run_id: str | None = None,
    ensemble_strategy: str | None = None,
    overwrite: bool = False,
    disable_cache: bool = False,
    num_workers: int = 1,
    prompt_uid_col=None,
    prompt_text_col=None,
    sut_uid_col=None,
    sut_response_col=None,
) -> RunArtifacts:
    """
    Run annotations and record measurements.
    """
    # this will set annotator_ids and optionally ensemble
    pipeline_kwargs = _get_annotator_settings(annotator_ids, ensemble_strategy)
    if not disable_cache:
        pipeline_kwargs["cache_dir"] = CACHE_DIR
    pipeline_kwargs["num_workers"] = num_workers

    # set the tags
    tags = {RUN_TYPE_TAG_NAME: RUN_TYPE_ANNOTATOR}
    # tag for each annotator id to help make them searchable
    tags.update(
        {
            f"annotator_{annotator_id}": "true"
            for annotator_id in pipeline_kwargs["annotators"]
        }
    )
    if ensemble_strategy is not None:
        tags["ensemble_strategy"] = ensemble_strategy

    experiment_id = get_experiment_id(experiment)
    if overwrite and response_run_id:
        run_id = response_run_id
    else:
        run_id = None

    params = {"num_workers": num_workers}

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, tags=tags) as run:
        mlflow.log_params(params)
        if response_run_id is not None:
            log_tags(response_run_id)

        with tempfile.TemporaryDirectory() as tmp:
            # load/transform the prompt responses from the specified run
            input_data = build_and_log_input(
                input_object=input_object,
                path=response_file,
                run_id=response_run_id,
                artifact_path=PROMPT_RESPONSE_ARTIFACT_NAME,
                dvc_repo=dvc_repo,
                dest_dir=tmp,
            )
            input_path = input_data.local_path()  # type: ignore
            pipeline_kwargs["input_path"] = pathlib.Path(input_path)
            pipeline_kwargs["output_dir"] = pathlib.Path(tmp)
            pipeline_runner = build_runner(
                prompt_uid_col=prompt_uid_col,
                prompt_text_col=prompt_text_col,
                sut_uid_col=sut_uid_col,
                sut_response_col=sut_response_col,
                **pipeline_kwargs,
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

            # log summary statistics
            annotator_uids = sorted(pipeline_kwargs["annotators"].keys())
            log_safety_summary(
                annotator_uids=(
                    annotator_uids
                    if ensemble_strategy is None
                    else annotator_uids + [DEFAULT_ENSEMBLE_ANNOTATOR_UID]
                ),
                data_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
                dir=tmp,
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


def _get_annotator_settings(
    annotator_ids: List[str],
    ensemble_strategy: str | None,
) -> Dict[str, Any]:

    kwargs = {}

    kwargs["annotators"] = _get_annotators(annotator_ids)

    if ensemble_strategy is not None:
        if ensemble_strategy not in ENSEMBLE_STRATEGIES:
            raise ValueError(
                f"Unknown ensemble strategy: {ensemble_strategy}. "
                f"Available strategies: {list(ENSEMBLE_STRATEGIES.keys())}"
            )
        annotator = EnsembleAnnotator(
            uid=DEFAULT_ENSEMBLE_ANNOTATOR_UID,
            annotators=annotator_ids,
            ensemble_strategy=ensemble_strategy,
        )
        kwargs["annotators"][DEFAULT_ENSEMBLE_ANNOTATOR_UID] = annotator
    return kwargs


def _get_annotators(annotator_ids: List[str]) -> Dict[str, Annotator]:
    secrets = setup_annotator_credentials(annotator_ids)
    annotators = {}
    for annotator_id in annotator_ids:
        annotators[annotator_id] = ANNOTATORS.make_instance(
            uid=annotator_id, secrets=secrets
        )
    return annotators


def log_safety_summary(
    annotator_uids: List[str],
    data_path: str,
    dir: str,
):
    all_log_probs = collections.defaultdict(list)
    total_safe = collections.Counter()
    total = collections.Counter()

    with AnnotationDataset(data_path, "r") as dataset:
        for item in dataset:
            if "is_safe" not in item.annotation:
                continue
            is_safe = item.annotation["is_safe"]
            if "logprobs" in item.annotation:
                all_log_probs[item.annotator_uid].append(item.annotation["logprobs"])
            total[item.annotator_uid] += 1
            if is_safe:
                total_safe[item.annotator_uid] += 1
    # Log metrics.
    for annotator_uid in annotator_uids:
        mlflow.log_metric(f"{annotator_uid}_total_safe", total_safe[annotator_uid])
        mlflow.log_metric(f"{annotator_uid}_total_count", total[annotator_uid])
        # TODO: the format for the log probs isn't always the same
        # in particular, the private ensemble uses a different format
        try:
            log_stats(f"{annotator_uid}_logprobs_", all_log_probs[annotator_uid])
            log_hist(dir, f"{annotator_uid}", all_log_probs[annotator_uid])
        except Exception as e:
            print(f"Failed to log stats for {annotator_uid}: {e}")


def log_stats(tag_prefix, values):
    values = np.array(values)
    # count non-NaN values
    if np.count_nonzero(~np.isnan(values)) == 0:
        return
    mlflow.log_metric(f"{tag_prefix}mean", float(np.nanmean(values)))
    mlflow.log_metric(f"{tag_prefix}min", float(np.nanmin(values)))
    mlflow.log_metric(f"{tag_prefix}max", float(np.nanmax(values)))
    mlflow.log_metric(f"{tag_prefix}std", float(np.nanstd(values)))


def log_hist(dir, tag, values):
    plt.figure()
    plt.hist(values, bins=30)
    plt.title(f"Log-Probabilities for {tag}")
    plt.xlabel("log P(is_safe)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    filename = os.path.join(dir, f"{tag}_logprobs_hist.png")
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)
