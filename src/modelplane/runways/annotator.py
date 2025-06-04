"""Runway for annotating responses from SUTs.

TODO: PROMPT_CSV_INPUT_COLUMNS / ANNOTATOR_CSV_INPUT_COLUMNS should be aligned
TODO: log_safety_summary is probably too specific to llamaguard
"""

import collections
import csv
import os
import pathlib
import tempfile
from typing import List

import jsonlines
import mlflow
import mlflow.artifacts
import numpy as np
from matplotlib import pyplot as plt

from modelgauge.annotation_pipeline import ANNOTATOR_CSV_INPUT_COLUMNS
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.pipeline_runner import AnnotatorRunner

from modelplane.runways.utils import (
    PROMPT_RESPONSE_ARTIFACT_NAME,
    get_experiment_id,
    is_debug_mode,
    setup_annotator_credentials,
)


def annotate(
    annotator_ids: List[str],
    experiment: str,
    response_run_id: str,
    overwrite: bool = False,
    cache_dir: str | None = None,
    n_jobs: int = 1,
) -> str:
    """
    Run annotations and record measurements.
    """
    secrets = setup_annotator_credentials(annotator_ids)
    annotators = {}
    for annotator_id in annotator_ids:
        annotators[annotator_id] = ANNOTATORS.make_instance(
            uid=annotator_id, secrets=secrets
        )
    params = {
        "cache_dir": cache_dir,
        "n_jobs": n_jobs,
    }
    # tag for each annotator id to help make them searchable
    tags = {f"annotator_{annotator_id}": "true" for annotator_id in annotator_ids}

    experiment_id = get_experiment_id(experiment)
    if overwrite:
        run_id = response_run_id
    else:
        run_id = None

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, tags=tags):
        mlflow.log_params(params)

        with tempfile.TemporaryDirectory() as tmp:
            # load/transform the prompt responses from the specified run
            input_path = transform_mlflow_responder_artifact(
                run_id=response_run_id, dir=tmp
            )
            pipeline_runner = AnnotatorRunner(
                annotators=annotators,
                num_workers=n_jobs,
                input_path=pathlib.Path(input_path),
                output_dir=pathlib.Path(tmp),
                cache_dir=cache_dir,
            )

            pipeline_runner.run(
                progress_callback=mlflow.log_metrics, debug=is_debug_mode()
            )

            # log the output to mlflow's artifact store
            mlflow.log_artifact(
                local_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
            )

            # log summary statistics
            log_safety_summary(
                annotator_uids=annotator_ids,
                data_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
                dir=tmp,
            )
        return mlflow.active_run().info.run_id  # type: ignore


def transform_mlflow_responder_artifact(run_id: str, dir: str) -> str:
    transformed_suffix = "transformed.csv"
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=PROMPT_RESPONSE_ARTIFACT_NAME,
        dst_path=dir,
    )
    # rename columns
    with open(os.path.join(dir, PROMPT_RESPONSE_ARTIFACT_NAME), "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if (
            fieldnames is None
            or len(fieldnames) < 3
            or fieldnames[0] != "UID"
            or fieldnames[1] != "Text"
        ):
            raise ValueError(
                "Invalid input file. Must have columns: UID, Text, sut_id1, sut_id1."
            )
        suts = fieldnames[2:]
        with open(os.path.join(dir, transformed_suffix), "w") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=ANNOTATOR_CSV_INPUT_COLUMNS)
            writer.writeheader()
            for row in reader:
                for sut in suts:
                    writer.writerow(
                        {
                            "UID": row["UID"],
                            "Prompt": row["Text"],
                            "SUT": sut,
                            "Response": row[sut],
                        }
                    )
    return os.path.join(dir, transformed_suffix)


def log_safety_summary(
    annotator_uids: List[str],
    data_path: str,
    dir: str,
):
    all_log_probs = collections.defaultdict(list)
    total_safe = collections.Counter()
    total = collections.Counter()

    with open(data_path, "r") as f:
        reader = jsonlines.Reader(f)
        for row in reader:
            for annotator_uid in annotator_uids:
                if (
                    annotator_uid not in row["Annotations"]
                    or "is_safe" not in row["Annotations"][annotator_uid]
                ):
                    continue
                is_safe = row["Annotations"][annotator_uid]["is_safe"]
                if "is_safe_logprob" in row["Annotations"][annotator_uid]:
                    all_log_probs[annotator_uid].append(
                        row["Annotations"][annotator_uid]["is_safe_logprob"]
                    )

                total[annotator_uid] += 1
                if is_safe:
                    total_safe[annotator_uid] += 1

        for annotator_uid in annotator_uids:
            mlflow.log_metric(f"{annotator_uid}_total_safe", total_safe[annotator_uid])
            mlflow.log_metric(f"{annotator_uid}_total_count", total[annotator_uid])
            log_stats(f"{annotator_uid}_log_prob_", all_log_probs[annotator_uid])
            log_hist(dir, f"{annotator_uid}", all_log_probs[annotator_uid])


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
    filename = os.path.join(dir, f"{tag}_log_prob_hist.png")
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)
