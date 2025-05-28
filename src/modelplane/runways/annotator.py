"""Runway for annotating responses from SUTs.

TODO: PROMPT_CSV_INPUT_COLUMNS / ANNOTATOR_CSV_INPUT_COLUMNS should be aligned
TODO: log_safety_summary is probably too specific to llamaguard
"""

import csv
import os
import pathlib
import tempfile
from collections import defaultdict

import click
import jsonlines
import mlflow
import mlflow.artifacts
import numpy as np
from matplotlib import pyplot as plt

from modelgauge.annotation_pipeline import ANNOTATOR_CSV_INPUT_COLUMNS
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.load_plugins import load_plugins
from modelgauge.pipeline_runner import AnnotatorRunner

from modelplane.runways.utils import (
    PROMPT_RESPONSE_ARTIFACT_NAME,
    get_experiment_id,
    is_debug_mode,
    setup_annotator_credentials,
)
from modelplane.utils.env import load_from_dotenv

load_plugins(disable_progress_bar=True)


@click.command(name="annotate")
@click.option(
    "--annotator_id",
    type=str,
    required=True,
    help="The SUT UID to use.",
)
@click.option(
    "--experiment",
    type=str,
    required=True,
    help="The experiment name to use. If the experiment does not exist, it will be created.",
)
@click.option(
    "--response_run_id",
    type=str,
    required=True,
    help="The run ID corresponding to the responses to annotate.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Use the response_run_id to save annotation artifact. Any existing annotation artifact will be overwritten. If not set, a new run will be created.",
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
def get_annotations(
    annotator_id: str,
    experiment: str,
    response_run_id: str,
    overwrite: bool = False,
    cache_dir: str | None = None,
    n_jobs: int = 1,
):
    return annotate(
        annotator_id=annotator_id,
        experiment=experiment,
        response_run_id=response_run_id,
        overwrite=overwrite,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
    )


def annotate(
    annotator_id: str,
    experiment: str,
    response_run_id: str,
    overwrite: bool = False,
    cache_dir: str | None = None,
    n_jobs: int = 1,
) -> None:
    """
    Run annotations and record measurements.
    """

    secrets = setup_annotator_credentials(annotator_id)
    annotator = ANNOTATORS.make_instance(uid=annotator_id, secrets=secrets)
    params = {
        "cache_dir": cache_dir,
        "n_jobs": n_jobs,
    }
    tags = {"annotator_id": annotator_id}

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
                annotators={annotator_id: annotator},
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
                annotator_uid=annotator_id,
                data_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
                dir=tmp,
            )


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
    annotator_uid: str,
    data_path: str,
    dir: str,
):
    log_probs_by_cat = defaultdict(list)
    safe_counts_by_cat = defaultdict(int)
    total_counts_by_cat = defaultdict(int)

    all_log_probs = []
    total_safe = 0
    total = 0

    with open(data_path, "r") as f:
        reader = jsonlines.Reader(f)
        for row in reader:
            is_safe = row["Annotations"][annotator_uid]["is_safe"]
            unsafe_categories = row["Annotations"][annotator_uid][
                "violation_categories"
            ]
            log_prob = row["Annotations"][annotator_uid]["is_safe_logprob"]

            for category in unsafe_categories:
                log_probs_by_cat[category].append(log_prob)
                total_counts_by_cat[category] += 1

                if is_safe:
                    safe_counts_by_cat[category] += 1

            total += 1
            all_log_probs.append(log_prob)
            if is_safe:
                total_safe += 1

        # per-category
        for category, probs in log_probs_by_cat.items():
            tag = f"{category}_log_prob_"
            log_stats(tag, probs)

            mlflow.log_metric(f"{category}_safe_count", safe_counts_by_cat[category])
            mlflow.log_metric(f"{category}_total_count", total_counts_by_cat[category])

            log_hist(dir, category, probs)

        # Overall stats
        log_stats(f"overall_log_prob_", all_log_probs)
        mlflow.log_metric(f"total_safe", total_safe)
        mlflow.log_metric(f"total_count", total)

        log_hist(dir, "_overall", all_log_probs)


def log_stats(tag_prefix, values):
    if len(values) == 0:
        return
    values = np.array(values)
    mlflow.log_metric(f"{tag_prefix}mean", float(np.mean(values)))
    mlflow.log_metric(f"{tag_prefix}min", float(np.min(values)))
    mlflow.log_metric(f"{tag_prefix}max", float(np.max(values)))
    mlflow.log_metric(f"{tag_prefix}std", float(np.std(values)))


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
