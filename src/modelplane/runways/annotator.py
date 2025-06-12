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
from modelgauge.ensemble_annotator_set import EnsembleAnnotatorSet, ENSEMBLE_STRATEGIES
from modelgauge.pipeline_runner import build_runner

from modelplane.runways.utils import (
    PROMPT_RESPONSE_ARTIFACT_NAME,
    get_experiment_id,
    is_debug_mode,
    setup_annotator_credentials,
)


def annotate(
    annotator_ids: List[str],
    experiment: str,
    response_file: str | None = None,
    response_run_id: str | None = None,
    ensemble_strategy: str | None = None,
    overwrite: bool = False,
    cache_dir: str | None = None,
    n_jobs: int = 1,
) -> str:
    """
    Run annotations and record measurements.
    """
    if not ((response_file is None) ^ (response_run_id is None)):
        raise ValueError(
            "Exactly one of response_file or response_run_id must be provided."
        )

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
    if overwrite and response_run_id:
        run_id = response_run_id
    else:
        run_id = None

    kwargs = {
        "annotators": annotators,
        "num_workers": n_jobs,
        "cache_dir": cache_dir,
    }
    if ensemble_strategy is not None:
        if ensemble_strategy not in ENSEMBLE_STRATEGIES:
            raise ValueError(
                f"Unknown ensemble strategy: {ensemble_strategy}. "
                f"Available strategies: {list(ENSEMBLE_STRATEGIES.keys())}"
            )
        tags["ensemble_strategy"] = ensemble_strategy
        kwargs["ensemble"] = EnsembleAnnotatorSet(
            annotators=annotator_ids,
            strategy=ENSEMBLE_STRATEGIES[ensemble_strategy],
        )

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, tags=tags):
        mlflow.log_params(params)

        with tempfile.TemporaryDirectory() as tmp:
            # load/transform the prompt responses from the specified run
            if response_run_id:
                mlflow.artifacts.download_artifacts(
                    run_id=response_run_id,
                    artifact_path=PROMPT_RESPONSE_ARTIFACT_NAME,
                    dst_path=tmp,
                )
                raw_path = os.path.join(tmp, PROMPT_RESPONSE_ARTIFACT_NAME)
            else:
                raw_path = response_file
            input_path = transform_annotation_file(src=raw_path, dest_dir=tmp)
            kwargs["input_path"] = pathlib.Path(input_path)
            kwargs["output_dir"] = pathlib.Path(tmp)
            pipeline_runner = build_runner(**kwargs)

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
                annotator_uids=(
                    annotator_ids
                    if ensemble_strategy is None
                    else annotator_ids + ["ensemble"]
                ),
                data_path=pipeline_runner.output_dir()
                / pipeline_runner.output_file_name,
                dir=tmp,
            )
        return mlflow.active_run().info.run_id  # type: ignore


def transform_annotation_file(src: str, dest_dir: str) -> str:
    transformed_suffix = "transformed.csv"
    # rename columns
    with open(src, "r") as f:
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
        with open(os.path.join(dest_dir, transformed_suffix), "w") as f_out:
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
    return os.path.join(dest_dir, transformed_suffix)


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
                if "logprobs" in row["Annotations"][annotator_uid]:
                    all_log_probs[annotator_uid].append(
                        row["Annotations"][annotator_uid]["logprobs"]
                    )

                total[annotator_uid] += 1
                if is_safe:
                    total_safe[annotator_uid] += 1

        for annotator_uid in annotator_uids:
            mlflow.log_metric(f"{annotator_uid}_total_safe", total_safe[annotator_uid])
            mlflow.log_metric(f"{annotator_uid}_total_count", total[annotator_uid])
            log_stats(f"{annotator_uid}_logprobs_", all_log_probs[annotator_uid])
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
    filename = os.path.join(dir, f"{tag}_logprobs_hist.png")
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)
