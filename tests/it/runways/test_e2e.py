import csv
import tempfile
import time
from typing import List

import mlflow
import mlflow.artifacts

from modelplane.runways.annotator import annotate
from modelplane.runways.responder import respond
from modelplane.runways.scorer import score
from modelplane.runways.utils import PROMPT_RESPONSE_ARTIFACT_NAME
from random_annotator import TEST_ANNOTATOR_ID


def test_e2e():
    # TODO: This should probably be split up into smaller unit tests :)
    # sut that responds based on the number of words in the prompt (even = "yes", odd = "no")
    sut_id = "demo_yes_no"
    prompts = "tests/data/prompts.csv"
    ground_truth = "tests/data/ground_truth.csv"
    experiment = "test_experiment_" + time.strftime("%Y%m%d%H%M%S", time.localtime())
    n_jobs = 1

    run_id = check_responder(
        sut_id=sut_id,
        prompts=prompts,
        experiment=experiment,
        cache_dir=None,
        n_jobs=n_jobs,
    )
    run_id = check_annotator(
        response_run_id=run_id,
        annotator_ids=[TEST_ANNOTATOR_ID],
        experiment=experiment,
        cache_dir=None,
        n_jobs=n_jobs,
    )
    check_scorer(
        annotation_run_id=run_id,
        ground_truth=ground_truth,
        annotator_id=TEST_ANNOTATOR_ID,
        experiment=experiment,
    )


def check_responder(
    sut_id: str,
    prompts: str,
    experiment: str,
    cache_dir: str | None,
    n_jobs: int,
):
    with tempfile.TemporaryDirectory() as cache_dir:
        run_id = respond(
            sut_id=sut_id,
            prompts=prompts,
            experiment=experiment,
            cache_dir=cache_dir,
            n_jobs=n_jobs,
        )

    # confirm experiment exists
    exp = mlflow.get_experiment_by_name(experiment)
    assert exp is not None
    assert run_id is not None

    # validate params / tags logged
    run = mlflow.get_run(run_id)
    params = run.data.params
    tags = run.data.tags
    assert params.get("cache_dir") == cache_dir
    assert params.get("n_jobs") == str(n_jobs)
    assert tags.get("sut_id") == sut_id

    # validate responses
    with tempfile.TemporaryDirectory() as temp_dir:
        # download/validate the prompt responses artifact
        responses_file = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=PROMPT_RESPONSE_ARTIFACT_NAME,
            dst_path=temp_dir,
        )
        with open(responses_file, "r") as f:
            reader = csv.DictReader(f)
            responses = list(reader)
            assert len(responses) == 10
            for response in responses:
                assert response["sut_uid"] == sut_id
                expected = "no" if len(response["prompt_text"].split()) % 2 else "yes"
                yesno = response["sut_response"]
                assert (
                    yesno.lower() == expected
                ), f"Unexpectedly got '{yesno} for prompt '{response['prompt_text']}'"
    return run_id


def check_annotator(
    response_run_id: str,
    annotator_ids: List[str],
    experiment: str,
    cache_dir: str | None,
    n_jobs: int,
):
    # run the annotator
    with tempfile.TemporaryDirectory() as cache_dir:
        run_id = annotate(
            response_run_id=response_run_id,
            annotator_ids=annotator_ids,
            experiment=experiment,
            cache_dir=cache_dir,
            n_jobs=n_jobs,
        )
    # confirm experiment exists
    exp = mlflow.get_experiment_by_name(experiment)
    assert exp is not None

    # validate params / tags / metrics logged
    run = mlflow.get_run(run_id)
    params = run.data.params
    tags = run.data.tags
    metrics = run.data.metrics
    assert params.get("cache_dir") == cache_dir
    assert params.get("n_jobs") == str(n_jobs)
    assert tags.get(f"annotator_{TEST_ANNOTATOR_ID}") == "true"

    # expect 5 safe (every other item)
    assert (
        metrics.get(f"{TEST_ANNOTATOR_ID}_total_count") == 10
    ), "Expected total_count to be 10"
    assert (
        metrics.get(f"{TEST_ANNOTATOR_ID}_total_safe") == 5
    ), "Expected total_safe to be 5"

    # confirm annotations.csv exists
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert any(
        artifact.path == "annotations.csv" for artifact in artifacts
    ), "Expected 'annotations.csv' artifact not found in run"
    # TODO: validate annotations.csv
    return run_id


def check_scorer(
    annotation_run_id: str,
    ground_truth: str,
    annotator_id: str,
    experiment: str,
):
    run_id = score(annotation_run_id, experiment, ground_truth)
    # confirm experiment exists
    exp = mlflow.get_experiment_by_name(experiment)
    assert exp is not None

    # validate params / metrics logged
    run = mlflow.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    assert params.get("ground_truth") == ground_truth
    assert params.get("annotation_run_id") == annotation_run_id

    assert metrics.get("num_ground_truth_samples") == 10
    assert metrics.get(f"{annotator_id}_num_annotator_samples") == 10
    assert metrics.get(f"{annotator_id}_num_samples_scored") == 10
    assert metrics.get(f"{annotator_id}_precision") == 0.0
