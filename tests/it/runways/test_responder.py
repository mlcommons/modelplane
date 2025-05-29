import csv
import tempfile

import mlflow
import mlflow.artifacts
from modelplane.runways.responder import respond
from modelplane.runways.utils import PROMPT_RESPONSE_ARTIFACT_NAME


def test_respond():
    sut_id = "demo_yes_no"
    prompts = "tests/data/prompts.csv"
    experiment = "test_experiment"
    n_jobs = 1

    with tempfile.TemporaryDirectory() as cache_dir:
        run_id = respond(
            sut_id=sut_id,
            prompts=prompts,
            experiment=experiment,
            cache_dir=cache_dir,
            n_jobs=n_jobs,
        )

    assert run_id is not None

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
                expected = "no" if len(response["Text"].split()) % 2 else "yes"
                yesno = response[sut_id]
                assert (
                    yesno.lower() == expected
                ), f"Unexpectedly got '{yesno} for prompt '{response['Text']}'"
        # validate params / tags logged
        run = mlflow.get_run(run_id)
        params = run.data.params
        tags = run.data.tags
        assert params.get("cache_dir") == cache_dir
        assert params.get("n_jobs") == str(n_jobs)
        assert tags.get("sut_id") == sut_id
