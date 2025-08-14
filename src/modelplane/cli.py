from typing import List

import click

from modelgauge.data_schema import DEFAULT_ANNOTATION_SCHEMA as ANNOTATION_SCHEMA
from modelgauge.ensemble_annotator_set import ENSEMBLE_STRATEGIES

from modelplane.runways.annotator import annotate, KNOWN_ENSEMBLES
from modelplane.runways.lister import (
    list_annotators,
    list_ensemble_strategies,
    list_suts,
)
from modelplane.runways.responder import respond
from modelplane.runways.scorer import score
from modelplane.utils.env import load_from_dotenv


@click.group(name="modelplane")
def cli():
    pass


@cli.command(name="list-annotators", help="List known annotators.")
def list_annotators_cli():
    list_annotators()


@cli.command(name="list-ensemble-strategies", help="List known ensemble strategies.")
def list_ensemble_strategies_cli():
    list_ensemble_strategies()


@cli.command(name="list-suts", help="List known suts.")
def list_suts_cli():
    list_suts()


@cli.command(name="get-sut-responses")
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
    "--dvc_repo",
    type=str,
    required=False,
    help="URL of the DVC repo to get the prompts from. E.g. https://github.com/my-org/my-repo.git. Can specify the revision using the `#` suffix, e.g. https://github.com/my-org/my-repo.git#main.",
)
@click.option(
    "--disable_cache",
    is_flag=True,
    default=False,
    help="Disable caching of LLM responses. If set, the pipeline will not cache SUT/annotator responses. Otherwise, cached responses will be stored locally in `.cache`.",
)
@click.option(
    "--num_workers",
    type=int,
    default=1,
    help="The number of workers to run in parallel. Defaults to 1.",
)
@click.option(
    "--prompt_uid_col",
    type=str,
    required=False,
    help="The name of the prompt UID column in the dataset.",
)
@click.option(
    "--prompt_text_col",
    type=str,
    required=False,
    help="The name of the prompt text column in the dataset.",
)
@load_from_dotenv
def get_sut_responses(
    sut_id: str,
    prompts: str,
    experiment: str,
    dvc_repo: str | None = None,
    disable_cache: bool = False,
    num_workers: int = 1,
    prompt_uid_col: str | None = None,
    prompt_text_col: str | None = None,
):
    """
    Run the pipeline to get responses from SUTs.
    """
    return respond(
        sut_id=sut_id,
        prompts=prompts,
        experiment=experiment,
        dvc_repo=dvc_repo,
        disable_cache=disable_cache,
        num_workers=num_workers,
        prompt_uid_col=prompt_uid_col,
        prompt_text_col=prompt_text_col,
    )


@cli.command(name="annotate")
@click.option(
    "--experiment",
    type=str,
    required=True,
    default=None,
    help="The experiment name to use. If the experiment does not exist, it will be created.",
)
@click.option(
    "--dvc_repo",
    type=str,
    required=False,
    help="URL of the DVC repo to get the responses file from. E.g. https://github.com/my-org/my-repo.git",
)
@click.option(
    "--response_file",
    type=str,
    required=False,
    default=None,
    help="The response file to annotate.",
)
@click.option(
    "--response_run_id",
    type=str,
    required=False,
    help="The run ID corresponding to the responses to annotate.",
)
@click.option(
    "--annotator_id",
    type=str,
    multiple=True,
    default=None,
    help="The annotator UID(s) to use. Multiple annotators can be specified.",
)
@click.option(
    "--ensemble_strategy",
    type=str,
    default=None,
    help="The ensemble strategy to use. If set, individual annotator results will be combined using the given strategy. "
    "Available strategies: " + ", ".join(list(ENSEMBLE_STRATEGIES.keys())),
)
@click.option(
    "--ensemble_id",
    type=str,
    default=None,
    help="Use a fixed ensemble id to use a predefined ensemble strategy. Options include: "
    + ", ".join(list(KNOWN_ENSEMBLES.keys())),
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Use the response_run_id to save annotation artifact. Any existing annotation artifact will be overwritten. If not set, a new run will be created. Only applies if not using response_run_file.",
)
@click.option(
    "--disable_cache",
    is_flag=True,
    default=False,
    help="Disable caching of LLM responses. If set, the pipeline will not cache SUT/annotator responses. Otherwise, cached responses will be stored locally in `.cache`.",
)
@click.option(
    "--num_workers",
    type=int,
    default=1,
    help="The number of workers to run in parallel. Defaults to 1.",
)
@click.option(
    "--prompt_uid_col",
    type=str,
    required=False,
    help="The name of the prompt UID column in the dataset.",
)
@click.option(
    "--prompt_text_col",
    type=str,
    required=False,
    help="The name of the prompt text column in the dataset.",
)
@click.option(
    "--sut_uid_col",
    type=str,
    required=False,
    help="The name of the SUT UID column in the dataset.",
)
@click.option(
    "--sut_response_col",
    type=str,
    required=False,
    help="The name of the SUT response column in the dataset.",
)
@load_from_dotenv
def get_annotations(
    experiment: str,
    dvc_repo: str | None = None,
    response_file: str | None = None,
    response_run_id: str | None = None,
    annotator_id: List[str] | None = None,
    ensemble_strategy: str | None = None,
    ensemble_id: str | None = None,
    overwrite: bool = False,
    disable_cache: bool = False,
    num_workers: int = 1,
    prompt_uid_col: str | None = None,
    prompt_text_col: str | None = None,
    sut_uid_col: str | None = None,
    sut_response_col: str | None = None,
):
    return annotate(
        experiment=experiment,
        dvc_repo=dvc_repo,
        response_file=response_file,
        response_run_id=response_run_id,
        annotator_ids=annotator_id,
        ensemble_strategy=ensemble_strategy,
        ensemble_id=ensemble_id,
        overwrite=overwrite,
        disable_cache=disable_cache,
        num_workers=num_workers,
        prompt_uid_col=prompt_uid_col,
        prompt_text_col=prompt_text_col,
        sut_uid_col=sut_uid_col,
        sut_response_col=sut_response_col,
    )


@cli.command(name="score")
@click.option(
    "--experiment",
    type=str,
    required=True,
    help="The experiment name to use. If the experiment does not exist, it will be created.",
)
@click.option(
    "--annotation_run_id",
    type=str,
    required=True,
    help="The run ID corresponding to the annotations to score.",
)
@click.option(
    "--ground_truth",
    type=str,  # TODO: Pathlib
    help="Path to the ground truth file.",
)
@click.option(
    "--dvc_repo",
    type=str,
    required=False,
    help="URL of the DVC repo to get the ground truth from. E.g. https://github.com/my-org/my-repo.git",
)
@click.option(
    "--sample_uid_col",
    type=str,
    required=False,
    help="The name of the sample uid columns in the annotations and ground truth files. prompt_uid x sut_uid will be used by default.",
)
@click.option(
    "--annotator_uid_col",
    type=str,
    required=False,
    help="The name of the annotator UID column in the annotations file.",
)
@click.option(
    "--annotation_col",
    type=str,
    required=False,
    help="The name of the JSON annotation column in the annotations file.",
)
@load_from_dotenv
def score_annotations(
    experiment: str,
    annotation_run_id: str,
    ground_truth: str,
    dvc_repo: str | None = None,
    sample_uid_col: str | None = None,
    annotator_uid_col: str = ANNOTATION_SCHEMA.annotator_uid,
    annotation_col: str = ANNOTATION_SCHEMA.annotation,
):
    return score(
        annotation_run_id=annotation_run_id,
        experiment=experiment,
        ground_truth=ground_truth,
        dvc_repo=dvc_repo,
        sample_uid_col=sample_uid_col,
        annotator_uid_col=annotator_uid_col,
        annotation_col=annotation_col,
    )


if __name__ == "__main__":
    cli()
