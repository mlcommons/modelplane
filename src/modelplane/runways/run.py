from typing import List

import click
from modelgauge.ensemble_annotator_set import ENSEMBLE_STRATEGIES

from modelplane.runways.annotator import annotate, KNOWN_ENSEMBLES
from modelplane.runways.responder import respond
from modelplane.runways.scorer import score
from modelplane.utils.env import load_from_dotenv


@click.group(name="modelplane")
def cli():
    pass


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
    help="URL of the DVC repo to get the prompts from. E.g. https://github.com/my-org/my-repo.git",
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
def get_sut_responses(
    sut_id: str,
    prompts: str,
    experiment: str,
    dvc_repo: str | None = None,
    cache_dir: str | None = None,
    n_jobs: int = 1,
):
    """
    Run the pipeline to get responses from SUTs.
    """
    return respond(
        sut_id=sut_id,
        prompts=prompts,
        experiment=experiment,
        dvc_repo=dvc_repo,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
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
    experiment: str,
    dvc_repo: str | None = None,
    response_file: str | None = None,
    response_run_id: str | None = None,
    annotator_id: List[str] | None = None,
    ensemble_strategy: str | None = None,
    ensemble_id: str | None = None,
    overwrite: bool = False,
    cache_dir: str | None = None,
    n_jobs: int = 1,
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
        cache_dir=cache_dir,
        n_jobs=n_jobs,
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
@load_from_dotenv
def score_annotations(
    experiment: str,
    annotation_run_id: str,
    ground_truth: str,
    dvc_repo: str | None = None,
):
    return score(
        annotation_run_id=annotation_run_id,
        experiment=experiment,
        ground_truth=ground_truth,
        dvc_repo=dvc_repo,
    )


if __name__ == "__main__":
    cli()
