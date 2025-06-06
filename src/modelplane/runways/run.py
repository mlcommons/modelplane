from typing import List
import click


from modelplane.runways.annotator import annotate
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
        cache_dir=cache_dir,
        n_jobs=n_jobs,
    )


@cli.command(name="annotate")
@click.option(
    "--annotator_id",
    type=str,
    multiple=True,
    required=True,
    help="The annotator UID(s) to use. Multiple annotators can be specified.",
)
@click.option(
    "--experiment",
    type=str,
    required=True,
    default=None,
    help="The experiment name to use. If the experiment does not exist, it will be created.",
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
    annotator_id: List[str],
    experiment: str,
    response_file: str | None = None,
    response_run_id: str | None = None,
    overwrite: bool = False,
    cache_dir: str | None = None,
    n_jobs: int = 1,
):
    return annotate(
        annotator_ids=annotator_id,
        experiment=experiment,
        response_file=response_file,
        response_run_id=response_run_id,
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
@load_from_dotenv
def score_annotations(
    experiment: str,
    annotation_run_id: str,
    ground_truth: str,
):
    return score(
        annotation_run_id=annotation_run_id,
        experiment=experiment,
        ground_truth=ground_truth,
    )


if __name__ == "__main__":
    cli()
