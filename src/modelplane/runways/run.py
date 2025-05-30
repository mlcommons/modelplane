import click


from modelplane.runways.annotator import annotate
from modelplane.runways.responder import respond
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


if __name__ == "__main__":
    cli()
