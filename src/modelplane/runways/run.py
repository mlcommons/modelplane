import click

from modelplane.runways.annotator import get_annotations
from modelplane.runways.responder import get_sut_responses


@click.group(name="modelplane")
def cli():
    pass


cli.add_command(get_sut_responses)
cli.add_command(get_annotations)


if __name__ == "__main__":
    cli()
