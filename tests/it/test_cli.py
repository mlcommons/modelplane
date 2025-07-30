import pytest
from click.testing import CliRunner

from modelplane.cli import cli


def test_main_help():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--help",
        ],
    )
    assert result.exit_code == 0
    assert "get-sut-responses" in result.output
    assert "annotate" in result.output


@pytest.mark.parametrize(
    "command",
    [
        "get-sut-responses",
        "annotate",
        "score",
        "list-suts",
        "list-annotators",
        "list-ensemble-strategies",
    ],
)
def test_command_help(command):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            command,
            "--help",
        ],
    )
    assert result.exit_code == 0
