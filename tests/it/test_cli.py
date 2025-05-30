from click.testing import CliRunner

from modelplane.runways.run import cli


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


def test_get_sut_responses_help():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "get-sut-responses",
            "--help",
        ],
    )
    assert result.exit_code == 0


def test_annotate_help():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "annotate",
            "--help",
        ],
    )
    assert result.exit_code == 0
