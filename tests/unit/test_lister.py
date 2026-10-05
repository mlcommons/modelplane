from modelplane.runways.lister import (
    list_annotators,
    list_suts,
)


def test_list_annotators(capsys):
    list_annotators()
    output = capsys.readouterr().out.strip()
    assert "demo_annotator" in output


def test_list_suts(capsys):
    list_suts()
    output = capsys.readouterr().out.strip()
    assert "demo_yes_no" in output
