from modelplane.runways.lister import (
    list_annotators,
    list_ensemble_strategies,
    list_suts,
)
from modelgauge.ensemble_annotator_set import ENSEMBLE_STRATEGIES


def test_list_annotators(capsys):
    list_annotators()
    output = capsys.readouterr().out.strip()
    assert "demo_annotator" in output


def test_list_ensemble_strategies(capsys):
    ENSEMBLE_STRATEGIES["demo_ensemble_strategy"] = "Demo Ensemble Strategy"
    list_ensemble_strategies()
    output = capsys.readouterr().out.strip()
    assert "demo_ensemble_strategy" in output

    del ENSEMBLE_STRATEGIES["demo_ensemble_strategy"]  # Clean up after test


def test_list_suts(capsys):
    list_suts()
    output = capsys.readouterr().out.strip()
    assert "demo_yes_no" in output
