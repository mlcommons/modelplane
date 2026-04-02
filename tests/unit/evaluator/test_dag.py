"""Unit tests for EvaluatorDAG construction, validation, execution, and visualization."""


import pandas as pd


def test_dag_run(simple_dag, sample_ctx):
    result = simple_dag.run(sample_ctx)
    assert result.is_safe()
