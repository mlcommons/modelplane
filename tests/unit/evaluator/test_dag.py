"""Unit tests for EvaluatorDAG construction, validation, execution, and visualization."""

import pandas as pd
import pytest

from modelplane.evaluator.safety import SAFE, UNSAFE

from .conftest import skip_in_ci


def test_dag_outputs(simple_dag):
    assert simple_dag.outputs == [SAFE, UNSAFE]


def test_add_node_with_same_name_as_existing_node(simple_dag, always_true_gate):
    always_true_gate.name = next(iter(simple_dag._nodes))
    with pytest.raises(ValueError, match="is already registered"):
        simple_dag.add_node(always_true_gate)  # same name as existing node


def test_add_node_with_same_name_as_output(simple_dag, always_true_gate):
    always_true_gate.name = SAFE.name
    with pytest.raises(ValueError, match="is already registered"):
        simple_dag.add_node(always_true_gate)  # same name as existing output


def test_add_node_with_undefined_target_node(simple_dag, bad_gate):
    simple_dag.add_node(bad_gate)
    with pytest.raises(ValueError, match="routes to unregistered node"):
        simple_dag._validate_and_build()


def test_dag_with_cycle(bad_dag_with_cycle):
    with pytest.raises(ValueError, match="DAG contains a cycle"):
        bad_dag_with_cycle._validate_and_build()


def test_dag_with_undefined_output(bad_dag_with_undefined_output):
    with pytest.raises(ValueError, match=r"has output\(s\) that are not declared"):
        bad_dag_with_undefined_output._validate_and_build()


def test_dag_with_bad_arbiter(bad_dag_with_bad_arbiter, sample_ctx):
    with pytest.raises(
        ValueError,
        match=r"DAG execution completed without reaching an Output node",
    ):
        bad_dag_with_bad_arbiter.run(sample_ctx)


def test_dag_run(simple_dag, sample_ctx):
    result = simple_dag.run(sample_ctx)
    assert result.name == "UNSAFE"


def test_dag_run_with_dataframe(simple_dag):
    # "hello world" (space lowers avg below threshold) → safe
    # "helloworld"  (no space, avg = 0.5 = threshold)  → unsafe
    # Alternate even/odd prompt lengths to exercise both enricher paths.
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["hello world", "helloworld", "hello world", "helloworld"],
        }
    )
    result_df = simple_dag.run_dataframe(df)

    assert len(result_df) == len(df)
    assert "prompt" in result_df.columns
    assert "response" in result_df.columns
    verdicts = result_df[simple_dag.DATAFRAME_OUTPUT_COL].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts


def test_dag_run_with_dataframe_parallel(simple_dag):
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["hello world", "helloworld", "hello world", "helloworld"],
        }
    )
    result_df = simple_dag.run_dataframe(df, n_jobs=-1)

    assert len(result_df) == len(df)
    assert "prompt" in result_df.columns
    assert "response" in result_df.columns
    verdicts = result_df[simple_dag.DATAFRAME_OUTPUT_COL].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts


def test_dag_cost_one_path(simple_dag, sample_ctx):
    cost = simple_dag.total_cost(sample_ctx)
    # lower_caser and prompt_parity are at the same level from always_true
    assert cost == 0.8
    cost = simple_dag.total_cost()
    assert cost == 0.8


def test_dag_cost_all_paths(simple_dag):
    costs = simple_dag.total_costs()
    assert costs == pytest.approx(
        {
            "always_true -> always_safe -> SAFE": 1.2,
            "always_true -> lower_caser -> prompt_parity -> lower_scorer -> upper_scorer -> threshold_arbiter -> SAFE": 3.7,
            "always_true -> lower_caser -> prompt_parity -> lower_scorer -> upper_scorer -> threshold_arbiter -> UNSAFE": 3.7,
            "always_true -> lower_caser -> prompt_parity -> upper_caser -> lower_scorer -> upper_scorer -> threshold_arbiter -> SAFE": 4.2,
            "always_true -> lower_caser -> prompt_parity -> upper_caser -> lower_scorer -> upper_scorer -> threshold_arbiter -> UNSAFE": 4.2,
        }
    )


@skip_in_ci
def test_dag_visualize_runs(simple_dag, sample_ctx):
    simple_dag.visualize()
    simple_dag.visualize_run(sample_ctx)
