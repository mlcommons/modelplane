"""Unit tests for EvaluatorDAG construction, validation, execution, and visualization."""

import pprint
from unittest.mock import patch

import pandas as pd
import pytest

from modelplane.evaluator.cost import CostInfo, RealizedCost
from modelplane.evaluator.dag import EvaluatorDAG
from modelplane.evaluator.safety import Safety

from .conftest import skip_in_ci


def test_dag_outputs(simple_dag):
    assert simple_dag.output_type == Safety


def test_dag_with_bad_output_type():
    with pytest.raises(
        ValueError,
        match="output_type must be a subclass of Output",
    ):
        EvaluatorDAG(name="bad_dag", output_type=str)


def test_add_node_with_same_name_as_existing_node(simple_dag, always_true_gate):
    always_true_gate.name = next(iter(simple_dag._nodes))
    with pytest.raises(ValueError, match="is already registered"):
        simple_dag.add_node(always_true_gate)  # same name as existing node


def test_add_node_with_undefined_target_node(simple_dag, bad_gate):
    simple_dag.add_node(bad_gate)
    with pytest.raises(ValueError, match="routes to unregistered node"):
        simple_dag._validate_and_build()


def test_dag_with_cycle(bad_dag_with_cycle):
    with pytest.raises(ValueError, match="DAG contains a cycle"):
        bad_dag_with_cycle._validate_and_build()


def test_dag_with_undefined_output(bad_dag_with_undefined_output):
    with pytest.raises(
        ValueError, match=r"which is not compatible with the DAG\'s output_type"
    ):
        bad_dag_with_undefined_output._validate_and_build()


def test_dag_with_bad_arbiter(bad_dag_with_bad_arbiter, sample_ctx):
    with pytest.raises(
        ValueError,
        match=r"DAG execution completed without reaching an Output node",
    ):
        bad_dag_with_bad_arbiter.run(sample_ctx)


def test_dag_with_bad_output_route(bad_one_step_dag, sample_ctx):
    with pytest.raises(
        ValueError,
        match=r"incompatible output",
    ):
        bad_one_step_dag.run(sample_ctx)


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
    verdicts = result_df[simple_dag.dataframe_output_col].tolist()
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
    verdicts = result_df[simple_dag.dataframe_output_col].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts


def test_dag_realized_cost(simple_dag, sample_ctx):
    cost = simple_dag.realized_costs(sample_ctx)
    # lower_caser and prompt_parity are at the same level from always_true
    assert cost.input_token_cost == pytest.approx(
        1.2
    )  # (lower_caser - 4 * 0.3) + (prompt_parity - 4 * 0)
    assert cost.output_token_cost == pytest.approx(
        1.6
    )  # (lower_caser - 4 * 0.4) + (prompt_parity - 4 * 0)
    assert cost.fixed_cost == pytest.approx(
        0.5
    )  # (lower_caser - 0.3) + (prompt_parity - 0.2)
    assert cost.latency_seconds == pytest.approx(
        0.5
    )  # (lower_caser - 0.3) + (prompt_parity - 0.2)
    assert cost.total_token_cost == pytest.approx(2.8)
    assert cost.total_cost == pytest.approx(3.3)


def test_dag_cost_all_paths(simple_dag):
    costs = simple_dag.potential_costs()
    assert costs.keys() == {
        "always_true -> always_safe -> Out (Safety)",
        "always_true -> lower_caser -> prompt_parity -> lower_scorer -> upper_scorer -> threshold_arbiter -> Out (Safety)",
        "always_true -> lower_caser -> prompt_parity -> upper_caser -> lower_scorer -> upper_scorer -> threshold_arbiter -> Out (Safety)",
    }

    key = "always_true -> always_safe -> Out (Safety)"
    assert costs[key].input_cost_per_token == pytest.approx(0.0)
    assert costs[key].output_cost_per_token == pytest.approx(0.0)
    assert costs[key].fixed_cost == pytest.approx(1.0)
    assert costs[key].latency_seconds == pytest.approx(1.0)

    key = "always_true -> lower_caser -> prompt_parity -> lower_scorer -> upper_scorer -> threshold_arbiter -> Out (Safety)"
    # 0.3 (lower_caser) + 0.0 (prompt_parity) + 0.0 (lower_scorer) + 0.0 (upper_scorer) + 0.0 (threshold_arbiter)
    assert costs[key].input_cost_per_token == pytest.approx(0.3)
    # 0.4 (lower_caser) + 0.0 (prompt_parity) + 0.0 (lower_scorer) + 0.0 (upper_scorer) + 0.0 (threshold_arbiter)
    assert costs[key].output_cost_per_token == pytest.approx(0.4)
    # 0.3 (lower_caser) + 0.2 (prompt_parity) + 0.7 (lower_scorer) + 0.8 (upper_scorer) + 1.1 (threshold_arbiter)
    assert costs[key].fixed_cost == pytest.approx(3.1)
    # 0.3 (lower_caser) + 0.2 (prompt_parity) + 0.7 (lower_scorer) + 0.8 (upper_scorer) + 1.1 (threshold_arbiter)
    assert costs[key].latency_seconds == pytest.approx(3.1)

    key = "always_true -> lower_caser -> prompt_parity -> upper_caser -> lower_scorer -> upper_scorer -> threshold_arbiter -> Out (Safety)"
    # above + 0.4 (upper_caser)
    assert costs[key].input_cost_per_token == pytest.approx(0.7)
    # above + 0.5 (upper_caser)
    assert costs[key].output_cost_per_token == pytest.approx(0.9)
    # above + 0.4 (upper_caser)
    assert costs[key].fixed_cost == pytest.approx(3.5)
    # above + 0.4 (upper_caser)
    assert costs[key].latency_seconds == pytest.approx(3.5)


@skip_in_ci
def test_dag_visualize_runs(simple_dag, one_step_dag, sample_ctx):
    simple_dag.visualize()
    simple_dag.visualize_run(sample_ctx)
    one_step_dag.visualize()
    one_step_dag.visualize_run(sample_ctx)


def test_visualize_raises_when_graphviz_binary_missing(simple_dag):
    import graphviz

    with patch.object(
        graphviz.Digraph,
        "pipe",
        side_effect=graphviz.ExecutableNotFound(["dot"]),
    ):
        with pytest.raises(
            RuntimeError,
            match="Graphviz system binaries not found",
        ):
            simple_dag.visualize()
