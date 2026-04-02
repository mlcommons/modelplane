"""Unit tests for individual EvaluatorDAGNode subclasses."""

import pytest

from .conftest import DEFAULT_BRANCH, FALSE_BRANCH, SCORE1, SCORE2, TRUE_BRANCH


def test_error_getting_next_nodes_before_run(sample_ctx, lower_caser):
    with pytest.raises(
        ValueError, match="Cannot get next nodes before running the node."
    ):
        lower_caser.next_nodes()


def test_true_routes_to_true_branch(sample_ctx, always_true_gate):
    output = always_true_gate.run(sample_ctx)
    assert output
    assert always_true_gate.next_nodes() == TRUE_BRANCH


def test_false_routes_to_false_branch(sample_ctx, always_false_gate):
    output = always_false_gate.run(sample_ctx)
    assert not output
    assert always_false_gate.next_nodes() == FALSE_BRANCH


def test_output_cached(sample_ctx, lower_caser):
    output1 = lower_caser.run(sample_ctx)
    assert lower_caser._was_run
    assert lower_caser._output == output1


def test_lower_caser(sample_ctx, lower_caser):
    output = lower_caser.run(sample_ctx)
    assert output == sample_ctx.response.lower()
    assert lower_caser.next_nodes() == DEFAULT_BRANCH


def test_fixed_scorer(sample_ctx, score_1):
    output = score_1.run(sample_ctx)
    assert output == SCORE1
    assert score_1.next_nodes() == DEFAULT_BRANCH


def test_consistent_arbiters(
    sample_ctx, score_1, score_2, always_violating, always_non_violating
):
    parent_outputs = {score_1.name: SCORE1, score_2.name: SCORE2}
    sample_ctx.set_parent_outputs(parent_outputs)
    output = always_violating.run(sample_ctx)
    assert not output.is_safe()
    output = always_non_violating.run(sample_ctx)
    assert output.is_safe()


def test_threshold_arbiter_true(sample_ctx, threshold_arbiter):
    sample_ctx.set_parent_outputs({"parent0": SCORE2, "parent1": SCORE2})
    output = threshold_arbiter.run(sample_ctx)
    assert not output.is_safe()


def test_threshold_arbiter_false(sample_ctx, threshold_arbiter):
    sample_ctx.set_parent_outputs({"parent0": SCORE1, "parent1": SCORE1})
    output = threshold_arbiter.run(sample_ctx)
    assert output.is_safe()
