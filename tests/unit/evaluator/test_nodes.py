"""Unit tests for individual EvaluatorDAGNode subclasses."""

import pytest

from modelplane.evaluator.outputs import SAFE, UNSAFE

from .mocks import AlwaysTrue, AlwaysUnsafe, LowerCaser

from .conftest import DEFAULT_BRANCH, FALSE_BRANCH, SCORE1, SCORE2, TRUE_BRANCH


def test_true_routes_to_true_branch(sample_ctx, always_true_gate):
    output = always_true_gate.run(sample_ctx)
    assert output
    assert always_true_gate.next_nodes(output) == TRUE_BRANCH


def test_false_routes_to_false_branch(sample_ctx, always_false_gate):
    output = always_false_gate.run(sample_ctx)
    assert not output
    assert always_false_gate.next_nodes(output) == FALSE_BRANCH


def test_lower_caser(sample_ctx, lower_caser):
    output = lower_caser.run(sample_ctx)
    assert output == sample_ctx.response.lower()
    assert lower_caser.next_nodes(output) == DEFAULT_BRANCH


def test_fixed_scorer(sample_ctx, score_1):
    output = score_1.run(sample_ctx)
    assert output == SCORE1
    assert score_1.next_nodes(output) == DEFAULT_BRANCH


def test_consistent_arbiters(sample_ctx, score_1, score_2, always_unsafe, always_safe):
    parent_outputs = {score_1.name: SCORE1, score_2.name: SCORE2}
    sample_ctx.set_parent_outputs(parent_outputs)
    output = always_unsafe.run(sample_ctx)
    assert output.name == "UNSAFE"
    output = always_safe.run(sample_ctx)
    assert output.name == "SAFE"


def test_threshold_arbiter_true(sample_ctx, threshold_arbiter):
    sample_ctx.set_parent_outputs({"parent0": SCORE2, "parent1": SCORE2})
    output = threshold_arbiter.run(sample_ctx)
    assert output.name == "UNSAFE"


def test_threshold_arbiter_false(sample_ctx, threshold_arbiter):
    sample_ctx.set_parent_outputs({"parent0": SCORE1, "parent1": SCORE1})
    output = threshold_arbiter.run(sample_ctx)
    assert output.name == "SAFE"


def test_gate_with_two_outputs():
    with pytest.raises(ValueError, match="has multiple Output routes"):
        AlwaysTrue(
            name="bad_gate",
            routes_true=[SAFE, UNSAFE],
            routes_false=FALSE_BRANCH,
        )


def test_gate_with_no_true_route():
    with pytest.raises(ValueError, match="requires both routes_true and routes_false"):
        AlwaysTrue(
            name="bad_gate",
            routes_false=FALSE_BRANCH,
        )


def test_gate_with_routes():
    with pytest.raises(ValueError, match="should not have routes"):
        AlwaysTrue(
            name="bad_gate",
            routes_true=TRUE_BRANCH,
            routes_false=FALSE_BRANCH,
            routes=DEFAULT_BRANCH,
        )


def test_enricher_with_binary_routes():
    with pytest.raises(
        ValueError, match="should not have routes_true= / routes_false="
    ):
        LowerCaser(
            name="bad_enricher",
            routes_true=TRUE_BRANCH,
            routes=DEFAULT_BRANCH,
        )


def test_enricher_with_no_routes():
    with pytest.raises(ValueError, match="requires routes="):
        LowerCaser(
            name="bad_enricher",
        )


def test_arbiter_with_routes():
    with pytest.raises(ValueError, match="is terminal and cannot have routing kwargs"):
        AlwaysUnsafe(
            name="bad_arbiter",
            routes=DEFAULT_BRANCH,
        )
