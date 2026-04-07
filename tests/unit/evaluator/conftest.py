"""Shared mock node implementations and helpers for evaluator tests."""

import pytest

from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.dag import EvaluatorDAG
from modelplane.evaluator.outputs import SAFE, UNSAFE, Output

from .mocks import (
    AlwaysFalse,
    AlwaysSafe,
    AlwaysTrue,
    AlwaysUnsafe,
    FixedScorer,
    LLMEnricher,
    LowerCaser,
    LowerCaseScorer,
    PromptLengthGate,
    ThresholdArbiter,
    UpperCaser,
    UpperCaseScorer,
)

TRUE_BRANCH: tuple[str | Output] = ("true_branch",)
FALSE_BRANCH: tuple[str | Output] = ("false_branch",)
DEFAULT_BRANCH: tuple[str | Output] = ("next_node",)
SCORE1 = 1.0
SCORE2 = 2.0


@pytest.fixture
def always_true_gate() -> AlwaysTrue:
    return AlwaysTrue(
        name="always_true", routes_true=TRUE_BRANCH, routes_false=FALSE_BRANCH
    )


@pytest.fixture
def always_false_gate() -> AlwaysFalse:
    return AlwaysFalse(
        name="always_false", routes_true=TRUE_BRANCH, routes_false=FALSE_BRANCH
    )


@pytest.fixture
def lower_caser() -> LowerCaser:
    return LowerCaser(name="lower_caser", routes=DEFAULT_BRANCH)


@pytest.fixture
def score_1() -> FixedScorer:
    return FixedScorer(name="score_1", value=SCORE1, routes=DEFAULT_BRANCH)


@pytest.fixture
def score_2() -> FixedScorer:
    return FixedScorer(name="score_2", value=SCORE2, routes=DEFAULT_BRANCH)


@pytest.fixture
def costly_enricher() -> LLMEnricher:
    return LLMEnricher(name="costly_enricher", routes=DEFAULT_BRANCH)


@pytest.fixture
def sample_ctx() -> EvalContext:
    return EvalContext(prompt="Hello, world!", response="This is a response.")


@pytest.fixture
def always_unsafe() -> AlwaysUnsafe:
    return AlwaysUnsafe(name="always_unsafe")


@pytest.fixture
def always_safe() -> AlwaysSafe:
    return AlwaysSafe(name="always_safe")


@pytest.fixture
def threshold_arbiter() -> ThresholdArbiter:
    return ThresholdArbiter(name="threshold_arbiter", threshold=1.5)


@pytest.fixture
def simple_dag():
    return (
        EvaluatorDAG("simple", outputs=[SAFE, UNSAFE])
        .add_node(
            PromptLengthGate(
                name="prompt_parity",
                routes_true=["lower_caser"],
                routes_false=["upper_caser"],
            )
        )
        .add_node(
            LowerCaser(name="lower_caser", routes=["lower_scorer", "upper_scorer"])
        )
        .add_node(
            UpperCaser(name="upper_caser", routes=["lower_scorer", "upper_scorer"])
        )
        .add_node(LowerCaseScorer(name="lower_scorer", routes=["threshold_arbiter"]))
        .add_node(UpperCaseScorer(name="upper_scorer", routes=["threshold_arbiter"]))
        .add_node(ThresholdArbiter(name="threshold_arbiter", threshold=0.5))
    )
