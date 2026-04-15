from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.cost import CostInfo
from modelplane.evaluator.nodes import Arbiter, Enricher, Gate
from modelplane.evaluator.outputs import Output
from modelplane.evaluator.safety import Safety


def context_token_count(ctx: EvalContext) -> int:
    return len(ctx.prompt.split() + ctx.response.split())


class PassthroughGate(Gate):
    ROUTE_TO_TAKE: bool

    def run(self, ctx: EvalContext) -> bool:
        return self.ROUTE_TO_TAKE

    def input_tokens(self, ctx: EvalContext) -> int:
        return context_token_count(ctx)

    def output_tokens(self, ctx: EvalContext) -> int:
        return 1


class AlwaysTrue(PassthroughGate):
    ROUTE_TO_TAKE = True


class AlwaysFalse(PassthroughGate):
    ROUTE_TO_TAKE = False

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.1,
            output_cost_per_token=0.2,
            fixed_cost=0.1,
            latency_seconds=0.1,
        )


class PromptLengthGate(Gate):
    def run(self, ctx: EvalContext) -> bool:
        return len(ctx.prompt) % 2 == 0

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.2,
            latency_seconds=0.2,
        )


class Caser(Enricher):
    def input_tokens(self, ctx: EvalContext) -> int:
        return len(ctx.response.split())

    def output_tokens(self, ctx: EvalContext) -> int:
        return len(ctx.response.split())


class LowerCaser(Caser):
    """Enriches by returning the response lowercased."""

    def run(self, ctx: EvalContext) -> str:
        return ctx.response.lower()

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.3,
            output_cost_per_token=0.4,
            fixed_cost=0.3,
            latency_seconds=0.3,
        )


class UpperCaser(Caser):
    """Enriches by returning the response uppercased."""

    def run(self, ctx: EvalContext) -> str:
        return ctx.response.upper()

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.4,
            output_cost_per_token=0.5,
            fixed_cost=0.4,
            latency_seconds=0.4,
        )


class LLMEnricher(Enricher):

    def run(self, ctx: EvalContext) -> str:
        return ctx.response

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.5,
            output_cost_per_token=0.6,
            fixed_cost=0.5,
            latency_seconds=0.5,
        )

    def input_tokens(self, ctx: EvalContext) -> int:
        return context_token_count(ctx)

    def output_tokens(self, ctx: EvalContext) -> int:
        return context_token_count(ctx)


class FixedScorer(Enricher):
    """Returns a fixed float score regardless of context."""

    def __init__(self, name: str, value: float, **kwargs):
        super().__init__(name, **kwargs)
        self.value = value

    def run(self, ctx: EvalContext) -> float:
        return self.value

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.6,
            latency_seconds=0.6,
        )


class LowerCaseScorer(Enricher):
    """Scores based on the percentage of lowercase characters in the response."""

    def run(self, ctx: EvalContext) -> float:
        if not ctx.response:
            return 0.0
        num_lower = sum(1 for c in ctx.response if c.islower())
        return num_lower / len(ctx.response)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.7,
            latency_seconds=0.7,
        )


class UpperCaseScorer(Enricher):
    """Scores based on the percentage of uppercase characters in the response."""

    def run(self, ctx: EvalContext) -> float:
        if not ctx.response:
            return 0.0
        num_upper = sum(1 for c in ctx.response if c.isupper())
        return num_upper / len(ctx.response)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.8,
            latency_seconds=0.8,
        )


class AlwaysUnsafe(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return Safety(is_safe=False)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.9,
            latency_seconds=0.9,
        )

    @property
    def output_type(self) -> type:
        return Safety


class AlwaysSafe(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return Safety(is_safe=True)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.0,
            latency_seconds=1.0,
        )

    @property
    def output_type(self) -> type:
        return Safety


class ThresholdArbiter(Arbiter):
    def __init__(self, name: str, threshold: float, **kwargs):
        super().__init__(name, **kwargs)
        self.threshold = threshold

    def run(self, ctx: EvalContext) -> Output:
        scores = ctx.parent_outputs()
        score = sum(scores) / len(scores)
        return Safety(is_safe=score < self.threshold)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.1,
            latency_seconds=1.1,
        )

    @property
    def output_type(self) -> type:
        return Safety


class UnexpectedOutput(Output):
    @property
    def name(self) -> str:
        return "UNEXPECTED_OUTPUT"


class UnexpectedArbiter(Arbiter):
    """An arbiter that returns an output not declared in outputs()."""

    def run(self, ctx: EvalContext) -> Output:
        return UnexpectedOutput()

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.2,
            latency_seconds=1.2,
        )

    @property
    def output_type(self) -> type:
        return UnexpectedOutput


class BadArbiter(Arbiter):
    """An arbiter that violates the contract by returning a non-Output value."""

    def run(self, ctx: EvalContext) -> str:
        return "safe"

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.3,
            latency_seconds=1.3,
        )

    @property
    def output_type(self) -> type:
        return Safety
