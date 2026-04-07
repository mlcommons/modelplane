from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.nodes import Arbiter, Enricher, Gate, Scorer
from modelplane.evaluator.outputs import SAFE, UNSAFE, Output


class PassthroughGate(Gate):
    ROUTE_TO_TAKE: bool

    def run(self, ctx: EvalContext) -> bool:
        return self.ROUTE_TO_TAKE


class AlwaysTrue(PassthroughGate):
    ROUTE_TO_TAKE = True


class AlwaysFalse(PassthroughGate):
    ROUTE_TO_TAKE = False


class PromptLengthGate(Gate):
    def run(self, ctx: EvalContext) -> bool:
        return len(ctx.prompt) % 2 == 0


class LowerCaser(Enricher):
    """Enriches by returning the response lowercased."""

    def run(self, ctx: EvalContext) -> str:
        return ctx.response.lower()


class UpperCaser(Enricher):
    """Enriches by returning the response uppercased."""

    def run(self, ctx: EvalContext) -> str:
        return ctx.response.upper()


class LLMEnricher(Enricher):

    def cost(self, ctx: EvalContext) -> float:
        return len(ctx.prompt) + len(ctx.response)

    def run(self, ctx: EvalContext) -> str:
        return ctx.response


class FixedScorer(Scorer):
    """Returns a fixed float score regardless of context."""

    def __init__(self, name: str, value: float, **kwargs):
        super().__init__(name, **kwargs)
        self.value = value

    def run(self, ctx: EvalContext) -> float:
        return self.value


class LowerCaseScorer(Scorer):
    """Scores based on the percentage of lowercase characters in the response."""

    def run(self, ctx: EvalContext) -> float:
        if not ctx.response:
            return 0.0
        num_lower = sum(1 for c in ctx.response if c.islower())
        return num_lower / len(ctx.response)


class UpperCaseScorer(Scorer):
    """Scores based on the percentage of uppercase characters in the response."""

    def run(self, ctx: EvalContext) -> float:
        if not ctx.response:
            return 0.0
        num_upper = sum(1 for c in ctx.response if c.isupper())
        return num_upper / len(ctx.response)


class AlwaysUnsafe(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return UNSAFE

    def outputs(self) -> list[Output]:
        return [UNSAFE]


class AlwaysSafe(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return SAFE

    def outputs(self) -> list[Output]:
        return [SAFE]


class ThresholdArbiter(Arbiter):
    def __init__(self, name: str, threshold: float, **kwargs):
        super().__init__(name, **kwargs)
        self.threshold = threshold

    def run(self, ctx: EvalContext) -> Output:
        scores = ctx.parent_outputs()
        score = sum(scores) / len(scores)
        return UNSAFE if score >= self.threshold else SAFE

    def outputs(self) -> list[Output]:
        return [UNSAFE, SAFE]
