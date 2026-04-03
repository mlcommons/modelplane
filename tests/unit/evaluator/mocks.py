from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.nodes import Arbiter, Enricher, Gate, Scorer
from modelplane.evaluator.outputs import NONVIOLATING, VIOLATING, Output


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
        return len(ctx.prompt_text) % 2 == 0


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
        return len(ctx.prompt_text) + len(ctx.response)

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


class AlwaysViolating(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return VIOLATING

    def outputs(self) -> list[Output]:
        return [VIOLATING]


class AlwaysNonViolating(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return NONVIOLATING

    def outputs(self) -> list[Output]:
        return [NONVIOLATING]


class ThresholdArbiter(Arbiter):
    def __init__(self, name: str, threshold: float, **kwargs):
        super().__init__(name, **kwargs)
        self.threshold = threshold

    def run(self, ctx: EvalContext) -> Output:
        scores = ctx.parent_outputs()
        score = sum(scores) / len(scores)
        return VIOLATING if score >= self.threshold else NONVIOLATING

    def outputs(self) -> list[Output]:
        return [VIOLATING, NONVIOLATING]
