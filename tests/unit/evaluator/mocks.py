from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.nodes import Arbiter, Enricher, Gate, Scorer
from modelplane.evaluator.outputs import SAFE, UNSAFE, Output


class PassthroughGate(Gate):
    ROUTE_TO_TAKE: bool

    def run(self, ctx: EvalContext) -> bool:
        return self.ROUTE_TO_TAKE


class AlwaysTrue(PassthroughGate):
    ROUTE_TO_TAKE = True

    def cost(self, ctx: EvalContext) -> float:
        return 0.1


class AlwaysFalse(PassthroughGate):
    ROUTE_TO_TAKE = False

    def cost(self, ctx: EvalContext) -> float:
        return 0.2


class PromptLengthGate(Gate):
    def run(self, ctx: EvalContext) -> bool:
        return len(ctx.prompt) % 2 == 0

    def cost(self, ctx: EvalContext) -> float:
        return 0.3


class LowerCaser(Enricher):
    """Enriches by returning the response lowercased."""

    def run(self, ctx: EvalContext) -> str:
        return ctx.response.lower()

    def cost(self, ctx: EvalContext) -> float:
        return 0.4


class UpperCaser(Enricher):
    """Enriches by returning the response uppercased."""

    def run(self, ctx: EvalContext) -> str:
        return ctx.response.upper()

    def cost(self, ctx: EvalContext) -> float:
        return 0.5


class LLMEnricher(Enricher):

    def run(self, ctx: EvalContext) -> str:
        return ctx.response

    def cost(self, ctx: EvalContext) -> float:
        return 0.6


class FixedScorer(Scorer):
    """Returns a fixed float score regardless of context."""

    def __init__(self, name: str, value: float, **kwargs):
        super().__init__(name, **kwargs)
        self.value = value

    def run(self, ctx: EvalContext) -> float:
        return self.value

    def cost(self, ctx: EvalContext) -> float:
        return 0.7


class LowerCaseScorer(Scorer):
    """Scores based on the percentage of lowercase characters in the response."""

    def run(self, ctx: EvalContext) -> float:
        if not ctx.response:
            return 0.0
        num_lower = sum(1 for c in ctx.response if c.islower())
        return num_lower / len(ctx.response)

    def cost(self, ctx: EvalContext) -> float:
        return 0.8


class UpperCaseScorer(Scorer):
    """Scores based on the percentage of uppercase characters in the response."""

    def run(self, ctx: EvalContext) -> float:
        if not ctx.response:
            return 0.0
        num_upper = sum(1 for c in ctx.response if c.isupper())
        return num_upper / len(ctx.response)

    def cost(self, ctx: EvalContext) -> float:
        return 0.9


class AlwaysUnsafe(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return UNSAFE

    def outputs(self) -> list[Output]:
        return [UNSAFE]

    def cost(self, ctx: EvalContext) -> float:
        return 1.0


class AlwaysSafe(Arbiter):
    def run(self, ctx: EvalContext) -> Output:
        return SAFE

    def outputs(self) -> list[Output]:
        return [SAFE]

    def cost(self, ctx: EvalContext) -> float:
        return 1.1


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

    def cost(self, ctx: EvalContext) -> float:
        return 1.2


class UnexpectedOutput(Output):
    @property
    def name(self) -> str:
        return "UNEXPECTED_OUTPUT"


class UnexpectedArbiter(Arbiter):
    """An arbiter that returns an output not declared in outputs()."""

    def run(self, ctx: EvalContext) -> Output:
        return UnexpectedOutput()

    def outputs(self) -> list[Output]:
        return [UnexpectedOutput()]

    def cost(self, ctx: EvalContext) -> float:
        return 1.3


class BadArbiter(Arbiter):
    """An arbiter that violates the contract by returning a non-Output value."""

    def run(self, ctx: EvalContext) -> str:
        return "safe"

    def outputs(self) -> list[Output]:
        return [SAFE]

    def cost(self, ctx: EvalContext) -> float:
        return 1.4
