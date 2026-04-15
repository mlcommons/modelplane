"""
Node types for the EvaluatorDAG pipeline.

Class hierarchy:

    EvaluatorNode (ABC)
    ├── Gate       (binary test; routes on True/False)
    ├── Enricher   (produces arbitary output; routes forward unconditionally)
    ├── Arbiter    (produces output; routes to outputs only)
    └── Output     (terminal node; carries a verdict value)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.cost import CostInfo, RealizedCost
from modelplane.evaluator.outputs import Output


class EvaluatorDAGNode(ABC):
    def __init__(
        self,
        name: str,
        routes_true: Optional[Sequence[str | Output]] = None,
        routes_false: Optional[Sequence[str | Output]] = None,
        routes: Optional[Sequence[str | Output]] = None,
    ) -> None:
        self.name = name
        self._routes_true: tuple[str | Output, ...] = tuple(routes_true or [])
        self._routes_false: tuple[str | Output, ...] = tuple(routes_false or [])
        self._routes: tuple[str | Output, ...] = tuple(routes or [])
        self.validate()

    @property
    def routes_true(self) -> tuple[str | Output, ...]:
        return self._routes_true

    @property
    def routes_false(self) -> tuple[str | Output, ...]:
        return self._routes_false

    @property
    def routes(self) -> tuple[str | Output, ...]:
        return self._routes

    @abstractmethod
    def run(self, ctx: EvalContext) -> Any:
        """Execute the node and return its output."""
        raise NotImplementedError

    @property
    def cost(self) -> CostInfo:
        """Override this to represent the cost of running this node."""
        return CostInfo()

    def realized_cost(self, ctx: EvalContext) -> RealizedCost:
        return RealizedCost(
            input_token_cost=self.input_tokens(ctx) * self.cost.input_cost_per_token,
            output_token_cost=self.output_tokens(ctx) * self.cost.output_cost_per_token,
            fixed_cost=self.cost.fixed_cost,
            latency_seconds=self.cost.latency_seconds,
        )

    def input_tokens(self, ctx: EvalContext) -> int:
        """Compute the number of LLM input tokens for this node given ctx."""
        return 0

    def output_tokens(self, ctx: EvalContext) -> int:
        """Compute the number of LLM output tokens for this node given ctx."""
        return 0

    def __repr__(self) -> str:
        return f"{self.name!r}: ({self.__class__.__name__})"

    def format_output(self, output: Any) -> str:
        """Convenience method to format the node's output for debugging/visualization."""
        if isinstance(output, float):
            return f"{output:.3g}"
        s = str(output)
        return s if len(s) <= 30 else s[:27] + "..."

    def all_routes(self) -> list[str | Output]:
        """Return a list of all route targets from this node."""
        return [*self.routes_true, *self.routes_false, *self.routes]

    def next_nodes(self, output: Any) -> tuple[str | Output, ...]:
        """Given the node's output value, return the tuple of next node names to activate."""
        if isinstance(self, Gate):
            return self.routes_true if output else self.routes_false
        else:
            return self.routes

    def validate(self) -> None:
        """Validate that the node's routing configuration is consistent with its type."""
        # validate that routes with Outputs only have one Output
        for route_list in [self.routes_true, self.routes_false, self.routes]:
            output_routes = [r for r in route_list if isinstance(r, Output)]
            if len(output_routes) > 1:
                raise ValueError(
                    f"{self!r} has multiple Output routes {output_routes}, which is not allowed."
                )


def _validate_binary_routes(node: EvaluatorDAGNode) -> None:
    if not node.routes_true or not node.routes_false:
        raise ValueError(f"{node!r} requires both routes_true and routes_false")
    if node.routes:
        raise ValueError(
            f"{node!r} should not have routes= (use routes_true= / routes_false=)"
        )


def _validate_unary_routes(node: EvaluatorDAGNode) -> None:
    if not node.routes:
        raise ValueError(f"{node!r} requires routes=")
    if node.routes_true or node.routes_false:
        raise ValueError(
            f"{node!r} should not have routes_true= / routes_false= (use routes=)"
        )


def _validate_terminal(node: EvaluatorDAGNode) -> None:
    if node.routes_true or node.routes_false or node.routes:
        raise ValueError(f"{node!r} is terminal and cannot have routing kwargs")


class Gate(EvaluatorDAGNode):
    """Binary test node."""

    @abstractmethod
    def run(self, ctx: EvalContext) -> bool:
        """Return True or False to indicate which route to take from this gate."""

    def validate(self) -> None:
        super().validate()
        _validate_binary_routes(self)


class Enricher(EvaluatorDAGNode):
    """Context transformation node."""

    @abstractmethod
    def run(self, ctx: EvalContext) -> Any:
        """Return data representing the enriched context."""

    def validate(self) -> None:
        super().validate()
        _validate_unary_routes(self)


class Arbiter(EvaluatorDAGNode):
    """Takes context and returns an Output indicating the final verdict (based on routes)."""

    @abstractmethod
    def run(self, ctx: EvalContext) -> Output:
        """Return an Output indicating the final verdict."""

    def validate(self) -> None:
        super().validate()
        _validate_terminal(self)

    @property
    @abstractmethod
    def output_type(self) -> type:
        """Return the expected type of the Output's value for validation."""
        raise NotImplementedError
