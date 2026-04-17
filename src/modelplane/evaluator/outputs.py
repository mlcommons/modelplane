from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from modelplane.evaluator.cost import RealizedCost


@dataclass
class NodeOutput:
    value: Any
    realized_cost: RealizedCost = field(default_factory=RealizedCost)

    def to_dict(self) -> dict:
        return {
            "value": str(self.value),
            "realized_cost": self.realized_cost.to_dict(),
        }


class Verdict:
    """DAG outputs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a string name for this output, used for routing and debugging."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.name})"


@dataclass
class DAGOutput:
    verdict: Verdict
    node_outputs: dict[str, NodeOutput]
    total_cost: RealizedCost
