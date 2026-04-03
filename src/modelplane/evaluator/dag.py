"""DAGAnnotator and EvaluatorDAG implementation."""

import collections
from dataclasses import dataclass, field
import functools
from typing import Any, Optional

import pandas as pd
from modelgauge.annotator import Annotator

from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.nodes import (
    Arbiter,
    EvaluatorDAGNode,
    Gate,
    Output,
)
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTResponse
from modelgauge.annotation import SafetyAnnotation


def requires_validate_and_build(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._validate_and_build()
        return method(self, *args, **kwargs)

    return wrapper


class EvaluatorDAG:
    """DAG of EvaluatorNodes.

    Usage:

        refusal_gate     = MyRefusalGate("RefusalGate", routes_true=[NONVIOLATING], routes_false=["NonRefusal"])
        eval_non_refusal = MyNonRefusalEvaluator("NonRefusal", routes=["Arbiter"])
        arbiter          = MyArbiter("Arbiter", routes_true=[VIOLATING], routes_false=[NONVIOLATING])

        dag = (
            EvaluatorDAG("refusal_evaluator", outputs=[NONVIOLATING, VIOLATING])
            .add_node(refusal_gate)
            .add_node(eval_non_refusal)
            .add_node(arbiter)
        )
        # run single
        result = dag.run(prompt_uid="123", prompt="...", response="...")
        # run batch
        results_df = dag.run_dataframe(df)
    """

    DATAFRAME_OUTPUT_COL = "output"

    def __init__(self, name: str, outputs: list[Output]) -> None:
        self.name = name
        self._nodes: dict[str, EvaluatorDAGNode] = {}
        self._root_nodes: list[str] = []
        self._ordered: list[str] = []
        self._validated: bool = False
        self._predecessors: dict[str, list[str]] = collections.defaultdict(list)
        self._outputs = {output.name: output for output in outputs}

    def add_node(
        self,
        node: EvaluatorDAGNode,
    ) -> "EvaluatorDAG":
        """Register a node with its routes."""

        if node.name in self._all_names():
            raise ValueError(
                f"A different node named {node.name!r} is already registered."
            )
        self._nodes[node.name] = node
        self._validated = False
        return self

    def _all_names(self) -> dict[str, EvaluatorDAGNode | Output]:
        return {**self._nodes, **self._outputs}

    def _validate_and_build(self) -> None:
        """
        Validate the DAG:
        - All routes reference registered nodes.
        - No cycles.
        - All paths lead to an Output node.
        - All Output nodes are declared as outputs in the DAG constructor.

        Build:
        - _predecessors: dict mapping node name to list of parent node names (for context during execution)
        - _root_nodes: list of node names with no incoming routes (starting points)
        - _ordered: list of node names in topological order (valid execution order)
        """
        # skip validation if we've already done it and the DAG hasn't changed
        if self._validated:
            return

        all_named_entities = self._all_names()
        # check that all route targets reference registered nodes
        for node_name, node in self._nodes.items():
            for target in node.all_routes():
                if target not in all_named_entities:
                    raise ValueError(
                        f"Node {node_name!r} routes to unregistered node {target!r}."
                    )

        # check for cycles (kahn's algorithm)
        all_routes = {name: node.all_routes() for name, node in self._nodes.items()}
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for route in all_routes.values():
            for t in route:
                in_degree[t] += 1

        root_nodes = [n for n in self._nodes if in_degree[n] == 0]
        queue = collections.deque(root_nodes)
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in all_routes.get(current, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(self._nodes):
            # missing nodes
            missing = set(self._nodes) - set(ordered)
            raise ValueError(f"Graph contains a cycle. Missing nodes: {missing}")

        # check all terminal nodes are Output nodes
        terminal_nodes = [n for n in self._nodes if not all_routes.get(n)]
        for terminal in terminal_nodes:
            entity = all_named_entities[terminal]
            if isinstance(entity, Output) and terminal not in self._outputs:
                raise ValueError(
                    f"Terminal Output node {terminal!r} is not declared as an output in the DAG constructor."
                )
            elif isinstance(entity, Arbiter):
                if any(o.name not in self._outputs for o in entity.outputs()):
                    raise ValueError(
                        f"Terminal Arbiter node {terminal!r} has output(s) that are not declared as outputs in the DAG constructor."
                    )
            else:
                raise ValueError(
                    f"Terminal node {terminal!r} is not an Output or Arbiter node."
                )

        # get predecessors
        for name, node in self._nodes.items():
            for target in node.all_routes():
                self._predecessors[target].append(name)

        self._validated = True
        self._root_nodes = root_nodes
        self._ordered = ordered

    @requires_validate_and_build
    def run(
        self,
        ctx: EvalContext,
    ) -> Output:
        """
        Execute the DAG on a single prompt/response.
        """
        active_nodes = self._root_nodes
        outputs: dict[str, Any] = {}
        while active_nodes:
            next_active = []
            for node_name in active_nodes:
                # set parent outputs in context for this node
                ctx.set_parent_outputs(
                    {
                        pred: outputs[pred]
                        for pred in self._predecessors[node_name]
                        if pred in outputs
                    }
                )
                # run the node
                node = self._nodes[node_name]
                output = node.run(ctx)
                if isinstance(output, Output):
                    return output
                outputs[node_name] = output
                # see which nodes to activate next based on output and routing
                next_active.extend(node.next_nodes(output))
            active_nodes = next_active
        raise ValueError("DAG execution completed without reaching an Output node.")

    @requires_validate_and_build
    def run_dataframe(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt",
        response_col: str = "response",
    ) -> pd.DataFrame:
        """Run the DAG over every row of a DataFrame."""

        def _run_row(row: Any) -> Output:
            ctx = EvalContext(
                prompt=str(row[prompt_col]),
                response=str(row[response_col]),
            )
            return self.run(ctx)

        records = [_run_row(row) for _, row in df.iterrows()]

        result_df = pd.DataFrame(
            {self.DATAFRAME_OUTPUT_COL: [r.name for r in records]}, index=df.index
        )
        return pd.concat([df, result_df], axis=1)

    @requires_validate_and_build
    def total_cost(
        self,
        prompt: Optional[str],
        response: Optional[str],
    ) -> dict[str, float]:
        """Run the DAG on all terminal paths and report total costs per path.
        If no prompt/response are provided, uses empty strings."""

        ctx = EvalContext(
            prompt=prompt or "",
            response=response or "",
        )

        path_costs: dict[str, float] = {}

        def _dfs(node_name: str, accumulated: float, path: list[str]) -> None:
            node = self._nodes[node_name]
            total = accumulated + node.cost(ctx)
            if isinstance(node, Output):
                path_costs[" -> ".join(path + [node_name])] = total
                return
            for target in node.all_routes():
                _dfs(target, total, path + [node_name])

        for root in self._root_nodes:
            _dfs(root, 0.0, [])

        return path_costs

    @requires_validate_and_build
    def visualize(self) -> None:
        """Render the DAG structure with ascii."""
        print(f"EvaluatorDAG: {self.name!r}")
        print("=" * (len(self.name) + 18))
        for node_name in self._ordered:
            node = self._nodes[node_name]
            node_type = type(node).__name__
            if isinstance(node, Output):
                route_str = f"  → verdict='{node.name}'"
            elif isinstance(node, Gate):
                route_str = f"  → True:{node.routes_true}  False:{node.routes_false}"
            else:
                route_str = f"  → {node.routes}"
            print(f"  [{node_type:10s}] {node_name}{route_str}")


class DAGAnnotator(Annotator):
    """Annotator that executes a DAG."""

    def __init__(self, uid: str, dag: EvaluatorDAG) -> None:
        super().__init__(uid)
        self.dag = dag

    def translate_prompt(
        self,
        prompt: TextPrompt | ChatPrompt,
        response: SUTResponse,
    ) -> EvalContext:
        prompt_str = (
            prompt.text if isinstance(prompt, TextPrompt) else format_chat(prompt)
        )
        return EvalContext(
            prompt=prompt_str,
            response=response.text,
        )

    def annotate(self, annotation_request: EvalContext) -> Output:
        return self.dag.run(annotation_request)

    def translate_response(
        self,
        request: EvalContext,
        response: Output,
    ) -> SafetyAnnotation:
        """Map DAGResult verdict to a SafetyAnnotation (is_safe bool)."""
        # TODO: unclear whether SafetyAnnotation is the right standardized output
        return SafetyAnnotation(is_safe=response.is_safe())
