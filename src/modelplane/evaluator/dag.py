"""DAGAnnotator and EvaluatorDAG implementation."""

import collections
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any, Optional

import pandas as pd

from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.nodes import Arbiter, EvaluatorDAGNode, Gate, Output


def requires_validate_and_build(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._validate_and_build()
        return method(self, *args, **kwargs)

    return wrapper


class EvaluatorDAG:
    """DAG of EvaluatorNodes.

    Usage:

        refusal_gate     = MyRefusalGate("RefusalGate", routes_true=[Score(value=1)], routes_false=["NonRefusal"])
        eval_non_refusal = MyNonRefusalEvaluator("NonRefusal", routes=["Arbiter"])
        arbiter          = MyArbiter("Arbiter")

        dag = (
            EvaluatorDAG("refusal_gated_safety_evaluator", output_type=Safety)
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

    def __init__(self, name: str, output_type: type) -> None:
        self.name = name
        self._nodes: dict[str, EvaluatorDAGNode] = {}
        self._root_nodes: list[str] = []
        self._ordered: list[str] = []
        self._validated: bool = False
        self._predecessors: dict[str, list[str]] = collections.defaultdict(list)
        if not issubclass(output_type, Output):
            raise ValueError("output_type must be a subclass of Output.")
        self._output_type = output_type

    def add_node(
        self,
        node: EvaluatorDAGNode,
    ) -> "EvaluatorDAG":
        """Register a node with its routes."""

        if node.name in self._nodes:
            raise ValueError(
                f"A different node named {node.name} is already registered."
            )
        self._nodes[node.name] = node
        self._validated = False
        return self

    def _validate_and_build(self) -> None:
        """
        Validate the DAG:
        - All routes reference registered nodes or instances of the output type.
        - No cycles.
        - All paths lead to an instance of the output type.

        Build:
        - _predecessors: dict mapping node name to list of parent node names (for context during execution)
        - _root_nodes: list of node names with no incoming routes (starting points)
        - _ordered: list of node names in topological order (valid execution order)
        """
        # skip validation if we've already done it and the DAG hasn't changed
        if self._validated:
            return

        # check that all route targets reference registered nodes or instances of the output type
        for node_name, node in self._nodes.items():
            for target in node.all_routes():
                if target not in self._nodes and not isinstance(
                    target, self._output_type
                ):
                    raise ValueError(
                        f"Node {node_name} routes to unregistered node {target} or incompatible output."
                    )

        # check for cycles (kahn's algorithm)
        all_routes = {name: node.all_routes() for name, node in self._nodes.items()}
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for routes in all_routes.values():
            for route in routes:
                if isinstance(route, Output):
                    continue
                in_degree[route] += 1

        root_nodes = [n for n in self._nodes if in_degree[n] == 0]
        queue = collections.deque(root_nodes)
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in all_routes.get(current, []):
                if isinstance(child, Output):
                    continue
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(self._nodes):
            nodes_in_cycle = set(self._nodes) - set(ordered)
            raise ValueError(f"DAG contains a cycle. Nodes in cycle: {nodes_in_cycle}")

        # check all terminal Arbiter nodes have correct output types
        terminal_nodes = [n for n in self._nodes if not all_routes.get(n)]
        for terminal in terminal_nodes:
            node = self._nodes[terminal]
            if isinstance(node, Arbiter):
                if not issubclass(node.output_type, self._output_type):
                    raise ValueError(
                        f"Terminal Arbiter node {terminal} has output_type {node.output_type}, which is not compatible with the DAG's output_type {self._output_type}."
                    )

        # build predecessors
        for name, node in self._nodes.items():
            for target in node.all_routes():
                if isinstance(target, Output):
                    continue
                self._predecessors[target].append(name)

        self._validated = True
        self._root_nodes = root_nodes
        self._ordered = ordered

    def _run_traced(
        self, ctx: EvalContext
    ) -> tuple[Output, dict[str, Any], set[tuple[str, str]]]:
        """Execute the DAG and return (final output, node outputs, traversed edges)."""
        node_outputs: dict[str, Any] = {}
        traversed_edges: set[tuple[str, str]] = set()
        reachable: set[str] = set(self._root_nodes)
        for node_name in self._ordered:
            if node_name not in reachable:
                continue
            ctx.set_parent_outputs(
                {
                    pred: node_outputs[pred]
                    for pred in self._predecessors[node_name]
                    if pred in node_outputs
                }
            )
            node = self._nodes[node_name]
            output = node.run(ctx)
            node_outputs[node_name] = output
            if isinstance(output, Output):
                traversed_edges.add((node_name, output.name))
                return output, node_outputs, traversed_edges
            for target in node.next_nodes(output):
                t = target if isinstance(target, str) else target.name
                traversed_edges.add((node_name, t))
                if isinstance(target, Output):
                    return target, node_outputs, traversed_edges
                reachable.add(t)
        raise ValueError("DAG execution completed without reaching an Output node.")

    @requires_validate_and_build
    def run(
        self,
        ctx: EvalContext,
    ) -> Output:
        """Execute the DAG on a single prompt/response."""
        output, _, _ = self._run_traced(ctx)
        return output

    @requires_validate_and_build
    def run_dataframe(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt",
        response_col: str = "response",
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Run the DAG over every row of a DataFrame."""

        def _run_row(row: Any) -> Output:
            ctx = EvalContext(
                prompt=str(row[prompt_col]),
                response=str(row[response_col]),
            )
            return self.run(ctx)

        rows = [row for _, row in df.iterrows()]

        if n_jobs == 1:
            records = [_run_row(row) for row in rows]
        else:
            max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                records = list(executor.map(_run_row, rows))

        result_df = pd.DataFrame(
            {self.DATAFRAME_OUTPUT_COL: [r.name for r in records]}, index=df.index
        )
        return pd.concat([df, result_df], axis=1)

    @requires_validate_and_build
    def total_cost(self, ctx: Optional[EvalContext] = None) -> float:
        """Run the DAG on ctx and return the total cost of the executed path."""
        if ctx is None:
            ctx = EvalContext(prompt="", response="")
        _, node_outputs, _ = self._run_traced(ctx)
        total = 0.0
        for node_name in node_outputs:
            node = self._nodes[node_name]
            total += node.cost(ctx)
        return total

    @requires_validate_and_build
    def total_costs(self) -> dict[str, float]:
        """Run the DAG on all terminal paths and report total costs per path."""
        ctx = EvalContext(prompt="", response="")
        gates = [name for name, node in self._nodes.items() if isinstance(node, Gate)]
        path_costs: dict[str, float] = {}

        for combo in product([True, False], repeat=len(gates)):
            gate_outcomes = dict(zip(gates, combo))
            reachable: set[str] = set(self._root_nodes)
            path: list[str] = []
            total = 0.0

            for node_name in self._ordered:
                if node_name not in reachable:
                    continue
                node = self._nodes[node_name]
                total += node.cost(ctx)
                path.append(node_name)
                if isinstance(node, Gate):
                    targets = (
                        node.routes_true
                        if gate_outcomes[node_name]
                        else node.routes_false
                    )
                elif isinstance(node, Arbiter):
                    targets = []
                else:
                    targets = node.routes
                for target in targets:
                    if not isinstance(target, Output):
                        reachable.add(
                            target if isinstance(target, str) else target.name
                        )

            base_path = " -> ".join(path)
            path_costs[f"{base_path} -> {self._output_type}"] = total

        return path_costs

    def _visualize(
        self,
        node_outputs: Optional[dict[str, Any]] = None,
        traversed_edges: Optional[set[tuple[str, str]]] = None,
        final_output: Optional[Output] = None,
    ):
        """Render the DAG as a PNG image. In a Jupyter notebook the image is displayed inline.

        When node_outputs/traversed_edges/final_output are provided (via visualize_run),
        the hot path is highlighted and each node shows its output value.
        """
        import graphviz
        from IPython.display import Image

        traced = node_outputs is not None

        _NODE_STYLES: dict[type, dict] = {
            Gate: {"shape": "diamond", "style": "filled", "fillcolor": "#d0e8f5"},
            Arbiter: {"shape": "box", "style": "filled", "fillcolor": "#c8e6c9"},
            Output: {"shape": "ellipse", "style": "filled", "fillcolor": "#fff9c4"},
        }
        _DEFAULT_STYLE = {"shape": "box", "style": "filled", "fillcolor": "#ffe0b2"}
        _DIM = {
            "style": "filled",
            "fillcolor": "#f0f0f0",
            "color": "#bbbbbb",
            "fontcolor": "#aaaaaa",
        }

        dot = graphviz.Digraph(name=self.name)
        dot.attr(
            label=self.name,
            labelloc="t",
            fontsize="13",
            fontname="Helvetica",
            rankdir="TB",
            ranksep="0.5",
            nodesep="0.4",
        )
        dot.attr("node", fontname="Helvetica", fontsize="11")
        dot.attr("edge", fontname="Helvetica", fontsize="10")

        # implicit input node pinned to the top
        top = graphviz.Digraph()
        top.attr(rank="min")
        top.node(
            "__input__",
            "prompt\nresponse",
            shape="box",
            style="dashed",
            fillcolor="white",
            color="#888888",
            fontcolor="#555555",
        )
        dot.subgraph(top)

        # output terminal nodes pinned to the bottom
        bottom = graphviz.Digraph()
        bottom.attr(rank="max")
        for output_name, output_node in self._outputs.items():
            attrs = dict(_NODE_STYLES[Output])
            if traced:
                if output_node is final_output:
                    attrs["penwidth"] = "2.5"
                else:
                    attrs = dict(_DIM, shape="ellipse")
            bottom.node(output_name, **attrs)
        dot.subgraph(bottom)

        # processing nodes
        for node_name, node in self._nodes.items():
            base_style = next(
                (s for t, s in _NODE_STYLES.items() if isinstance(node, t)),
                _DEFAULT_STYLE,
            )
            node_was_active = (
                node_outputs is not None and node_name in node_outputs
            ) or (
                traversed_edges is not None
                and any(src == node_name for src, _ in traversed_edges)
            )
            if traced and not node_was_active:
                attrs = dict(_DIM, shape=base_style.get("shape", "box"))
                label = node_name
            else:
                attrs = dict(base_style)
                if traced:
                    raw = node_outputs[node_name]  # type: ignore[index]
                    label = f"{node_name}\n{node.format_output(raw)}"
                    attrs["penwidth"] = "2.5"
                else:
                    label = node_name
            dot.node(node_name, label, **attrs)

        # dashed edges from implicit input to root nodes
        for root in self._root_nodes:
            dot.edge(
                "__input__", root, style="dashed", color="#888888", arrowhead="open"
            )

        # edges between processing nodes
        for node_name, node in self._nodes.items():
            if isinstance(node, Gate):
                for target in node.routes_true:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    dot.edge(
                        node_name,
                        t,
                        label=" True",
                        color="#2e7d32" if hot else "#cccccc",
                        fontcolor="#2e7d32" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )
                for target in node.routes_false:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    dot.edge(
                        node_name,
                        t,
                        label=" False",
                        color="#c62828" if hot else "#cccccc",
                        fontcolor="#c62828" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )
            elif isinstance(node, Arbiter):
                for output in node.outputs():
                    hot = not traced or (node_name, output.name) in traversed_edges  # type: ignore[operator]
                    dot.edge(
                        node_name,
                        output.name,
                        color="#555555" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )
            else:
                for target in node.routes:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    dot.edge(
                        node_name,
                        t,
                        color="#555555" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )

        try:
            return Image(dot.pipe(format="png"))
        except graphviz.ExecutableNotFound as e:
            raise RuntimeError(
                "Graphviz system binaries not found. Install them with:\n"
                "  macOS:  brew install graphviz\n"
                "  Ubuntu: apt-get install graphviz\n"
                "  conda:  conda install graphviz"
            ) from e

    @requires_validate_and_build
    def visualize(self):
        """Visualize the DAG structure without execution."""
        return self._visualize()

    @requires_validate_and_build
    def visualize_run(self, ctx: EvalContext):
        """Run the DAG on ctx and return a visualization with the executed path highlighted."""
        final_output, node_outputs, traversed_edges = self._run_traced(ctx)
        return self._visualize(
            node_outputs=node_outputs,
            traversed_edges=traversed_edges,
            final_output=final_output,
        )
