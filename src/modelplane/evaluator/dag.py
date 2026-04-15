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

    @property
    def output_type(self) -> type:
        return self._output_type

    @property
    def dataframe_output_col(self) -> str:
        return f"{self.name}_output"

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

        # check that all route targets reference registered nodes or instances
        # of the output type, and that all Arbiters have compatible output types
        for node_name, node in self._nodes.items():
            if isinstance(node, Arbiter):
                if not issubclass(node.output_type, self.output_type):
                    raise ValueError(
                        f"Node {node_name} is an Arbiter with output_type {node.output_type.__name__}, which is not compatible with the DAG's output_type {self.output_type.__name__}."
                    )
            for target in node.all_routes():
                if target not in self._nodes and not isinstance(
                    target, self.output_type
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
            {self.dataframe_output_col: [r.name for r in records]}, index=df.index
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
            path_costs[f"{base_path} -> Out ({self.output_type.__name__})"] = total

        return path_costs

    def _visualize(
        self,
        node_outputs: Optional[dict[str, Any]] = None,
        traversed_edges: Optional[set[tuple[str, str]]] = None,
        final_output: Optional[Output] = None,
        ctx: Optional[EvalContext] = None,
    ):
        """Render the DAG as a PNG image. In a Jupyter notebook the image is displayed inline.

        When node_outputs/traversed_edges/final_output are provided (via visualize_run),
        the hot path is highlighted and each node shows its output value.

        NOTE: this helper method is vibe-coded and provided as-is.
        """
        import graphviz
        from IPython.display import Image

        traced = node_outputs is not None

        _NODE_STYLES: dict[type, dict] = {
            Gate: {"shape": "diamond", "style": "filled", "fillcolor": "#ffe082"},
            Arbiter: {"shape": "hexagon", "style": "filled", "fillcolor": "#e1bee7"},
            Output: {
                "shape": "rectangle",
                "style": "filled,rounded",
                "fillcolor": "#dcedc8",
            },
        }
        _OUTPUT_TYPE_STYLE = {
            "shape": "rectangle",
            "style": "filled,rounded,dashed",
            "fillcolor": "#dcedc8",
        }
        _DEFAULT_STYLE = {
            "shape": "rectangle",
            "style": "filled",
            "fillcolor": "#eeeeee",
        }
        _DIM = {
            "style": "filled",
            "fillcolor": "#f0f0f0",
            "color": "#bbbbbb",
            "fontcolor": "#aaaaaa",
        }

        _NODE_W, _NODE_H = 1.5, 0.5  # inches, fixed for all nodes

        def _fontsize(
            label: str, max_fs: float = 11.0, min_fs: float = 7.0, fill: float = 0.8
        ) -> str:
            """Scale font size so the longest line fits within _NODE_W.

            fill: fraction of the node width usable for text. Shapes like diamonds,
            hexagons, and parallelograms have less usable area than rectangles, so
            pass a smaller fill value for those.
            """
            longest = max((len(line) for line in label.split("\n")), default=1)
            # approx: each char ≈ 0.55 × fontsize points
            fs = (_NODE_W * 72 * fill) / (longest * 0.55)
            return f"{max(min_fs, min(max_fs, fs)):.1f}"

        dot = graphviz.Digraph(name=self.name)
        dot.attr(
            label=self.name,
            labelloc="t",
            fontsize="13",
            fontname="Helvetica",
            rankdir="LR",
            ranksep="0.5",
            nodesep="0.4",
        )
        dot.attr(
            "node",
            fontname="Helvetica",
            fontsize="11",
            width=str(_NODE_W),
            height=str(_NODE_H),
            fixedsize="true",
        )
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        # implicit input node pinned to the left
        top = graphviz.Digraph()
        top.attr(rank="min")

        def _truncate(s: str, n: int = 24) -> str:
            return s if len(s) <= n else s[: n - 1] + "…"

        if ctx is not None:
            input_label = f"p: {_truncate(ctx.prompt)}\nr: {_truncate(ctx.response)}"
        else:
            input_label = "prompt\nresponse"
        top.node(
            "__input__",
            input_label,
            shape="parallelogram",
            style="filled",
            fillcolor="#b2dfdb",
            color="#4db6ac",
            fontcolor="#00695c",
            fontsize=_fontsize(input_label, fill=0.45),
        )
        dot.subgraph(top)

        # collect Output instances directly referenced in routes (from non-Arbiter nodes)
        direct_outputs: dict[str, Output] = {}
        has_arbiter = any(isinstance(n, Arbiter) for n in self._nodes.values())
        for node in self._nodes.values():
            if not isinstance(node, Arbiter):
                for target in node.all_routes():
                    if isinstance(target, Output):
                        direct_outputs[target.name] = target

        # whether the final output came from a direct route or an arbiter
        final_from_direct = traced and final_output in direct_outputs.values()

        bottom = graphviz.Digraph()
        bottom.attr(rank="max")

        # individual nodes for directly-routed Output instances, shown with their repr
        for out_name, out_inst in direct_outputs.items():
            attrs = dict(_NODE_STYLES[Output])
            if traced:
                if out_inst is final_output:
                    attrs["penwidth"] = "2.5"
                else:
                    attrs = dict(_DIM, shape="rectangle", style="filled,rounded")
            bottom.node(
                out_name, repr(out_inst), fontsize=_fontsize(repr(out_inst)), **attrs
            )

        # synthetic output type node for Arbiters
        if has_arbiter:
            output_node_id = f"__output_{self.output_type.__name__}__"
            output_label = f"{self.output_type.__name__} (?)"
            attrs = dict(_OUTPUT_TYPE_STYLE)
            if traced:
                if not final_from_direct and final_output is not None:
                    attrs = dict(_NODE_STYLES[Output])
                    attrs["penwidth"] = "2.5"
                    output_label = repr(final_output)
                elif final_from_direct:
                    attrs = dict(_DIM, shape="rectangle", style="filled,rounded")
            bottom.node(
                output_node_id, output_label, fontsize=_fontsize(output_label), **attrs
            )

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
            _fill = (
                0.45
                if isinstance(node, Gate)
                else 0.65 if isinstance(node, Arbiter) else 0.8
            )
            dot.node(node_name, label, fontsize=_fontsize(label, fill=_fill), **attrs)

        # edges from implicit input to root nodes
        for root in self._root_nodes:
            dot.edge("__input__", root, color="#888888")

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
                output_node_id = f"__output_{self.output_type.__name__}__"
                hot = not traced or node_name in (node_outputs or {})
                dot.edge(
                    node_name,
                    output_node_id,
                    color="#555555" if hot else "#cccccc",
                    penwidth="2" if hot and traced else "1",
                )
            else:
                for target in node.routes:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    edge_label = ""
                    if traced and hot and node_name in (node_outputs or {}):
                        edge_label = f" {node.format_output(node_outputs[node_name])}"  # type: ignore[index]
                    dot.edge(
                        node_name,
                        t,
                        label=edge_label,
                        color="#555555" if hot else "#cccccc",
                        fontcolor="#555555" if hot else "#cccccc",
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
        """Render the DAG structure as a PNG image (inline in Jupyter notebooks).

        The graph flows left to right. Node shapes and colors:
          - Input          — teal parallelogram (implicit; represents the prompt/response pair)
          - Gate           — amber diamond; edges labelled "True" (green) / "False" (red)
          - Enricher       — light grey rectangle; edges are unlabelled
          - Arbiter        — light purple hexagon; edge labelled with the output type name
          - Output (direct instance)   — soft green rounded rectangle, solid border;
                                         label is repr(output)
          - Output (type placeholder)  — soft green rounded rectangle, dashed border;
                                         label is the class name; shown when the DAG contains
                                         an Arbiter whose concrete value is only known at runtime

        Raises:
            RuntimeError: if the Graphviz system binaries are not installed.
        """
        return self._visualize()

    @requires_validate_and_build
    def visualize_run(self, ctx: EvalContext):
        """Run the DAG on ctx and return a visualization with the executed path highlighted.

        Identical layout to visualize(), with the following additions:
          - Active nodes are bolded and show their output value beneath the node name.
          - Inactive nodes are greyed out.
        """
        final_output, node_outputs, traversed_edges = self._run_traced(ctx)
        return self._visualize(
            node_outputs=node_outputs,
            traversed_edges=traversed_edges,
            final_output=final_output,
            ctx=ctx,
        )
