"""DAGAnnotator and EvaluatorDAG implementation."""

import collections
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTResponse

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

        refusal_gate     = MyRefusalGate("RefusalGate", routes_true=[NONVIOLATING], routes_false=["NonRefusal"])
        eval_non_refusal = MyNonRefusalEvaluator("NonRefusal", routes=["Arbiter"])
        arbiter          = MyArbiter("Arbiter")

        dag = (
            EvaluatorDAG("refusal_gated_safety_evaluator", outputs=[NONVIOLATING, VIOLATING])
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

    @property
    def outputs(self) -> list[Output]:
        """Return the list of Output nodes declared in the DAG constructor."""
        return list(self._outputs.values())

    def add_node(
        self,
        node: EvaluatorDAGNode,
    ) -> "EvaluatorDAG":
        """Register a node with its routes."""

        if node.name in self._all_names():
            raise ValueError(
                f"A different node named {node.name} is already registered."
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
                if t in self._outputs:
                    continue
                in_degree[t] += 1

        root_nodes = [n for n in self._nodes if in_degree[n] == 0]
        queue = collections.deque(root_nodes)
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in all_routes.get(current, []):
                if child in self._outputs:
                    continue
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

    def _run_traced(
        self, ctx: EvalContext
    ) -> tuple[Output, dict[str, Any], set[tuple[str, str]]]:
        """Execute the DAG and return (final output, node outputs, traversed edges)."""
        active_nodes = self._root_nodes
        node_outputs: dict[str, Any] = {}
        traversed_edges: set[tuple[str, str]] = set()
        while active_nodes:
            next_active = []
            for node_name in active_nodes:
                ctx.set_parent_outputs(
                    {
                        pred: node_outputs[pred]
                        for pred in self._predecessors[node_name]
                        if pred in node_outputs
                    }
                )
                node = self._nodes[node_name]
                output = node.run(ctx)
                if isinstance(output, Output):
                    traversed_edges.add((node_name, output.name))
                    return output, node_outputs, traversed_edges
                node_outputs[node_name] = output
                for target in node.next_nodes(output):
                    t = target if isinstance(target, str) else target.name
                    traversed_edges.add((node_name, t))
                    if isinstance(target, Output):
                        return target, node_outputs, traversed_edges
                    next_active.append(t)
            active_nodes = next_active
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
    def visualize(
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

        def _format_output(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.3g}"
            s = str(value)
            return s if len(s) <= 30 else s[:27] + "..."

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
            if traced and node_name not in node_outputs:
                attrs = dict(_DIM, shape=base_style.get("shape", "box"))
                label = node_name
            else:
                attrs = dict(base_style)
                if traced:
                    raw = node_outputs[node_name]  # type: ignore[index]
                    label = f"{node_name}\n{_format_output(raw)}"
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
        except graphviz.ExecutableNotFound:
            raise RuntimeError(
                "Graphviz system binaries not found. Install them with:\n"
                "  macOS:  brew install graphviz\n"
                "  Ubuntu: apt-get install graphviz\n"
                "  conda:  conda install graphviz"
            ) from None

    @requires_validate_and_build
    def visualize_run(self, ctx: EvalContext):
        """Run the DAG on ctx and return a PNG with the executed path highlighted."""
        final_output, node_outputs, traversed_edges = self._run_traced(ctx)
        return self.visualize(
            node_outputs=node_outputs,
            traversed_edges=traversed_edges,
            final_output=final_output,
        )


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


def SafetyDAGAnnotator(DAGAnnotator):

    def __init__(self, uid: str, dag: EvaluatorDAG) -> None:
        super().__init__(uid, dag)
        if not all(isinstance(o, Safety) for o in dag.outputs):
            raise ValueError("All outputs of the DAG must be of type Safety.")

    def translate_response(
        self,
        request: EvalContext,
        response: Output,
    ) -> SafetyAnnotation:
        """Map DAGResult verdict to a SafetyAnnotation (is_safe bool)."""
        # TODO: unclear whether SafetyAnnotation is the right standardized output
        return SafetyAnnotation(is_safe=response.is_safe)
