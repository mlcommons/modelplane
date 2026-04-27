from pathlib import Path
from typing import Optional

from modelbench.cache import DiskCache, NullCache
from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator, SUTResponse, TextPrompt

from modelplane.evaluator.annotator import DAGAnnotator
from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.dag import EvaluatorDAG
from modelplane.evaluator.nodes import Arbiter, NodeOutput
from modelplane.evaluator.verdict import Verdict


class Safety(Verdict):

    def __init__(self, is_safe: bool) -> None:
        self.is_safe = is_safe

    @property
    def name(self) -> str:
        return "SAFE" if self.is_safe else "UNSAFE"


class SafetyArbiter(Arbiter):
    @property
    def verdict_type(self) -> type:
        return Safety


class SafetyDAGAnnotator(DAGAnnotator):
    """Implementation of DAGAnnotator that produces a SafetyAnnotation."""

    def __init__(self, uid: str, dag: EvaluatorDAG) -> None:
        super().__init__(uid, dag)
        if not issubclass(dag.verdict_type, Safety):
            raise ValueError("All outputs of the DAG must be of type Safety.")

    def translate_response(
        self,
        request: EvalContext,
        response: Safety,
    ) -> SafetyAnnotation:
        """Map DAGResult verdict to a SafetyAnnotation (is_safe bool)."""
        return SafetyAnnotation(is_safe=response.is_safe)


class AnnotatorArbiter(SafetyArbiter):
    """Arbiter that outputs SAFE or UNSAFE based on the output of a (safety) Annotator.

    Optionally caches annotations."""

    def __init__(
        self,
        name: str,
        annotator: Annotator,
        cache_path: Optional[Path] = None,
    ) -> None:
        super().__init__(name=name)
        self.annotator = annotator
        self._cache = DiskCache(cache_path) if cache_path else NullCache()

    def _run(self, ctx: EvalContext) -> Safety:
        prompt = TextPrompt(text=ctx.prompt)
        response = SUTResponse(text=ctx.response)
        annotation = self.annotator.process(prompt, response)
        return Safety(is_safe=annotation.is_safe)

    def run(self, ctx: EvalContext) -> NodeOutput:
        key = (self.name, ctx.prompt, ctx.response)
        if key in self._cache:
            val = self._cache[key]
            assert isinstance(val, Safety)
            return NodeOutput(value=val)
        else:
            val = self._run(ctx)
            self._cache[key] = val
            return NodeOutput(value=val)
