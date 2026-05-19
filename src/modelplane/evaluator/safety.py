from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator, SUTResponse, TextPrompt

from modelplane.evaluator.annotator import DAGAnnotator
from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.dag import Composer
from modelplane.evaluator.nodes import Arbiter, CacheableComposerNode, NodeOutput
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

    def __init__(self, uid: str, dag: Composer) -> None:
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


class AnnotatorArbiter(SafetyArbiter, CacheableComposerNode):
    """Arbiter that outputs SAFE or UNSAFE based on the output of a (safety) Annotator."""

    def __init__(self, name: str, annotator: Annotator) -> None:
        super().__init__(name=name)
        self.annotator = annotator

    def cache_key(self, ctx: EvalContext) -> tuple:
        return ctx.prompt, ctx.response

    def run(self, ctx: EvalContext) -> Safety:
        prompt = TextPrompt(text=ctx.prompt)
        response = SUTResponse(text=ctx.response)
        annotation = self.annotator.process(prompt, response)
        val = Safety(is_safe=annotation.is_safe)
        return NodeOutput(value=val, original_ctx=ctx)
