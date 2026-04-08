from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator, SUTResponse, TextPrompt

from modelplane.evaluator.annotator import DAGAnnotator
from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.dag import EvaluatorDAG
from modelplane.evaluator.nodes import Arbiter
from modelplane.evaluator.outputs import Output


class Safety(Output):

    def __init__(self, is_safe: bool) -> None:
        self.is_safe = is_safe

    @property
    def name(self) -> str:
        return "SAFE" if self.is_safe else "UNSAFE"


SAFE = Safety(is_safe=True)
UNSAFE = Safety(is_safe=False)


class SafetyDAGAnnotator(DAGAnnotator):
    """Implementation of DAGAnnotator that produces a SafetyAnnotation."""

    def __init__(self, uid: str, dag: EvaluatorDAG) -> None:
        super().__init__(uid, dag)
        if not all(isinstance(o, Safety) for o in dag.outputs):
            raise ValueError("All outputs of the DAG must be of type Safety.")

    def translate_response(
        self,
        request: EvalContext,
        response: Safety,
    ) -> SafetyAnnotation:
        """Map DAGResult verdict to a SafetyAnnotation (is_safe bool)."""
        return SafetyAnnotation(is_safe=response.is_safe)


class AnnotatorArbiter(Arbiter):
    """Arbiter that outputs SAFE or UNSAFE based on the output of a (safety) Annotator."""

    def __init__(self, name: str, annotator: Annotator) -> None:
        super().__init__(name=name)
        self.annotator = annotator

    def run(self, ctx: EvalContext) -> Output:
        prompt = TextPrompt(text=ctx.prompt)
        response = SUTResponse(text=ctx.response)
        annotation = self.annotator.process(prompt, response)
        return SAFE if annotation.is_safe else UNSAFE
