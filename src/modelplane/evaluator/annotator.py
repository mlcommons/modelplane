from modelgauge.annotator import Annotator
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTResponse

from modelplane.evaluator.context import EvalContext
from modelplane.evaluator.dag import Composer, SuccessfulDAGOutput
from modelplane.evaluator.verdict import Verdict


class DAGAnnotator(Annotator):
    """Annotator that executes a DAG."""

    def __init__(self, uid: str, dag: Composer) -> None:
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

    def annotate(self, annotation_request: EvalContext) -> Verdict:
        dag_output = self.dag.run(annotation_request)
        if isinstance(dag_output, SuccessfulDAGOutput):
            return dag_output.verdict
        else:
            raise dag_output.error
