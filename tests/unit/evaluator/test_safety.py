from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotators.demo_annotator import DemoYBadAnnotator
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse

from modelplane.evaluator.safety import AnnotatorArbiter, Safety, SafetyDAGAnnotator


def test_safety_annotator_arbiter(sample_ctx):
    annotator = DemoYBadAnnotator("demo_annotator")
    arbiter = AnnotatorArbiter(name="demo_arbiter", annotator=annotator)
    output = arbiter.run(sample_ctx)
    assert output.is_safe
    assert isinstance(output, Safety)
    assert arbiter.output_type == Safety


def test_safety_dag_run(simple_dag, sample_ctx):
    safety_annotator = SafetyDAGAnnotator("safety", simple_dag)
    output = safety_annotator.process(
        prompt=TextPrompt(text=sample_ctx.prompt),
        response=SUTResponse(text=sample_ctx.response),
    )
    assert not output.is_safe
    assert isinstance(output, SafetyAnnotation)
