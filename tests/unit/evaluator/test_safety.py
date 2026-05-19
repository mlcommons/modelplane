from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotators.demo_annotator import DemoYBadAnnotator
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse
import pytest

from modelplane.evaluator.dag import Composer
from modelplane.evaluator.safety import AnnotatorArbiter, Safety, SafetyDAGAnnotator
from modelplane.evaluator.verdict import Verdict


def test_safety_annotator_arbiter(sample_ctx):
    annotator = DemoYBadAnnotator("demo_annotator")
    arbiter = AnnotatorArbiter(name="demo_arbiter", annotator=annotator)
    output = arbiter.run(sample_ctx)
    assert output.value.is_safe
    assert isinstance(output.value, Safety)
    assert arbiter.verdict_type == Safety


def test_safety_dag_run(simple_dag, sample_ctx):
    safety_annotator = SafetyDAGAnnotator("safety", simple_dag)
    output = safety_annotator.process(
        prompt=TextPrompt(text=sample_ctx.prompt),
        response=SUTResponse(text=sample_ctx.response),
    )
    assert not output.is_safe
    assert isinstance(output, SafetyAnnotation)


def test_safety_dag_with_bad_verdict_type():
    with pytest.raises(
        ValueError,
        match="All outputs of the DAG must be of type Safety.",
    ):
        SafetyDAGAnnotator("bad_dag", Composer("bad_dag", verdict_type=Verdict))
