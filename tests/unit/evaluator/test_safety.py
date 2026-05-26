import pytest
from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotators.demo_annotator import DemoYBadAnnotator
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse

from modelplane.evaluator.dag import Composer, FailedDAGOutput
from modelplane.evaluator.safety import AnnotatorArbiter, Safety, SafetyDAGAnnotator
from modelplane.evaluator.verdict import Verdict

from .mocks import FailingNode


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


def test_safety_dag_with_bad_node(sample_ctx, threshold_arbiter):
    failing_node = FailingNode(name="failing_node", routes=["threshold_arbiter"])
    dag = (
        Composer(
            "bad_node_dag",
            verdict_type=Safety,
        )
        .add_node(failing_node)
        .add_node(threshold_arbiter)
    )
    dag_output = dag.run(sample_ctx)
    assert isinstance(dag_output, FailedDAGOutput)
    assert str(dag_output.error) == "I'm afraid I can't do that, Dave."

    dag_annotator = SafetyDAGAnnotator("safety_annotator", dag)
    with pytest.raises(
        type(dag_output.error), match="I'm afraid I can't do that, Dave."
    ):
        dag_annotator.process(
            prompt=TextPrompt(text=sample_ctx.prompt),
            response=SUTResponse(text=sample_ctx.response),
        )
