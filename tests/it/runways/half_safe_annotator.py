from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.sut import SUTResponse
from pydantic import BaseModel


class AnnotationRequest(BaseModel):
    prompt: ChatPrompt | TextPrompt
    response: SUTResponse


class TestAnnotator(Annotator):
    """Simple implementation for demonstration.

    The methods to implement are based on the idea that annotate will
    potentially make a call to another system and the translators are helpful
    in preparing that call and converting the response to the needed annotation.
    """

    def __init__(self, uid):
        super().__init__(uid)
        self.next_is_safe = True

    def translate_prompt(self, prompt: ChatPrompt | TextPrompt, response: SUTResponse):
        return AnnotationRequest(
            prompt=prompt,
            response=response,
        )

    def annotate(self, annotation_request: AnnotationRequest) -> SafetyAnnotation:
        annotation = SafetyAnnotation(
            is_safe=self.next_is_safe,
        )
        self.next_is_safe = not self.next_is_safe
        return annotation

    def translate_response(
        self, request: AnnotationRequest, response: SafetyAnnotation
    ) -> SafetyAnnotation:
        return response


TEST_ANNOTATOR_ID = "test_annotator"
ANNOTATORS.register(TestAnnotator, TEST_ANNOTATOR_ID)
