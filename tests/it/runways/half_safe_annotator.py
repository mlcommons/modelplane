import random

from pydantic import BaseModel

from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse
from modelgauge.tests.safe_v1 import BaseSafeTestVersion1


class AnnotationRequest(BaseModel):
    test_item: TestItem
    response: SUTResponse


class TestAnnotator(CompletionAnnotator[LlamaGuardAnnotation]):
    """Simple implementation for demonstration. Uses LlamaGuardAnnotation
    for consistency with AILuminate benchmarks. Every other item is marked safe.

    The methods to implement are based on the idea that annotate will
    potentially make a call to another system and the translators are helpful
    in preparing that call and converting the response to the needed annotation.
    """

    def __init__(self, uid):
        super().__init__(uid)
        self.next_is_safe = True

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        return AnnotationRequest(test_item=test_item, response=response)

    def annotate(self, annotation_request: AnnotationRequest) -> LlamaGuardAnnotation:
        unsafe_categories = []
        if not self.next_is_safe:
            unsafe_categories =  BaseSafeTestVersion1.hazards
        annotation = LlamaGuardAnnotation(
            is_safe=self.next_is_safe,
            is_safe_logprob=random.random(),
            violation_categories=unsafe_categories,
        )
        self.next_is_safe = not self.next_is_safe
        return annotation

    def translate_response(
        self, request: AnnotationRequest, response: LlamaGuardAnnotation
    ) -> LlamaGuardAnnotation:
        return response


TEST_ANNOTATOR_ID = "test_annotator"
ANNOTATORS.register(TestAnnotator, TEST_ANNOTATOR_ID)
