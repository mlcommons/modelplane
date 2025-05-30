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
    """Simple random implementation for demonstration. Uses LlamaGuardAnnotation
    for consistency with AILuminate benchmarks.

    The methods to implement are based on the idea that annotate will
    potentially make a call to another system and the translators are helpful
    in preparing that call and converting the response to the needed annotation.
    """

    # Probability of unsafe per category
    PROB_UNSAFE = 0.01

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        return AnnotationRequest(test_item=test_item, response=response)

    def annotate(self, annotation_request: AnnotationRequest) -> LlamaGuardAnnotation:
        unsafe_categories = []
        for category in BaseSafeTestVersion1.hazards:
            if random.random() < self.PROB_UNSAFE:
                unsafe_categories.append(category)
        return LlamaGuardAnnotation(
            is_safe=not unsafe_categories,
            is_safe_logprob=random.random(),
            violation_categories=unsafe_categories,
        )

    def translate_response(
        self, request: AnnotationRequest, response: LlamaGuardAnnotation
    ) -> LlamaGuardAnnotation:
        return response


random.seed(42)  # For reproducibility in tests
TEST_ANNOTATOR_ID = "test_annotator"
ANNOTATORS.register(TestAnnotator, TEST_ANNOTATOR_ID)
