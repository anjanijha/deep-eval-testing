import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

from app.rag_app import ask
from evaluation.dataset_loader import load_golden_dataset

test_data = load_golden_dataset()

faithfulness = FaithfulnessMetric(threshold=0.8)


@pytest.mark.parametrize("test_data", test_data, ids=lambda data: data["id"])
def test_no_hallucination(test_data):

    result = ask(test_data["question"])

    test_case = LLMTestCase(
        input=test_data["question"],
        actual_output=result["answer"],
        retrieval_context=result["context"],
    )

    assert_test(test_case=test_case, metrics=[faithfulness])
