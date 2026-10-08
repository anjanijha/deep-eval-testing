import pytest
from deepeval import assert_test
from app.rag_app import ask
from evaluation.dataset_loader import load_golden_dataset
from evaluation.evaluator import create_test_case
from evaluation.metrics import get_rag_metrics

test_data = load_golden_dataset()
metrics = get_rag_metrics()


@pytest.mark.parametrize("test_data", test_data, ids=lambda data: data["id"])
def test_rag(test_data):
    result = ask(test_data["question"])
    test_case = create_test_case(test_data, result)
    assert_test(test_case=test_case, metrics=metrics)
