from deepeval.test_case import LLMTestCase


def create_test_case(test_data: dict, application_result: dict) -> LLMTestCase:

    return LLMTestCase(
        input=test_data["question"],
        actual_output=application_result["answer"],
        expected_output=test_data["expected_answer"],
        retrieval_context=application_result["context"],
    )
