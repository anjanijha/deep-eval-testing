from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

# ============================================================
# Individual test-case thresholds
# ============================================================

FAITHFULNESS_THRESHOLD = 0.80
ANSWER_RELEVANCY_THRESHOLD = 0.80
CONTEXT_RELEVANCY_THRESHOLD = 0.80


# ============================================================
# Overall quality-gate thresholds
# ============================================================

OVERALL_PASS_RATE_THRESHOLD = 0.80

MIN_AVERAGE_FAITHFULNESS = 0.85
MIN_AVERAGE_ANSWER_RELEVANCY = 0.85
MIN_AVERAGE_CONTEXT_RELEVANCY = 0.80


def get_rag_metrics():
    """
    Returns the DeepEval metrics used for RAG evaluation.
    """

    return [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD),
        AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD),
        ContextualRelevancyMetric(threshold=CONTEXT_RELEVANCY_THRESHOLD),
    ]
