import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.rag_app import ask
from evaluation.dataset_loader import load_golden_dataset
from evaluation.evaluator import create_test_case
from evaluation.metrics import (
    get_rag_metrics,
    OVERALL_PASS_RATE_THRESHOLD,
    MIN_AVERAGE_FAITHFULNESS,
    MIN_AVERAGE_ANSWER_RELEVANCY,
    MIN_AVERAGE_CONTEXT_RELEVANCY,
)

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_FILE = REPORT_DIR / "evaluation_summary.json"


# ============================================================
# Evaluate one test case
# ============================================================


def evaluate_test_case(test_data: dict) -> dict:

    result = ask(test_data["question"])

    test_case = create_test_case(test_data, result)

    metrics = get_rag_metrics()

    metric_results = []

    for metric in metrics:

        try:
            # Execute the DeepEval metric
            metric.measure(test_case)

            score = metric.score

            threshold = metric.threshold

            passed = score is not None and score >= threshold

            metric_results.append(
                {
                    "metric": metric.__class__.__name__,
                    "score": (round(score, 4) if score is not None else None),
                    "threshold": threshold,
                    "passed": passed,
                    "reason": getattr(metric, "reason", None),
                }
            )

        except Exception as error:

            metric_results.append(
                {
                    "metric": metric.__class__.__name__,
                    "score": None,
                    "threshold": getattr(metric, "threshold", None),
                    "passed": False,
                    "reason": f"Metric execution failed: {error}",
                }
            )

    # Test case passes only when ALL metrics pass
    test_passed = all(metric["passed"] for metric in metric_results)

    return {
        "id": test_data["id"],
        "category": test_data.get("category", "unknown"),
        "question": test_data["question"],
        "passed": test_passed,
        "metrics": metric_results,
    }


# ============================================================
# Calculate average metric scores
# ============================================================


def calculate_average_metrics(results: list[dict]) -> dict:

    metric_scores = {}

    for result in results:

        for metric in result["metrics"]:

            metric_name = metric["metric"]

            score = metric["score"]

            if score is not None:

                metric_scores.setdefault(metric_name, []).append(score)

    average_scores = {}

    for metric_name, scores in metric_scores.items():

        if scores:

            average_scores[metric_name] = round(sum(scores) / len(scores), 4)

        else:

            average_scores[metric_name] = None

    return average_scores


# ============================================================
# Check average metric quality gates
# ============================================================


def check_metric_quality_gates(average_scores: dict) -> dict:

    faithfulness = average_scores.get("FaithfulnessMetric")

    answer_relevancy = average_scores.get("AnswerRelevancyMetric")

    contextual_relevancy = average_scores.get("ContextualRelevancyMetric")

    faithfulness_passed = (
        faithfulness is not None and faithfulness >= MIN_AVERAGE_FAITHFULNESS
    )

    answer_relevancy_passed = (
        answer_relevancy is not None
        and answer_relevancy >= MIN_AVERAGE_ANSWER_RELEVANCY
    )

    contextual_relevancy_passed = (
        contextual_relevancy is not None
        and contextual_relevancy >= MIN_AVERAGE_CONTEXT_RELEVANCY
    )

    return {
        "faithfulness": {
            "score": faithfulness,
            "threshold": MIN_AVERAGE_FAITHFULNESS,
            "passed": faithfulness_passed,
        },
        "answer_relevancy": {
            "score": answer_relevancy,
            "threshold": MIN_AVERAGE_ANSWER_RELEVANCY,
            "passed": answer_relevancy_passed,
        },
        "contextual_relevancy": {
            "score": contextual_relevancy,
            "threshold": MIN_AVERAGE_CONTEXT_RELEVANCY,
            "passed": contextual_relevancy_passed,
        },
    }


# ============================================================
# Main quality gate
# ============================================================


def run_quality_gate():

    print()
    print("=" * 70)
    print("STARTING LLM QUALITY GATE")
    print("=" * 70)

    # --------------------------------------------------------
    # Create report directory
    # --------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load golden dataset
    # --------------------------------------------------------

    dataset = load_golden_dataset()

    print(f"\nLoaded {len(dataset)} test cases.")

    # --------------------------------------------------------
    # Execute test cases
    # --------------------------------------------------------

    results = []

    for test_data in dataset:

        print(f"\nEvaluating " f"{test_data['id']}...")

        result = evaluate_test_case(test_data)

        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"

        print(f"{test_data['id']} " f"-> {status}")

        # Print individual metrics

        for metric in result["metrics"]:

            score = metric["score"]

            threshold = metric["threshold"]

            metric_status = "PASS" if metric["passed"] else "FAIL"

            score_text = f"{score:.4f}" if score is not None else "N/A"

            print(
                f"    "
                f"{metric['metric']}: "
                f"{score_text} "
                f"(threshold={threshold}) "
                f"-> {metric_status}"
            )

    # --------------------------------------------------------
    # Test statistics
    # --------------------------------------------------------

    total_tests = len(results)

    passed_tests = sum(1 for result in results if result["passed"])

    failed_tests = total_tests - passed_tests

    pass_rate = passed_tests / total_tests if total_tests > 0 else 0

    # --------------------------------------------------------
    # Average metric scores
    # --------------------------------------------------------

    average_scores = calculate_average_metrics(results)

    # --------------------------------------------------------
    # Metric quality gates
    # --------------------------------------------------------

    metric_gates = check_metric_quality_gates(average_scores)

    metric_gate_passed = all(gate["passed"] for gate in metric_gates.values())

    # --------------------------------------------------------
    # Overall pass-rate gate
    # --------------------------------------------------------

    pass_rate_gate_passed = pass_rate >= OVERALL_PASS_RATE_THRESHOLD

    # --------------------------------------------------------
    # Final quality gate
    # --------------------------------------------------------

    quality_gate_passed = pass_rate_gate_passed and metric_gate_passed

    quality_gate_status = "PASS" if quality_gate_passed else "FAIL"

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "pass_rate": round(pass_rate, 4),
        "required_pass_rate": (OVERALL_PASS_RATE_THRESHOLD),
        "pass_rate_gate": {
            "actual": round(pass_rate, 4),
            "required": (OVERALL_PASS_RATE_THRESHOLD),
            "passed": pass_rate_gate_passed,
        },
        "average_metrics": average_scores,
        "metric_quality_gates": metric_gates,
        "quality_gate": quality_gate_status,
        "test_results": results,
    }

    # --------------------------------------------------------
    # Write JSON report
    # --------------------------------------------------------

    with open(REPORT_FILE, "w", encoding="utf-8") as file:

        json.dump(summary, file, indent=4)

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 70)
    print("LLM QUALITY GATE SUMMARY")
    print("=" * 70)

    print(f"Total tests        : {total_tests}")

    print(f"Passed tests       : {passed_tests}")

    print(f"Failed tests       : {failed_tests}")

    print(f"Pass rate          : " f"{pass_rate:.2%}")

    print(f"Required pass rate : " f"{OVERALL_PASS_RATE_THRESHOLD:.2%}")

    print()
    print("Average Metrics")
    print("-" * 70)

    for metric_name, score in average_scores.items():

        print(f"{metric_name:<35}: " f"{score if score is not None else 'N/A'}")

    print()
    print("Metric Quality Gates")
    print("-" * 70)

    for name, gate in metric_gates.items():

        status = "PASS" if gate["passed"] else "FAIL"

        score = gate["score"]

        threshold = gate["threshold"]

        print(f"{name:<25}: " f"{score} >= {threshold} " f"-> {status}")

    print()
    print("=" * 70)

    print(f"FINAL QUALITY GATE: " f"{quality_gate_status}")

    print("=" * 70)

    print(f"\nJSON report:" f"\n{REPORT_FILE}")

    # ========================================================
    # CI/CD exit code
    # ========================================================

    if not quality_gate_passed:

        print("\n❌ QUALITY GATE FAILED")

        sys.exit(1)

    print("\n✅ QUALITY GATE PASSED")

    sys.exit(0)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    run_quality_gate()
