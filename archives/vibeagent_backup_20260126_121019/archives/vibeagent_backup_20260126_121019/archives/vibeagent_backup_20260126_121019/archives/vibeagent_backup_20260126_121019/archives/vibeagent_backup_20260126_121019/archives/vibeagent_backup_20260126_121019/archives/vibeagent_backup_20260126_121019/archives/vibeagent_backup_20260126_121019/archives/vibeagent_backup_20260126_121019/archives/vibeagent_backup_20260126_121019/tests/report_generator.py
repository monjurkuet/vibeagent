import json
from datetime import datetime
from typing import Dict, Any, List


def generate_json_report(results: Dict[str, Any], output_path: str) -> None:
    summary = generate_summary_statistics(results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "model_results": {},
    }

    for model_id, model_result in results.get("models", {}).items():
        report["model_results"][model_id] = {
            "model": model_result.model,
            "model_id": model_result.model_id,
            "total_tests": model_result.total_tests,
            "passed": model_result.passed,
            "failed": model_result.failed,
            "average_response_time": model_result.average_response_time,
            "supports_tool_calling": model_result.supports_tool_calling,
            "success_rate": (model_result.passed / model_result.total_tests * 100)
            if model_result.total_tests > 0
            else 0,
            "detailed_results": [
                {
                    "test_case": result.test_case,
                    "success": result.success,
                    "passed": result.passed,
                    "response_time": result.response_time,
                    "tool_calls": result.tool_calls,
                    "error": result.error,
                }
                for result in model_result.results
            ],
            "errors": model_result.errors,
        }

    top_models = sorted(
        results.get("models", {}).items(),
        key=lambda x: (x[1].passed / x[1].total_tests * 100)
        if x[1].total_tests > 0
        else 0,
        reverse=True,
    )[:10]

    report["top_performing_models"] = [
        {
            "model": model_id,
            "success_rate": (result.passed / result.total_tests * 100)
            if result.total_tests > 0
            else 0,
            "passed": result.passed,
            "failed": result.failed,
            "average_response_time": result.average_response_time,
        }
        for model_id, result in top_models
    ]

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def generate_console_report(results: Dict[str, Any]) -> None:
    summary = generate_summary_statistics(results)

    print("\n" + "=" * 80)
    print("LLM TOOL CALLING TEST SUITE REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n📊 SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total Models Tested:           {summary['total_models']}")
    print(f"Models with Tool Calling:      {summary['models_with_tool_calling']}")
    print(f"Models without Tool Calling:   {summary['models_without_tool_calling']}")
    print(f"Total Tests Executed:          {summary['total_tests']}")
    print(f"Tests Passed:                  {summary['total_passed']}")
    print(f"Tests Failed:                  {summary['total_failed']}")
    print(f"Overall Success Rate:          {summary['success_rate']:.2f}%")
    print(f"Average Response Time:         {summary['average_response_time']:.3f}s")

    top_models = sorted(
        results.get("models", {}).items(),
        key=lambda x: (x[1].passed / x[1].total_tests * 100)
        if x[1].total_tests > 0
        else 0,
        reverse=True,
    )[:10]

    print("\n🏆 TOP 10 PERFORMING MODELS")
    print("-" * 80)
    print(
        f"{'Rank':<6} {'Model':<40} {'Success':<10} {'Passed':<8} {'Failed':<8} {'Avg Time':<10}"
    )
    print("-" * 80)
    for rank, (model_id, result) in enumerate(top_models, 1):
        success_rate = (
            (result.passed / result.total_tests * 100) if result.total_tests > 0 else 0
        )
        print(
            f"{rank:<6} {model_id:<40} {success_rate:>6.2f}% {result.passed:<8} {result.failed:<8} {result.average_response_time:>8.3f}s"
        )

    no_tool_calling = [
        model_id
        for model_id, result in results.get("models", {}).items()
        if not result.supports_tool_calling
    ]

    if no_tool_calling:
        print("\n❌ MODELS NOT SUPPORTING TOOL CALLING")
        print("-" * 80)
        for model_id in no_tool_calling:
            print(f"  • {model_id}")

    print("\n" + "=" * 80)
    print("Detailed report saved to: ./test_results.json")
    print("=" * 80 + "\n")


def generate_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    models = results.get("models", {})

    total_models = len(models)
    models_with_tool_calling = sum(
        1 for m in models.values() if m.supports_tool_calling
    )
    models_without_tool_calling = total_models - models_with_tool_calling

    total_tests = sum(m.total_tests for m in models.values())
    total_passed = sum(m.passed for m in models.values())
    total_failed = sum(m.failed for m in models.values())

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    average_response_time = (
        sum(m.average_response_time for m in models.values()) / total_models
        if total_models > 0
        else 0
    )

    return {
        "total_models": total_models,
        "models_with_tool_calling": models_with_tool_calling,
        "models_without_tool_calling": models_without_tool_calling,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "success_rate": success_rate,
        "average_response_time": average_response_time,
    }
