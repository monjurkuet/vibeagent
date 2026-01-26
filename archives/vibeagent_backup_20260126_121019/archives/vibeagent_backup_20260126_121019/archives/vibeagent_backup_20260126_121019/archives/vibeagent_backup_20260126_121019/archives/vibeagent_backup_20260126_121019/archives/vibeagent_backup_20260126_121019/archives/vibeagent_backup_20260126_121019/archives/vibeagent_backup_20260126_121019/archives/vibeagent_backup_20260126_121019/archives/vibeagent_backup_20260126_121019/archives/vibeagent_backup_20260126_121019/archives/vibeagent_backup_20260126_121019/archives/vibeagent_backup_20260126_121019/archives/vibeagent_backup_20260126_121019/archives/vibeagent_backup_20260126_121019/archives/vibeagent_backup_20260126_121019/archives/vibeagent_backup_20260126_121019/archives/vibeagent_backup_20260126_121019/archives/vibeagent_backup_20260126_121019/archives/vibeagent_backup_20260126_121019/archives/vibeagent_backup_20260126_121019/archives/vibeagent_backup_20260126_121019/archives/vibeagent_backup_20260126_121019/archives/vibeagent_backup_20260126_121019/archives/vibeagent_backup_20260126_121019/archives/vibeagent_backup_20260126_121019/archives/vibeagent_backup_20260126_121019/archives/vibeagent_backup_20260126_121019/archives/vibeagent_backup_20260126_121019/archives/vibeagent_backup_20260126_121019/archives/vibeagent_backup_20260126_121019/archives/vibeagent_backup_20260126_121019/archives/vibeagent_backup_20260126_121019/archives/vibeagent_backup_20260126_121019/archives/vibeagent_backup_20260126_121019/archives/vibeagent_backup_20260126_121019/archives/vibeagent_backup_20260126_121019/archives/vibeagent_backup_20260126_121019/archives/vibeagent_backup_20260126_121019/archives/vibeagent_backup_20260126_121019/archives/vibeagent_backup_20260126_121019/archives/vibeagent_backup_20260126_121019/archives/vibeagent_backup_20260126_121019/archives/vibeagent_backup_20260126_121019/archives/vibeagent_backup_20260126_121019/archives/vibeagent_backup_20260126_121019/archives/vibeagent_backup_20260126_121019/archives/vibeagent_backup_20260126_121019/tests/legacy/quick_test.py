#!/usr/bin/env python3
"""
Quick start script for LLM tool calling tests.
This provides an easy interface to run tests with various options.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.llm_tool_calling_tester import LLMToolCallingTester
from tests.test_cases import TEST_CASES
from tests.report_generator import (
    generate_json_report,
    generate_console_report,
    generate_summary_statistics,
)


def run_tests(args):
    """Run the test suite with given arguments."""

    print("\n" + "=" * 80)
    print("🧪 LLM TOOL CALLING TEST SUITE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base URL: {args.base_url}")
    print(f"  Output: {args.output}")
    print(f"  Parallel: {'Yes' if args.parallel else 'No'}")
    print(f"  LLM Judge: {'Yes' if args.use_llm_judge else 'No (exact matching)'}")
    if args.use_llm_judge:
        print(f"  Judge Model: {args.judge_model}")
    if args.models:
        print(f"  Models: {', '.join(args.models)}")
    else:
        print(f"  Models: ALL (will fetch from API)")
    print()

    # Initialize tester
    tester = LLMToolCallingTester(
        base_url=args.base_url,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
    )

    # Fetch models
    try:
        models = tester._fetch_models()
        model_ids = [m["id"] for m in models]
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}")
        return 1

    # Filter models if specified
    if args.models:
        model_ids = [m for m in model_ids if m in args.models]
        if not model_ids:
            print(f"❌ No matching models found for: {args.models}")
            return 1
        print(f"✓ Testing {len(model_ids)} specified models")
    else:
        print(f"✓ Testing all {len(model_ids)} available models")

    print()

    # Run tests
    all_results = {}
    total_tests = len(model_ids) * len(TEST_CASES)
    current_test = 0

    for i, model_id in enumerate(model_ids, 1):
        print(f"\n📊 Progress: [{i}/{len(model_ids)}] Testing model: {model_id}")
        print(f"   Overall progress: {current_test}/{total_tests} tests completed")

        try:
            result = tester.test_model_tool_calling(model_id, TEST_CASES)
            all_results[model_id] = result
            current_test += len(TEST_CASES)
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            print(f"   Completed {current_test}/{total_tests} tests")
            break
        except Exception as e:
            print(f"❌ Error testing {model_id}: {e}")
            continue

    # Generate reports
    if all_results:
        print("\n" + "=" * 80)
        print("📊 GENERATING REPORTS")
        print("=" * 80)

        # Convert to expected format
        results_list = []
        for model_id, result in all_results.items():
            results_list.append(
                {
                    "model": model_id,
                    "supports_tool_calling": result.supports_tool_calling,
                    "metrics": {
                        "total_tests": result.total_tests,
                        "passed": result.passed,
                        "failed": result.failed,
                        "avg_response_time": result.average_response_time,
                        "errors": result.errors,
                    },
                    "tests": [
                        {
                            "test_name": tr.test_case,
                            "passed": tr.passed,
                            "response_time": tr.response_time,
                            "tool_calls": tr.tool_calls,
                            "error": tr.error,
                        }
                        for tr in result.results
                    ],
                }
            )

        # Generate summary stats
        summary = generate_summary_statistics(results_list)

        # Save JSON report
        import json
        from datetime import datetime

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "results": results_list,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"✓ JSON report saved to: {output_path}")

        # Print console report
        generate_console_report(results_list)

        # Final summary
        print("\n" + "=" * 80)
        print("🎉 TEST COMPLETED")
        print("=" * 80)
        print(f"Total models tested: {len(all_results)}")
        print(f"Total tests run: {current_test}")
        print(f"Models with tool calling: {summary['models_with_tool_calling']}")
        print(f"Models without tool calling: {summary['models_without_tool_calling']}")
        print(f"Overall success rate: {summary['success_rate']:.1f}%")
        print(f"Average response time: {summary['average_response_time']:.2f}s")
        print(f"\n📄 Detailed report: {output_path}")
        print("=" * 80 + "\n")

        return 0
    else:
        print("\n❌ No test results to report")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test LLM tool calling capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with LLM judge (semantic verification)
  python quick_test.py

  # Test with exact matching (no LLM judge)
  python quick_test.py --no-llm-judge

  # Test specific models
  python quick_test.py --models glm-4.7 deepseek-v3.2

  # Use different judge model
  python quick_test.py --judge-model gemini-3-flash-preview

  # Test with custom output
  python quick_test.py --output my_report.json

  # Test against different endpoint
  python quick_test.py --base-url http://localhost:8080/v1
        """,
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8087/v1",
        help="LLM API base URL (default: http://localhost:8087/v1)",
    )

    parser.add_argument(
        "--models", nargs="+", help="Specific models to test (default: all models)"
    )

    parser.add_argument(
        "--output",
        default="tests/results/quick_test_report.json",
        help="Output JSON report path (default: tests/results/quick_test_report.json)",
    )

    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel (experimental)"
    )

    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=True,
        help="Use LLM judge for semantic verification (default: True)",
    )

    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM judge, use exact matching instead",
    )

    parser.add_argument(
        "--judge-model",
        default="iflow-rome-30ba3b",
        help="Model to use as LLM judge (default: gemini-2.5-flash)",
    )

    args = parser.parse_args()

    # Handle --no-llm-judge flag
    if args.no_llm_judge:
        args.use_llm_judge = False

    try:
        return run_tests(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
