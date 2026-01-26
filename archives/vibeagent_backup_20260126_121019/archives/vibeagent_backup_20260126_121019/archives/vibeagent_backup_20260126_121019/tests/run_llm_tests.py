#!/usr/bin/env python3
import argparse
import sys
import traceback
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_tool_calling_tester import LLMToolCallingTester
from test_cases import TEST_CASES
import report_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM tool calling tests against available models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: all available models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Custom output path for the JSON report (default: test_results.json)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel across models",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8087/v1",
        help="Base URL for the LLM API (default: http://localhost:8087/v1)",
    )
    return parser.parse_args()


def test_single_model(
    tester: LLMToolCallingTester, model_id: str, test_cases: List[Dict[str, Any]]
) -> tuple[str, Any]:
    """Test a single model and return its results."""
    results = tester.test_model_tool_calling(model_id, test_cases)
    return model_id, results


def run_tests_sequential(
    tester: LLMToolCallingTester,
    models: List[str],
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run tests sequentially on each model."""
    all_results = {}

    for idx, model_id in enumerate(models, 1):
        print(f"[{idx}/{len(models)}] Testing model: {model_id}")
        try:
            model_id, results = test_single_model(tester, model_id, test_cases)
            all_results[model_id] = results
        except Exception as e:
            print(f"  Error testing {model_id}: {e}")
            all_results[model_id] = None

    return all_results


def run_tests_parallel(
    tester: LLMToolCallingTester,
    models: List[str],
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run tests in parallel across models."""
    all_results = {}
    completed_count = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_model = {
            executor.submit(test_single_model, tester, model_id, test_cases): model_id
            for model_id in models
        }

        for future in as_completed(future_to_model):
            completed_count += 1
            model_id = future_to_model[future]
            try:
                model_id, results = future.result()
                print(f"[{completed_count}/{len(models)}] Tested model: {model_id}")
                all_results[model_id] = results
            except Exception as e:
                print(
                    f"[{completed_count}/{len(models)}] Error testing {model_id}: {e}"
                )
                all_results[model_id] = None

    return all_results


def main() -> int:
    args = parse_args()

    try:
        print("Initializing LLM Tool Calling Tester...")
        tester = LLMToolCallingTester(base_url=args.base_url)

        print("Fetching available models...")
        all_models = tester._fetch_models()
        model_ids = [
            m.get("id", m.get("name", ""))
            for m in all_models
            if m.get("id") or m.get("name")
        ]

        if not model_ids:
            print("No models found!")
            return 1

        if args.models:
            selected_models = [m for m in model_ids if m in args.models]
            if not selected_models:
                print(
                    f"None of the specified models found. Available models: {model_ids}"
                )
                return 1
            models_to_test = selected_models
        else:
            models_to_test = model_ids

        print(
            f"Testing {len(models_to_test)} models with {len(TEST_CASES)} test cases each...\n"
        )

        if args.parallel:
            print("Running tests in parallel...")
            results = run_tests_parallel(tester, models_to_test, TEST_CASES)
        else:
            results = run_tests_sequential(tester, models_to_test, TEST_CASES)

        print("\nGenerating reports...")

        results_dict = {"models": {k: v for k, v in results.items() if v is not None}}

        report_generator.generate_console_report(results_dict)
        report_generator.generate_json_report(results_dict, args.output)

        print(f"\nJSON report saved to: {args.output}")

        summary = report_generator.generate_summary_statistics(results_dict)
        total_failed = summary.get("total_failed", 0)
        if total_failed > 0:
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during test execution: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
