#!/usr/bin/env python3
"""
Quick script to run tool calling tests with detailed logs
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tool_test.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Run tool calling tests with detailed logging."""
    logger.info("=" * 70)
    logger.info("Starting VibeAgent Tool Calling Tests")
    logger.info("=" * 70)

    try:
        from llm_tool_calling_tester import LLMToolCallingTester
        from test_cases import TEST_CASES
        import report_generator

        # Initialize tester
        logger.info("Initializing LLM Tool Calling Tester...")
        base_url = "http://localhost:8087/v1"
        tester = LLMToolCallingTester(base_url=base_url)

        # Fetch available models
        logger.info("Fetching available models...")
        all_models = tester._fetch_models()
        model_ids = [
            m.get("id", m.get("name", ""))
            for m in all_models
            if m.get("id") or m.get("name")
        ]

        if not model_ids:
            logger.error(
                "No models found! Make sure LLM API is running at %s", base_url
            )
            return 1

        logger.info(f"Found {len(model_ids)} models: {model_ids}")

        # Test first model with all test cases
        model_to_test = model_ids[0]
        logger.info(f"\nTesting model: {model_to_test}")
        logger.info(f"Running {len(TEST_CASES)} test cases...\n")

        results = tester.test_model_tool_calling(model_to_test, TEST_CASES)

        # Generate console report
        logger.info("\n" + "=" * 70)
        logger.info("Test Results")
        logger.info("=" * 70)
        report_generator.generate_console_report({"models": {model_to_test: results}})

        # Generate JSON report
        output_file = "tool_test_results.json"
        report_generator.generate_json_report(
            {"models": {model_to_test: results}}, output_file
        )
        logger.info(f"\nJSON report saved to: {output_file}")

        # Summary
        passed = sum(1 for r in results if r.get("success", False))
        total = len(results)
        logger.info(
            f"\nSummary: {passed}/{total} tests passed ({passed / total * 100:.1f}%)"
        )

        return 0 if passed == total else 1

    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
