"""Example demonstrating the self-correction system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from core.self_corrector import (
    SelfCorrector,
    SelfCorrectorConfig,
    CorrectionTrigger,
    CorrectionType,
)
from core.skill import SkillResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_correction():
    """Demonstrate basic self-correction flow."""
    print("\n=== Basic Self-Correction Example ===\n")

    config = SelfCorrectorConfig(
        max_correction_attempts=3,
        consecutive_error_threshold=2,
        confidence_threshold=0.5,
        enable_llm_guidance=False,
    )

    corrector = SelfCorrector(config=config)

    tool_result = SkillResult(
        success=False,
        error="Connection timeout after 30 seconds",
    )

    context = {
        "tool_result": tool_result,
        "tool_name": "api_fetcher",
        "iteration": 3,
        "consecutive_errors": 2,
        "parameters": {"url": "https://api.example.com/data"},
        "confidence": 0.4,
    }

    should_correct, trigger = corrector.should_self_correct(context)

    print(f"Should correct: {should_correct}")
    print(f"Trigger: {trigger.value if trigger else None}")

    if should_correct:
        reflection = corrector.reflect_on_failure(tool_result.error, context)

        print(f"\nReflection:")
        print(f"  Error pattern: {reflection['error_pattern']}")
        print(f"  Root causes: {reflection['root_causes']}")

        alternatives = corrector.generate_alternatives(context)

        print(f"\nGenerated {len(alternatives)} alternative strategies:")
        for i, strategy in enumerate(alternatives, 1):
            print(f"  {i}. {strategy.strategy_type.value}: {strategy.description}")
            print(f"     Confidence: {strategy.confidence_score:.2f}")

        best_strategy = corrector.select_best_strategy(alternatives)

        if best_strategy:
            print(f"\nSelected best strategy: {best_strategy.strategy_type.value}")

            result = corrector.apply_correction(best_strategy, context)

            print(f"Correction result: {'Success' if result.success else 'Failed'}")
            if result.error:
                print(f"Error: {result.error}")

    stats = corrector.get_correction_statistics()
    print(f"\nCorrection statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")


def example_error_pattern_detection():
    """Demonstrate error pattern detection."""
    print("\n=== Error Pattern Detection Example ===\n")

    corrector = SelfCorrector()

    test_errors = [
        "Network connection refused",
        "Request timed out after 30s",
        "Rate limit exceeded (429)",
        "Invalid input parameter",
        "Authentication failed: invalid token",
        "Resource not found (404)",
        "Permission denied (403)",
        "Internal server error (500)",
        "Unknown error occurred",
    ]

    for error in test_errors:
        pattern = corrector.get_error_pattern(error)
        print(f"{error:40s} -> {pattern.value}")


def example_confidence_scoring():
    """Demonstrate confidence scoring."""
    print("\n=== Confidence Scoring Example ===\n")

    corrector = SelfCorrector()

    scenarios = [
        {"consecutive_errors": 0, "total_errors": 0, "iteration": 1},
        {"consecutive_errors": 1, "total_errors": 2, "iteration": 3},
        {"consecutive_errors": 2, "total_errors": 5, "iteration": 5},
        {"consecutive_errors": 3, "total_errors": 8, "iteration": 8},
    ]

    for scenario in scenarios:
        confidence = corrector.calculate_confidence(scenario)
        print(f"Scenario: {scenario}")
        print(f"  Confidence: {confidence:.2f}\n")


def example_learning_from_corrections():
    """Demonstrate learning from past corrections."""
    print("\n=== Learning from Corrections Example ===\n")

    config = SelfCorrectorConfig(
        enable_pattern_learning=True,
        max_correction_attempts=5,
    )

    corrector = SelfCorrector(config=config)

    context = {
        "tool_name": "api_fetcher",
        "parameters": {"url": "https://api.example.com/data"},
        "error_pattern": "timeout_error",
    }

    print("Simulating correction attempts with learning...")

    for i in range(3):
        tool_result = SkillResult(
            success=(i == 2),
            error="Timeout" if i < 2 else None,
        )

        context["tool_result"] = tool_result
        context["error"] = tool_result.error

        if not tool_result.success:
            reflection = corrector.reflect_on_failure(tool_result.error, context)
            alternatives = corrector.generate_alternatives(context)

            if alternatives:
                best = corrector.select_best_strategy(alternatives)
                if best:
                    result = corrector.apply_correction(best, context)
                    print(
                        f"Attempt {i + 1}: {'Success' if result.success else 'Failed'}"
                    )

    stats = corrector.get_correction_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Successful: {stats['successful_corrections']}")
    print(f"  Failed: {stats['failed_corrections']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")


def example_multiple_triggers():
    """Demonstrate multiple correction triggers."""
    print("\n=== Multiple Correction Triggers Example ===\n")

    corrector = SelfCorrector(
        config=SelfCorrectorConfig(
            consecutive_error_threshold=2,
            stagnation_threshold=3,
            confidence_threshold=0.6,
            confidence_drop_threshold=0.15,
        )
    )

    scenarios = [
        {
            "name": "Repeated failures",
            "context": {
                "tool_result": SkillResult(success=False, error="API error"),
                "consecutive_errors": 3,
                "iteration": 5,
                "confidence": 0.7,
            },
        },
        {
            "name": "Confidence drop",
            "context": {
                "tool_result": SkillResult(success=True, data={"result": "ok"}),
                "consecutive_errors": 0,
                "iteration": 2,
                "confidence": 0.4,
            },
        },
        {
            "name": "Stagnation",
            "context": {
                "tool_result": SkillResult(success=True, data={"result": "ok"}),
                "consecutive_errors": 0,
                "iteration": 5,
                "confidence": 0.8,
            },
        },
        {
            "name": "No issues",
            "context": {
                "tool_result": SkillResult(success=True, data={"result": "ok"}),
                "consecutive_errors": 0,
                "iteration": 2,
                "confidence": 0.9,
            },
        },
    ]

    for scenario in scenarios:
        should_correct, trigger = corrector.should_self_correct(scenario["context"])
        print(
            f"{scenario['name']:20s}: {'Correct' if should_correct else 'No correction':15s} ({trigger.value if trigger else 'N/A'})"
        )


def example_strategy_generation():
    """Demonstrate strategy generation for different error types."""
    print("\n=== Strategy Generation Example ===\n")

    corrector = SelfCorrector()

    error_scenarios = [
        {
            "error": "Connection timeout after 30 seconds",
            "tool_name": "api_fetcher",
            "parameters": {"url": "https://api.example.com/data"},
        },
        {
            "error": "Invalid input: missing required field 'id'",
            "tool_name": "data_validator",
            "parameters": {"name": "test"},
        },
        {
            "error": "Rate limit exceeded (429)",
            "tool_name": "api_caller",
            "parameters": {"endpoint": "/api/data"},
        },
    ]

    for scenario in error_scenarios:
        print(f"\nError: {scenario['error']}")
        print(f"Tool: {scenario['tool_name']}")

        context = {
            "error": scenario["error"],
            "tool_name": scenario["tool_name"],
            "parameters": scenario["parameters"],
            "tool_result": SkillResult(success=False, error=scenario["error"]),
        }

        alternatives = corrector.generate_alternatives(context)

        print(f"Generated {len(alternatives)} strategies:")
        for i, strategy in enumerate(alternatives[:3], 1):
            print(f"  {i}. {strategy.strategy_type.value}: {strategy.description}")
            print(f"     Confidence: {strategy.confidence_score:.2f}")


def example_integration_with_tool_orchestrator():
    """Demonstrate integration with ToolOrchestrator."""
    print("\n=== Integration with ToolOrchestrator Example ===\n")

    from core.self_corrector import SelfCorrector

    corrector = SelfCorrector()

    print("Simulating tool execution with self-correction:")

    for iteration in range(1, 6):
        print(f"\nIteration {iteration}:")

        if iteration < 4:
            tool_result = SkillResult(
                success=False,
                error=f"API error (attempt {iteration})",
            )
            consecutive_errors = iteration
        else:
            tool_result = SkillResult(
                success=True,
                data={"result": "success"},
            )
            consecutive_errors = 0

        context = {
            "tool_result": tool_result,
            "tool_name": "api_client",
            "iteration": iteration,
            "consecutive_errors": consecutive_errors,
            "parameters": {"endpoint": "/api/data"},
            "session_id": 1,
        }

        should_correct, trigger = corrector.should_self_correct(context)

        if should_correct and not tool_result.success:
            print(f"  Triggered correction: {trigger.value}")

            reflection = corrector.reflect_on_failure(tool_result.error, context)
            alternatives = corrector.generate_alternatives(context)

            if alternatives:
                best_strategy = corrector.select_best_strategy(alternatives)
                if best_strategy:
                    result = corrector.apply_correction(best_strategy, context)
                    print(f"  Applied: {best_strategy.strategy_type.value}")
                    print(f"  Result: {'Success' if result.success else 'Failed'}")
        else:
            print(f"  Execution: {'Success' if tool_result.success else 'Failed'}")

    stats = corrector.get_correction_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total corrections: {stats['total_attempts']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  By trigger: {stats['by_trigger']}")
    print(f"  By strategy: {stats['by_strategy']}")


def main():
    """Run all examples."""
    print("Self-Correction System Examples")
    print("=" * 50)

    example_basic_correction()
    example_error_pattern_detection()
    example_confidence_scoring()
    example_learning_from_corrections()
    example_multiple_triggers()
    example_strategy_generation()
    example_integration_with_tool_orchestrator()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
