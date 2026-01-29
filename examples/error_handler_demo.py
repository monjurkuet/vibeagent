"""Example demonstrating ErrorHandler integration with ToolOrchestrator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database_manager import DatabaseManager
from core.error_handler import ErrorHandler


class DemoErrorHandler:
    """Demo of ErrorHandler usage."""

    def __init__(self):
        """Initialize demo with error handler and database."""
        self.db_manager = DatabaseManager("data/demo_error_handler.db")
        self.error_handler = ErrorHandler(self.db_manager)

    def demo_error_classification(self):
        """Demonstrate error classification."""
        print("=" * 60)
        print("DEMO: Error Classification")
        print("=" * 60)

        test_errors = [
            ValueError("Invalid parameter format"),
            ConnectionError("Network unreachable"),
            TimeoutError("Request timed out"),
            PermissionError("Access denied"),
            FileNotFoundError("Resource not found"),
            RuntimeError("Rate limit exceeded (429)"),
            Exception("Internal server error (500)"),
        ]

        for error in test_errors:
            classification = self.error_handler.classifier.classify(error)
            print(f"\nError: {str(error)[:50]}")
            print(f"  Type: {classification.error_type.value}")
            print(f"  Severity: {classification.severity.value}")
            print(f"  Retryable: {classification.retryability.value}")
            print(f"  Confidence: {classification.confidence:.2f}")

    def demo_error_context_building(self):
        """Demonstrate error context building."""
        print("\n" + "=" * 60)
        print("DEMO: Error Context Building")
        print("=" * 60)

        context = self.error_handler.build_error_context(
            tool_name="arxiv_search",
            parameters={
                "query": "machine learning",
                "max_results": 10,
            },
            attempt_number=2,
            previous_attempts=[
                {"attempt": 1, "error": "Network error", "recovery": "retry"},
            ],
        )

        print(f"\nTool: {context.tool_name}")
        print(f"Parameters: {context.parameters}")
        print(f"Attempt: {context.attempt_number}")
        print(f"Previous attempts: {len(context.previous_attempts)}")

    def demo_recovery_strategies(self):
        """Demonstrate recovery strategy generation."""
        print("\n" + "=" * 60)
        print("DEMO: Recovery Strategy Generation")
        print("=" * 60)

        from core.error_handler import ErrorType

        error_types = [
            ErrorType.NETWORK,
            ErrorType.VALIDATION,
            ErrorType.TIMEOUT,
            ErrorType.PERMISSION,
        ]

        for error_type in error_types:
            context = self.error_handler.build_error_context(
                tool_name="test_tool",
                parameters={"param1": "value1"},
                attempt_number=1,
            )

            suggestions = self.error_handler.get_recovery_strategy(error_type, context)

            print(f"\nError Type: {error_type.value}")
            print("  Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"    {i}. {suggestion.strategy.value}: {suggestion.description}")
                print(f"       Success rate: {suggestion.estimated_success_rate:.2%}")
                print(f"       Confidence: {suggestion.confidence:.2%}")

    def demo_error_formatting(self):
        """Demonstrate error formatting for LLM."""
        print("\n" + "=" * 60)
        print("DEMO: Error Formatting for LLM")
        print("=" * 60)

        error = ConnectionError("Network unreachable after 3 attempts")
        context = self.error_handler.build_error_context(
            tool_name="arxiv_search",
            parameters={"query": "machine learning", "max_results": 10},
            attempt_number=3,
        )

        classification = self.error_handler.classifier.classify(error, context)
        formatted = self.error_handler.format_error_for_llm(error, classification, context)

        print(f"\n{formatted}")

    def demo_retry_logic(self):
        """Demonstrate retry logic."""
        print("\n" + "=" * 60)
        print("DEMO: Retry Logic")
        print("=" * 60)

        from core.error_handler import ErrorType

        error_types = [
            (ConnectionError("Network error"), ErrorType.NETWORK),
            (ValueError("Invalid parameter"), ErrorType.VALIDATION),
            (TimeoutError("Timeout"), ErrorType.TIMEOUT),
        ]

        for error, error_type in error_types:
            print(f"\nError: {str(error)[:40]}")
            print(f"  Retryable: {self.error_handler.is_retryable_error(error)}")

            for attempt in range(1, 4):
                delay = self.error_handler.get_retry_delay(attempt, error_type)
                print(f"  Attempt {attempt}: {delay:.1f}s delay")

            context = self.error_handler.build_error_context(
                tool_name="test_tool",
                parameters={},
                attempt_number=2,
            )
            print(f"  Should abort: {self.error_handler.should_abort(error, context)}")

    def demo_pattern_learning(self):
        """Demonstrate pattern learning."""
        print("\n" + "=" * 60)
        print("DEMO: Pattern Learning")
        print("=" * 60)

        fingerprint = "test_pattern_123"
        context = self.error_handler.build_error_context(
            tool_name="arxiv_search",
            parameters={"query": "test"},
            attempt_number=1,
        )

        print("\nSimulating error pattern learning...")

        for i in range(5):
            success = i > 2
            self.error_handler.record_error_recovery(
                session_id=1,
                tool_call_id=1,
                error=ConnectionError("Network error"),
                context=context,
                recovery_strategy="retry",
                success=success,
            )
            print(f"  Attempt {i + 1}: {'Success' if success else 'Failed'}")

        pattern = self.error_handler.pattern_db.get_pattern(fingerprint)
        if pattern:
            print("\nPattern learned:")
            print(f"  Total occurrences: {pattern.total_occurrences}")
            print(f"  Successful recoveries: {pattern.successful_recoveries}")
            print(f"  Success rate: {pattern.success_rate:.1f}%")
            print("  Recovery strategies:")
            for strategy, rate in pattern.recovery_strategies.items():
                print(f"    {strategy}: {rate:.2%}")

    def demo_full_error_handling(self):
        """Demonstrate full error handling workflow."""
        print("\n" + "=" * 60)
        print("DEMO: Full Error Handling Workflow")
        print("=" * 60)

        error = TimeoutError("Request timed out after 30 seconds")

        formatted_message, suggestions = self.error_handler.handle_error(
            error=error,
            tool_name="arxiv_search",
            parameters={"query": "machine learning", "max_results": 10},
            session_id=1,
            tool_call_id=1,
            attempt_number=1,
        )

        print(f"\nFormatted error message:\n{formatted_message}")
        print("\nRecovery suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion.strategy.value}: {suggestion.description}")

    def run_all_demos(self):
        """Run all demonstrations."""
        print("\n" + "=" * 60)
        print("ENHANCED ERROR HANDLER DEMONSTRATIONS")
        print("=" * 60)

        self.demo_error_classification()
        self.demo_error_context_building()
        self.demo_recovery_strategies()
        self.demo_error_formatting()
        self.demo_retry_logic()
        self.demo_pattern_learning()
        self.demo_full_error_handling()

        print("\n" + "=" * 60)
        print("All demonstrations completed!")
        print("=" * 60)


if __name__ == "__main__":
    demo = DemoErrorHandler()
    demo.run_all_demos()
