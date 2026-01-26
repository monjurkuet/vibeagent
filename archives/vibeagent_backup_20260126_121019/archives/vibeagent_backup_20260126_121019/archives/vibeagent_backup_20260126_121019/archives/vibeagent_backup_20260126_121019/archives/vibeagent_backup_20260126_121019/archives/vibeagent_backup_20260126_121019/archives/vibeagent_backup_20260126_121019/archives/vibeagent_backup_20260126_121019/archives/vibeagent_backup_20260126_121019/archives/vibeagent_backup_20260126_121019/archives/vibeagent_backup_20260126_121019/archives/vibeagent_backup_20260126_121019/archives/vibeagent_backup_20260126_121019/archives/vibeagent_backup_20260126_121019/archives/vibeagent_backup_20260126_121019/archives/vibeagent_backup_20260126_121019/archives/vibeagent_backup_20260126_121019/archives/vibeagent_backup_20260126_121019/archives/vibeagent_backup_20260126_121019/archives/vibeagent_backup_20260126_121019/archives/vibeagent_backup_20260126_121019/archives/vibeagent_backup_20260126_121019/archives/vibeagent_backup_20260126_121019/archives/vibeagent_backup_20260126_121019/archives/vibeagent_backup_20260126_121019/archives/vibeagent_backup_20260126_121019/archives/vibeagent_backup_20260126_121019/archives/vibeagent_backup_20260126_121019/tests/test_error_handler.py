"""Tests for ErrorHandler functionality."""

import pytest
import json
from datetime import datetime
from core.error_handler import (
    ErrorHandler,
    ErrorClassifier,
    ErrorType,
    SeverityLevel,
    Retryability,
    RecoveryStrategy,
    ErrorContext,
    ErrorClassification,
    RecoverySuggestion,
    ErrorPatternDatabase,
)


class TestErrorClassifier:
    """Test ErrorClassifier functionality."""

    def test_classify_validation_error(self):
        """Test classification of validation errors."""
        classifier = ErrorClassifier()
        error = ValueError("Invalid parameter format")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.VALIDATION
        assert classification.severity == SeverityLevel.LOW
        assert classification.retryability == Retryability.NON_RETRYABLE

    def test_classify_network_error(self):
        """Test classification of network errors."""
        classifier = ErrorClassifier()
        error = ConnectionError("Network unreachable")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.NETWORK
        assert classification.severity == SeverityLevel.MEDIUM
        assert classification.retryability == Retryability.RETRYABLE

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        classifier = ErrorClassifier()
        error = TimeoutError("Request timed out")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.TIMEOUT
        assert classification.severity == SeverityLevel.MEDIUM
        assert classification.retryability == Retryability.RETRYABLE

    def test_classify_permission_error(self):
        """Test classification of permission errors."""
        classifier = ErrorClassifier()
        error = PermissionError("Access denied")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.PERMISSION
        assert classification.severity == SeverityLevel.HIGH
        assert classification.retryability == Retryability.NON_RETRYABLE

    def test_classify_not_found_error(self):
        """Test classification of not found errors."""
        classifier = ErrorClassifier()
        error = FileNotFoundError("Resource not found")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.NOT_FOUND
        assert classification.severity == SeverityLevel.LOW
        assert classification.retryability == Retryability.NON_RETRYABLE

    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        classifier = ErrorClassifier()

        class RateLimitError(Exception):
            pass

        error = RateLimitError("Rate limit exceeded (429)")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.RATE_LIMIT
        assert classification.severity == SeverityLevel.MEDIUM
        assert classification.retryability == Retryability.CONDITIONAL

    def test_classify_internal_error(self):
        """Test classification of internal errors."""
        classifier = ErrorClassifier()

        class InternalError(Exception):
            pass

        error = InternalError("Internal server error (500)")

        classification = classifier.classify(error)

        assert classification.error_type == ErrorType.INTERNAL
        assert classification.severity == SeverityLevel.CRITICAL
        assert classification.retryability == Retryability.RETRYABLE


class TestErrorHandler:
    """Test ErrorHandler functionality."""

    def test_get_error_fingerprint(self):
        """Test error fingerprint generation."""
        handler = ErrorHandler()

        error = ValueError("Invalid parameter")
        context = ErrorContext(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            attempt_number=1,
        )

        fingerprint = handler.get_error_fingerprint(error, context)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16

        same_fingerprint = handler.get_error_fingerprint(error, context)
        assert fingerprint == same_fingerprint

    def test_build_error_context(self):
        """Test error context building."""
        handler = ErrorHandler()

        context = handler.build_error_context(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            attempt_number=2,
            previous_attempts=[{"attempt": 1, "error": "first error"}],
        )

        assert context.tool_name == "test_tool"
        assert context.parameters == {"param1": "value1"}
        assert context.attempt_number == 2
        assert len(context.previous_attempts) == 1

    def test_is_retryable_error(self):
        """Test retryable error detection."""
        handler = ErrorHandler()

        retryable_errors = [
            ConnectionError("Network error"),
            TimeoutError("Timeout"),
        ]

        for error in retryable_errors:
            assert handler.is_retryable_error(error) is True

    def test_is_not_retryable_error(self):
        """Test non-retryable error detection."""
        handler = ErrorHandler()

        non_retryable_errors = [
            ValueError("Invalid parameter"),
            PermissionError("Access denied"),
            FileNotFoundError("Not found"),
        ]

        for error in non_retryable_errors:
            assert handler.is_retryable_error(error) is False

    def test_get_retry_delay(self):
        """Test retry delay calculation."""
        handler = ErrorHandler()

        delay1 = handler.get_retry_delay(1, ErrorType.NETWORK)
        delay2 = handler.get_retry_delay(2, ErrorType.NETWORK)
        delay3 = handler.get_retry_delay(3, ErrorType.NETWORK)

        assert delay1 < delay2 < delay3
        assert delay3 <= 60.0

    def test_should_abort_critical_error(self):
        """Test abort decision for critical errors."""
        handler = ErrorHandler()

        class CriticalError(Exception):
            pass

        error = CriticalError("Internal server error (500)")
        context = ErrorContext(
            tool_name="test_tool",
            parameters={},
            attempt_number=1,
        )

        assert handler.should_abort(error, context) is True

    def test_should_abort_after_max_attempts(self):
        """Test abort decision after max attempts."""
        handler = ErrorHandler()

        error = ValueError("Invalid parameter")
        context = ErrorContext(
            tool_name="test_tool",
            parameters={},
            attempt_number=3,
        )

        assert handler.should_abort(error, context) is True

    def test_should_not_abort_early(self):
        """Test should not abort early for retryable errors."""
        handler = ErrorHandler()

        error = ConnectionError("Network error")
        context = ErrorContext(
            tool_name="test_tool",
            parameters={},
            attempt_number=1,
        )

        assert handler.should_abort(error, context) is False

    def test_get_recovery_strategy(self):
        """Test recovery strategy generation."""
        handler = ErrorHandler()

        context = ErrorContext(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            attempt_number=1,
        )

        suggestions = handler.get_recovery_strategy(ErrorType.NETWORK, context)

        assert len(suggestions) > 0
        assert all(isinstance(s, RecoverySuggestion) for s in suggestions)

        assert any(s.strategy == RecoveryStrategy.RETRY for s in suggestions)

    def test_handle_error(self):
        """Test full error handling."""
        handler = ErrorHandler()

        error = ConnectionError("Network unreachable")
        formatted_message, suggestions = handler.handle_error(
            error=error,
            tool_name="test_tool",
            parameters={"param1": "value1"},
            attempt_number=1,
        )

        assert isinstance(formatted_message, str)
        assert "Error Information" in formatted_message
        assert "Recovery Suggestions" in formatted_message

        assert len(suggestions) > 0
        assert isinstance(suggestions[0], RecoverySuggestion)

    def test_format_error_for_llm(self):
        """Test error formatting for LLM."""
        handler = ErrorHandler()

        error = ValueError("Invalid parameter format")
        classification = ErrorClassification(
            error_type=ErrorType.VALIDATION,
            severity=SeverityLevel.LOW,
            retryability=Retryability.NON_RETRYABLE,
            confidence=0.9,
            description="Validation error",
        )
        context = ErrorContext(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            attempt_number=1,
        )

        formatted = handler.format_error_for_llm(error, classification, context)

        assert "Error Information" in formatted
        assert "validation" in formatted.lower()
        assert "low" in formatted.lower()
        assert "test_tool" in formatted
        assert "Recovery Suggestions" in formatted
        assert "Next Steps" in formatted

    def test_get_likely_causes(self):
        """Test likely causes generation."""
        handler = ErrorHandler()

        causes = handler._get_likely_causes(
            ErrorType.NETWORK, ConnectionError("Network error")
        )

        assert isinstance(causes, str)
        assert len(causes) > 0
        assert "connectivity" in causes.lower()


class TestErrorPatternDatabase:
    """Test ErrorPatternDatabase functionality."""

    def test_update_and_get_pattern(self):
        """Test pattern update and retrieval."""
        db = ErrorPatternDatabase()

        db.update_pattern(
            fingerprint="test123",
            error_type=ErrorType.NETWORK,
            pattern_key="tool:abc123",
            recovery_strategy="retry",
            success=True,
        )

        pattern = db.get_pattern("test123")

        assert pattern is not None
        assert pattern.fingerprint == "test123"
        assert pattern.error_type == ErrorType.NETWORK
        assert pattern.total_occurrences == 1
        assert pattern.successful_recoveries == 1

    def test_multiple_pattern_updates(self):
        """Test multiple pattern updates."""
        db = ErrorPatternDatabase()

        db.update_pattern(
            fingerprint="test456",
            error_type=ErrorType.TIMEOUT,
            pattern_key="tool:def456",
            recovery_strategy="retry",
            success=True,
        )

        db.update_pattern(
            fingerprint="test456",
            error_type=ErrorType.TIMEOUT,
            pattern_key="tool:def456",
            recovery_strategy="retry",
            success=False,
        )

        pattern = db.get_pattern("test456")

        assert pattern.total_occurrences == 2
        assert pattern.successful_recoveries == 1
        assert pattern.success_rate == 50.0

    def test_get_similar_patterns(self):
        """Test getting similar patterns."""
        db = ErrorPatternDatabase()

        db.update_pattern(
            fingerprint="net1",
            error_type=ErrorType.NETWORK,
            pattern_key="tool:net1",
            recovery_strategy="retry",
            success=True,
        )

        db.update_pattern(
            fingerprint="net2",
            error_type=ErrorType.NETWORK,
            pattern_key="tool:net2",
            recovery_strategy="retry",
            success=False,
        )

        similar = db.get_similar_patterns(ErrorType.NETWORK)

        assert len(similar) >= 2
        assert all(p.error_type == ErrorType.NETWORK for p in similar)


class TestIntegration:
    """Integration tests for ErrorHandler with DatabaseManager."""

    def test_full_error_handling_workflow(self):
        """Test complete error handling workflow."""
        from core.database_manager import DatabaseManager
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db_manager = DatabaseManager(db_path)
            handler = ErrorHandler(db_manager)

            error = ConnectionError("Network unreachable")
            formatted_message, suggestions = handler.handle_error(
                error=error,
                tool_name="test_tool",
                parameters={"param1": "value1"},
                session_id=1,
                tool_call_id=1,
                attempt_number=1,
            )

            assert isinstance(formatted_message, str)
            assert len(suggestions) > 0

            context = handler.build_error_context(
                tool_name="test_tool",
                parameters={"param1": "value1"},
                attempt_number=1,
            )

            handler.record_error_recovery(
                session_id=1,
                tool_call_id=1,
                error=error,
                context=context,
                recovery_strategy="retry",
                success=True,
            )

            pattern = handler.pattern_db.get_pattern(
                handler.get_error_fingerprint(error, context)
            )

            assert pattern is not None
            assert pattern.total_occurrences == 1
            assert pattern.successful_recoveries == 1

    def test_recovery_learning(self):
        """Test that system learns from successful recoveries."""
        handler = ErrorHandler()

        error = ConnectionError("Network error")
        context = ErrorContext(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            attempt_number=1,
        )

        handler.record_error_recovery(
            session_id=1,
            tool_call_id=1,
            error=error,
            context=context,
            recovery_strategy="retry",
            success=True,
        )

        suggestions = handler.get_recovery_strategy(ErrorType.NETWORK, context)

        assert len(suggestions) > 0
        retry_suggestion = next(
            (s for s in suggestions if s.strategy == RecoveryStrategy.RETRY), None
        )
        assert retry_suggestion is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
