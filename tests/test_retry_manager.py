"""Test suite for RetryManager."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from core.retry_manager import (
    BackoffStrategy,
    ErrorType,
    RetryAttempt,
    RetryManager,
    RetryPolicy,
)
from core.skill import SkillResult


class TestRetryPolicy:
    """Test RetryPolicy dataclass."""

    def test_default_policy(self):
        """Test default retry policy."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay_ms == 1000
        assert policy.max_delay_ms == 30000
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert policy.jitter_enabled is True
        assert policy.jitter_factor == 0.1

    def test_custom_policy(self):
        """Test custom retry policy."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay_ms=2000,
            max_delay_ms=60000,
            backoff_strategy=BackoffStrategy.LINEAR,
            jitter_enabled=False,
        )
        assert policy.max_retries == 5
        assert policy.base_delay_ms == 2000
        assert policy.max_delay_ms == 60000
        assert policy.backoff_strategy == BackoffStrategy.LINEAR
        assert policy.jitter_enabled is False

    def test_to_dict(self):
        """Test converting policy to dictionary."""
        policy = RetryPolicy(max_retries=2)
        policy_dict = policy.to_dict()
        assert policy_dict["max_retries"] == 2
        assert policy_dict["backoff_strategy"] == "exponential"


class TestErrorType:
    """Test error type classification."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_classify_network_error(self, retry_manager):
        """Test classification of network errors."""
        error = Exception("Connection failed")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.NETWORK

    def test_classify_timeout_error(self, retry_manager):
        """Test classification of timeout errors."""
        error = Exception("Request timed out")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.TIMEOUT

    def test_classify_rate_limit_error(self, retry_manager):
        """Test classification of rate limit errors."""
        error = Exception("429 Too Many Requests")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.RATE_LIMIT

    def test_classify_validation_error(self, retry_manager):
        """Test classification of validation errors."""
        error = Exception("Invalid parameter value")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.VALIDATION

    def test_classify_permission_error(self, retry_manager):
        """Test classification of permission errors."""
        error = Exception("403 Forbidden")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.PERMISSION

    def test_classify_not_found_error(self, retry_manager):
        """Test classification of not found errors."""
        error = Exception("404 Not Found")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.NOT_FOUND

    def test_classify_unknown_error(self, retry_manager):
        """Test classification of unknown errors."""
        error = Exception("Some unknown error")
        error_type = retry_manager.classify_error(error)
        assert error_type == ErrorType.UNKNOWN


class TestIsRetryable:
    """Test retryable error detection."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_network_error_is_retryable(self, retry_manager):
        """Test that network errors are retryable."""
        error = Exception("Connection failed")
        assert retry_manager.is_retryable(error) is True

    def test_timeout_error_is_retryable(self, retry_manager):
        """Test that timeout errors are retryable."""
        error = Exception("Request timed out")
        assert retry_manager.is_retryable(error) is True

    def test_rate_limit_error_is_retryable(self, retry_manager):
        """Test that rate limit errors are retryable."""
        error = Exception("429 Too Many Requests")
        assert retry_manager.is_retryable(error) is True

    def test_validation_error_is_not_retryable(self, retry_manager):
        """Test that validation errors are not retryable."""
        error = Exception("Invalid parameter value")
        assert retry_manager.is_retryable(error) is False

    def test_permission_error_is_not_retryable(self, retry_manager):
        """Test that permission errors are not retryable."""
        error = Exception("403 Forbidden")
        assert retry_manager.is_retryable(error) is False

    def test_not_found_error_is_not_retryable(self, retry_manager):
        """Test that not found errors are not retryable."""
        error = Exception("404 Not Found")
        assert retry_manager.is_retryable(error) is False


class TestBackoffCalculation:
    """Test backoff calculation strategies."""

    @pytest.fixture
    def policy(self):
        """Create a retry policy for testing."""
        return RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=10000,
            jitter_enabled=False,
        )

    def test_exponential_backoff(self, policy):
        """Test exponential backoff calculation."""
        retry_manager = RetryManager()

        backoff_0 = retry_manager.calculate_backoff(0, policy, BackoffStrategy.EXPONENTIAL)
        assert backoff_0 == 1000

        backoff_1 = retry_manager.calculate_backoff(1, policy, BackoffStrategy.EXPONENTIAL)
        assert backoff_1 == 2000

        backoff_2 = retry_manager.calculate_backoff(2, policy, BackoffStrategy.EXPONENTIAL)
        assert backoff_2 == 4000

    def test_linear_backoff(self, policy):
        """Test linear backoff calculation."""
        retry_manager = RetryManager()

        backoff_0 = retry_manager.calculate_backoff(0, policy, BackoffStrategy.LINEAR)
        assert backoff_0 == 1000

        backoff_1 = retry_manager.calculate_backoff(1, policy, BackoffStrategy.LINEAR)
        assert backoff_1 == 2000

        backoff_2 = retry_manager.calculate_backoff(2, policy, BackoffStrategy.LINEAR)
        assert backoff_2 == 3000

    def test_fixed_backoff(self, policy):
        """Test fixed backoff calculation."""
        retry_manager = RetryManager()

        backoff_0 = retry_manager.calculate_backoff(0, policy, BackoffStrategy.FIXED)
        assert backoff_0 == 1000

        backoff_1 = retry_manager.calculate_backoff(1, policy, BackoffStrategy.FIXED)
        assert backoff_1 == 1000

        backoff_2 = retry_manager.calculate_backoff(2, policy, BackoffStrategy.FIXED)
        assert backoff_2 == 1000

    def test_max_delay_cap(self):
        """Test that backoff is capped at max_delay_ms."""
        policy = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=5000,
            jitter_enabled=False,
        )
        retry_manager = RetryManager()

        backoff = retry_manager.calculate_backoff(10, policy, BackoffStrategy.EXPONENTIAL)
        assert backoff == 5000

    def test_jitter_enabled(self):
        """Test that jitter adds randomness to backoff."""
        policy = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=10000,
            jitter_enabled=True,
            jitter_factor=0.1,
        )
        retry_manager = RetryManager()

        backoffs = [
            retry_manager.calculate_backoff(1, policy, BackoffStrategy.EXPONENTIAL)
            for _ in range(10)
        ]

        assert len(set(backoffs)) > 1
        for backoff in backoffs:
            assert 1800 <= backoff <= 2200


class TestExecuteWithRetry:
    """Test execute_with_retry functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_success_on_first_attempt(self, retry_manager):
        """Test successful execution on first attempt."""

        def successful_func():
            return SkillResult(success=True, data={"result": "success"})

        result = retry_manager.execute_with_retry(successful_func, "test_tool")

        assert result.success is True
        assert result.data == {"result": "success"}
        assert len(retry_manager.attempt_history) == 0

    def test_retry_on_network_error(self, retry_manager):
        """Test retry on network error."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Connection failed")
            return SkillResult(success=True, data={"result": "success"})

        result = retry_manager.execute_with_retry(flaky_func, "test_tool")

        assert result.success is True
        assert call_count == 2
        assert len(retry_manager.attempt_history) == 2
        assert retry_manager.statistics.total_retries == 2
        assert retry_manager.statistics.successful_retries == 1
        assert retry_manager.statistics.failed_retries == 1

    def test_no_retry_on_validation_error(self, retry_manager):
        """Test that validation errors are not retried."""
        call_count = 0

        def invalid_func():
            nonlocal call_count
            call_count += 1
            return SkillResult(success=False, error="Invalid parameter value")

        result = retry_manager.execute_with_retry(invalid_func, "test_tool")

        assert result.success is False
        assert call_count == 1
        assert len(retry_manager.attempt_history) == 0

    def test_max_retries_exceeded(self, retry_manager):
        """Test that max retries is respected."""
        call_count = 0

        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Connection failed")

        result = retry_manager.execute_with_retry(always_failing_func, "test_tool")

        assert result.success is False
        assert call_count == 6  # 1 initial + 5 retries (network error type policy)
        assert len(retry_manager.attempt_history) == 5

    @patch("time.sleep")
    def test_backoff_is_applied(self, mock_sleep, retry_manager):
        """Test that backoff delay is applied."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection failed")
            return SkillResult(success=True, data={"result": "success"})

        result = retry_manager.execute_with_retry(flaky_func, "test_tool")

        assert result.success is True
        assert mock_sleep.call_count == 2

    def test_database_tracking(self, retry_manager):
        """Test that retries are tracked in database."""
        mock_db = Mock()
        retry_manager.db_manager = mock_db

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Connection failed")
            return SkillResult(success=True, data={"result": "success"})

        result = retry_manager.execute_with_retry(
            flaky_func,
            "test_tool",
            session_id=123,
            tool_call_id=456,
        )

        assert result.success is True
        assert mock_db.add_error_recovery.call_count == 1


class TestRetryStatistics:
    """Test retry statistics tracking."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_statistics_initialization(self, retry_manager):
        """Test that statistics are initialized correctly."""
        stats = retry_manager.statistics
        assert stats.total_retries == 0
        assert stats.successful_retries == 0
        assert stats.failed_retries == 0
        assert stats.retry_success_rate == 0.0

    def test_update_statistics(self, retry_manager):
        """Test updating statistics with attempts."""
        attempt = RetryAttempt(
            attempt_number=1,
            tool_name="test_tool",
            error_type="network",
            error_message="Connection failed",
            backoff_ms=1000,
            timestamp=datetime.now(),
            success=True,
        )

        retry_manager.statistics.update(attempt)

        assert retry_manager.statistics.total_retries == 1
        assert retry_manager.statistics.successful_retries == 1
        assert retry_manager.statistics.failed_retries == 0
        assert retry_manager.statistics.retry_success_rate == 1.0

    def test_get_statistics(self, retry_manager):
        """Test getting statistics as dictionary."""
        stats_dict = retry_manager.get_statistics()
        assert "total_retries" in stats_dict
        assert "successful_retries" in stats_dict
        assert "retry_success_rate" in stats_dict

    def test_reset_statistics(self, retry_manager):
        """Test resetting statistics."""
        attempt = RetryAttempt(
            attempt_number=1,
            tool_name="test_tool",
            error_type="network",
            error_message="Connection failed",
            backoff_ms=1000,
            timestamp=datetime.now(),
            success=True,
        )

        retry_manager.statistics.update(attempt)
        retry_manager.reset_statistics()

        assert retry_manager.statistics.total_retries == 0
        assert len(retry_manager.attempt_history) == 0


class TestRetryPolicies:
    """Test retry policy configuration."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_get_global_policy(self, retry_manager):
        """Test getting global retry policy."""
        policy = retry_manager.get_retry_policy()
        assert policy.max_retries == 3

    def test_get_tool_specific_policy(self, retry_manager):
        """Test getting tool-specific retry policy."""
        policy = retry_manager.get_retry_policy("arxiv_search_papers")
        assert policy.max_retries == 5
        assert policy.base_delay_ms == 2000

    def test_get_error_type_policy(self, retry_manager):
        """Test getting error type-specific retry policy."""
        policy = retry_manager.get_retry_policy(error_type=ErrorType.RATE_LIMIT)
        assert policy.max_retries == 5
        assert policy.base_delay_ms == 5000

    def test_add_tool_retry_rule(self, retry_manager):
        """Test adding tool-specific retry rules."""
        retry_manager.add_tool_retry_rule("custom_tool", [ErrorType.NETWORK, ErrorType.TIMEOUT])

        assert "custom_tool" in retry_manager.tool_retry_rules
        assert ErrorType.NETWORK in retry_manager.tool_retry_rules["custom_tool"]
        assert ErrorType.TIMEOUT in retry_manager.tool_retry_rules["custom_tool"]

    def test_set_model_retry_policy(self, retry_manager):
        """Test setting model-specific retry policy."""
        custom_policy = RetryPolicy(max_retries=10)
        retry_manager.set_model_retry_policy("gpt-4", custom_policy)

        retrieved_policy = retry_manager.get_model_retry_policy("gpt-4")
        assert retrieved_policy.max_retries == 10


class TestRetryDecorator:
    """Test retry decorator functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_retry_decorator(self, retry_manager):
        """Test using retry decorator."""
        call_count = 0

        @retry_manager.retry_decorator(tool_name="test_tool")
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Connection failed")
            return SkillResult(success=True, data={"result": "success"})

        result = flaky_function()

        assert result.success is True
        assert call_count == 2


class TestRecoveryRate:
    """Test recovery rate calculation."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_recovery_rate_zero_attempts(self, retry_manager):
        """Test recovery rate with zero attempts."""
        rate = retry_manager.get_recovery_rate()
        assert rate == 0.0

    def test_recovery_rate_half_success(self, retry_manager):
        """Test recovery rate with 50% success rate."""
        for i in range(5):
            attempt = RetryAttempt(
                attempt_number=i + 1,
                tool_name="test_tool",
                error_type="network",
                error_message="Connection failed",
                backoff_ms=1000,
                timestamp=datetime.now(),
                success=(i < 2),
            )
            retry_manager.statistics.update(attempt)

        rate = retry_manager.get_recovery_rate()
        assert rate == 40.0

    def test_recovery_rate_all_success(self, retry_manager):
        """Test recovery rate with 100% success rate."""
        for i in range(3):
            attempt = RetryAttempt(
                attempt_number=i + 1,
                tool_name="test_tool",
                error_type="network",
                error_message="Connection failed",
                backoff_ms=1000,
                timestamp=datetime.now(),
                success=True,
            )
            retry_manager.statistics.update(attempt)

        rate = retry_manager.get_recovery_rate()
        assert rate == 100.0


class TestOptimalRetryLimits:
    """Test optimal retry limit calculation."""

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager()

    def test_calculate_optimal_retry_limits(self, retry_manager):
        """Test calculating optimal retry limits."""
        for i in range(50):
            attempt = RetryAttempt(
                attempt_number=i + 1,
                tool_name="frequent_tool",
                error_type="network",
                error_message="Connection failed",
                backoff_ms=1000,
                timestamp=datetime.now(),
                success=True,
            )
            retry_manager.statistics.update(attempt)

        optimal_limits = retry_manager.calculate_optimal_retry_limits()
        assert "frequent_tool" in optimal_limits
        assert optimal_limits["frequent_tool"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
