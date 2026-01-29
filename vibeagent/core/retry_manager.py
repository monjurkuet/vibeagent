"""Retry management system for tool execution with intelligent backoff."""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps

from .skill import SkillResult

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry attempts."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class ErrorType(Enum):
    """Error types for retry classification."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TEMPORARY = "temporary"
    VALIDATION = "validation"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter_enabled: bool = True
    jitter_factor: float = 0.1

    def to_dict(self) -> dict:
        """Convert policy to dictionary."""
        return {
            "max_retries": self.max_retries,
            "base_delay_ms": self.base_delay_ms,
            "max_delay_ms": self.max_delay_ms,
            "backoff_strategy": self.backoff_strategy.value,
            "jitter_enabled": self.jitter_enabled,
            "jitter_factor": self.jitter_factor,
        }


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""

    attempt_number: int
    tool_name: str
    error_type: str
    error_message: str
    backoff_ms: int
    timestamp: datetime
    success: bool
    recovery_strategy: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert attempt to dictionary."""
        return {
            "attempt_number": self.attempt_number,
            "tool_name": self.tool_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "backoff_ms": self.backoff_ms,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "recovery_strategy": self.recovery_strategy,
            "metadata": self.metadata,
        }


@dataclass
class RetryStatistics:
    """Statistics about retry behavior."""

    total_retries: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    retries_by_tool: dict[str, int] = field(default_factory=dict)
    retries_by_error_type: dict[str, int] = field(default_factory=dict)
    retry_success_rate: float = 0.0
    avg_retries_per_success: float = 0.0
    last_updated: datetime | None = None

    def update(self, attempt: RetryAttempt):
        """Update statistics with a new attempt."""
        self.total_retries += 1

        if attempt.success:
            self.successful_retries += 1
        else:
            self.failed_retries += 1

        self.retries_by_tool[attempt.tool_name] = self.retries_by_tool.get(attempt.tool_name, 0) + 1
        self.retries_by_error_type[attempt.error_type] = (
            self.retries_by_error_type.get(attempt.error_type, 0) + 1
        )

        if self.total_retries > 0:
            self.retry_success_rate = self.successful_retries / self.total_retries
            self.avg_retries_per_success = (
                self.total_retries / self.successful_retries if self.successful_retries > 0 else 0
            )

        self.last_updated = datetime.now()

    def to_dict(self) -> dict:
        """Convert statistics to dictionary."""
        return {
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "retries_by_tool": self.retries_by_tool,
            "retries_by_error_type": self.retries_by_error_type,
            "retry_success_rate": self.retry_success_rate,
            "avg_retries_per_success": self.avg_retries_per_success,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class RetryManager:
    """Manages retry logic for tool execution with intelligent backoff."""

    RETRYABLE_ERROR_TYPES = {
        ErrorType.NETWORK,
        ErrorType.TIMEOUT,
        ErrorType.RATE_LIMIT,
        ErrorType.TEMPORARY,
    }

    NON_RETRYABLE_ERROR_TYPES = {
        ErrorType.VALIDATION,
        ErrorType.PERMISSION,
        ErrorType.NOT_FOUND,
    }

    DEFAULT_GLOBAL_POLICY = RetryPolicy(
        max_retries=3,
        base_delay_ms=1000,
        max_delay_ms=30000,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        jitter_enabled=True,
        jitter_factor=0.1,
    )

    ERROR_TYPE_PATTERNS = {
        ErrorType.NETWORK: [
            "connection",
            "network",
            "dns",
            "unreachable",
            "socket",
            "ssl",
            "tls",
        ],
        ErrorType.TIMEOUT: ["timeout", "timed out", "deadline exceeded"],
        ErrorType.RATE_LIMIT: [
            "rate limit",
            "429",
            "too many requests",
            "quota exceeded",
        ],
        ErrorType.TEMPORARY: ["temporary", "transient", "try again", "unavailable"],
        ErrorType.VALIDATION: [
            "validation",
            "invalid",
            "malformed",
            "bad request",
            "400",
        ],
        ErrorType.PERMISSION: [
            "permission",
            "unauthorized",
            "forbidden",
            "401",
            "403",
            "access denied",
        ],
        ErrorType.NOT_FOUND: [
            "not found",
            "404",
            "does not exist",
            "no such",
        ],
    }

    TOOL_SPECIFIC_POLICIES = {
        "arxiv_search_papers": RetryPolicy(
            max_retries=5,
            base_delay_ms=2000,
            max_delay_ms=60000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ),
        "web_search_search_text": RetryPolicy(
            max_retries=3,
            base_delay_ms=1000,
            max_delay_ms=10000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ),
        "scraper": RetryPolicy(
            max_retries=2,
            base_delay_ms=500,
            max_delay_ms=5000,
            backoff_strategy=BackoffStrategy.LINEAR,
        ),
    }

    ERROR_TYPE_POLICIES = {
        ErrorType.NETWORK: RetryPolicy(
            max_retries=5,
            base_delay_ms=2000,
            max_delay_ms=60000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ),
        ErrorType.TIMEOUT: RetryPolicy(
            max_retries=3,
            base_delay_ms=3000,
            max_delay_ms=30000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ),
        ErrorType.RATE_LIMIT: RetryPolicy(
            max_retries=5,
            base_delay_ms=5000,
            max_delay_ms=120000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ),
        ErrorType.TEMPORARY: RetryPolicy(
            max_retries=3,
            base_delay_ms=1000,
            max_delay_ms=20000,
            backoff_strategy=BackoffStrategy.LINEAR,
        ),
    }

    def __init__(self, db_manager=None, config=None):
        """Initialize the retry manager.

        Args:
            db_manager: Optional DatabaseManager for tracking retries
            config: Optional Config object for loading retry policies
        """
        self.db_manager = db_manager
        self.config = config
        self.global_policy = self.DEFAULT_GLOBAL_POLICY
        self.statistics = RetryStatistics()
        self.attempt_history: list[RetryAttempt] = []
        self.tool_retry_rules: dict[str, set[ErrorType]] = {}
        self.model_retry_settings: dict[str, RetryPolicy] = {}

        self._load_configuration()

    def _load_configuration(self):
        """Load retry policies from configuration."""
        if self.config:
            retry_config = self.config.get("retry")
            if retry_config:
                if "global" in retry_config:
                    self.global_policy = RetryPolicy(**retry_config["global"])
                if "tool_policies" in retry_config:
                    for tool_name, policy_config in retry_config["tool_policies"].items():
                        self.TOOL_SPECIFIC_POLICIES[tool_name] = RetryPolicy(**policy_config)
                if "model_settings" in retry_config:
                    for model_name, policy_config in retry_config["model_settings"].items():
                        self.model_retry_settings[model_name] = RetryPolicy(**policy_config)

    def classify_error(self, error: Exception, tool_name: str = "") -> ErrorType:
        """Classify an error into an ErrorType.

        Args:
            error: The exception to classify
            tool_name: Optional tool name for context

        Returns:
            ErrorType classification
        """
        error_message = str(error).lower()
        error_class_name = error.__class__.__name__.lower()

        for error_type, patterns in self.ERROR_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_message or pattern in error_class_name:
                    return error_type

        tool_specific_rules = self.tool_retry_rules.get(tool_name, set())
        for error_type in tool_specific_rules:
            return error_type

        return ErrorType.UNKNOWN

    def is_retryable(self, error: Exception, tool_name: str = "") -> bool:
        """Check if an error is retryable.

        Args:
            error: The exception to check
            tool_name: Optional tool name for tool-specific rules

        Returns:
            True if the error is retryable, False otherwise
        """
        error_type = self.classify_error(error, tool_name)

        if error_type in self.NON_RETRYABLE_ERROR_TYPES:
            return False

        if error_type in self.RETRYABLE_ERROR_TYPES:
            return True

        tool_specific_rules = self.tool_retry_rules.get(tool_name, set())
        if error_type in tool_specific_rules:
            return True

        return False

    def get_retry_policy(
        self, tool_name: str = "", error_type: ErrorType | None = None
    ) -> RetryPolicy:
        """Get the retry policy for a specific tool and error type.

        Args:
            tool_name: Optional tool name
            error_type: Optional error type

        Returns:
            RetryPolicy to use
        """
        if tool_name in self.TOOL_SPECIFIC_POLICIES:
            tool_policy = self.TOOL_SPECIFIC_POLICIES[tool_name]

            if (
                error_type
                and error_type in self.ERROR_TYPE_POLICIES
                and tool_policy.max_retries < self.ERROR_TYPE_POLICIES[error_type].max_retries
            ):
                return self.ERROR_TYPE_POLICIES[error_type]

            return tool_policy

        if error_type and error_type in self.ERROR_TYPE_POLICIES:
            return self.ERROR_TYPE_POLICIES[error_type]

        return self.global_policy

    def calculate_backoff(
        self,
        attempt: int,
        policy: RetryPolicy,
        strategy: BackoffStrategy | None = None,
    ) -> int:
        """Calculate backoff delay before retry.

        Args:
            attempt: Current attempt number (0-based)
            policy: Retry policy to use
            strategy: Optional override for backoff strategy

        Returns:
            Delay in milliseconds
        """
        if strategy is None:
            strategy = policy.backoff_strategy

        if strategy == BackoffStrategy.EXPONENTIAL:
            delay = policy.base_delay_ms * (2**attempt)
        elif strategy == BackoffStrategy.LINEAR:
            delay = policy.base_delay_ms * (attempt + 1)
        else:
            delay = policy.base_delay_ms

        delay = min(delay, policy.max_delay_ms)

        if policy.jitter_enabled:
            jitter = delay * policy.jitter_factor
            delay = delay + random.uniform(-jitter, jitter)

        return int(max(0, delay))

    def execute_with_retry(
        self,
        func: Callable,
        tool_name: str = "",
        *args,
        session_id: int | None = None,
        tool_call_id: int | None = None,
        **kwargs,
    ) -> SkillResult:
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            tool_name: Name of the tool being executed
            *args: Positional arguments to pass to function
            session_id: Optional database session ID for tracking
            tool_call_id: Optional database tool call ID for tracking
            **kwargs: Keyword arguments to pass to function

        Returns:
            SkillResult from successful execution or last failed attempt
        """
        last_error = None
        last_result = None
        attempt_num = 0

        while True:
            try:
                result = func(*args, **kwargs)

                if result.success:
                    if attempt_num > 0:
                        error_type = self.classify_error(last_error, tool_name)
                        policy = self.get_retry_policy(tool_name, error_type)
                        success_attempt = RetryAttempt(
                            attempt_number=attempt_num,
                            tool_name=tool_name,
                            error_type=error_type.value,
                            error_message=str(last_error),
                            backoff_ms=0,
                            timestamp=datetime.now(),
                            success=True,
                            recovery_strategy=policy.backoff_strategy.value,
                        )
                        self.attempt_history.append(success_attempt)
                        self.statistics.update(success_attempt)

                        self._track_retry_success(
                            attempt_num,
                            tool_name,
                            session_id,
                            tool_call_id,
                            result,
                        )
                    return result

                if result.error:
                    error = Exception(result.error)
                    last_error = error
                    last_result = result
                else:
                    return result

            except Exception as e:
                last_error = e
                last_result = SkillResult(success=False, error=str(e))

            if not self.is_retryable(last_error, tool_name):
                logger.debug(f"Error not retryable for tool '{tool_name}': {last_error}")
                break

            error_type = self.classify_error(last_error, tool_name)
            policy = self.get_retry_policy(tool_name, error_type)

            if attempt_num >= policy.max_retries:
                logger.debug(f"Max retries ({policy.max_retries}) reached for tool '{tool_name}'")
                break

            backoff_ms = self.calculate_backoff(attempt_num, policy)

            attempt = RetryAttempt(
                attempt_number=attempt_num + 1,
                tool_name=tool_name,
                error_type=error_type.value,
                error_message=str(last_error),
                backoff_ms=backoff_ms,
                timestamp=datetime.now(),
                success=False,
                recovery_strategy=policy.backoff_strategy.value,
            )

            self.attempt_history.append(attempt)
            self.statistics.update(attempt)

            if self.db_manager and session_id and tool_call_id:
                try:
                    self.db_manager.add_error_recovery(
                        session_id=session_id,
                        tool_call_id=tool_call_id,
                        error_type=error_type.value,
                        recovery_strategy=f"retry_{policy.backoff_strategy.value}",
                        attempt_number=attempt_num + 1,
                        success=False,
                        original_error=str(last_error),
                        recovery_details={
                            "backoff_ms": backoff_ms,
                            "policy": policy.to_dict(),
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to track retry attempt: {e}")

            logger.debug(
                f"Retry {attempt_num + 1}/{policy.max_retries} for tool '{tool_name}' "
                f"after {backoff_ms}ms (error: {error_type.value})"
            )

            time.sleep(backoff_ms / 1000.0)
            attempt_num += 1

        if last_result:
            self._track_retry_failure(
                attempt_num,
                tool_name,
                session_id,
                tool_call_id,
                last_result,
            )

        return last_result or SkillResult(success=False, error="All retry attempts failed")

    def _track_retry_success(
        self,
        attempt_num: int,
        tool_name: str,
        session_id: int | None,
        tool_call_id: int | None,
        result: SkillResult,
    ):
        """Track a successful retry."""
        if self.db_manager and session_id and tool_call_id:
            try:
                self.db_manager.update_tool_call(
                    tool_call_id,
                    retry_count=attempt_num,
                )
            except Exception as e:
                logger.error(f"Failed to update tool call retry count: {e}")

    def _track_retry_failure(
        self,
        attempt_num: int,
        tool_name: str,
        session_id: int | None,
        tool_call_id: int | None,
        result: SkillResult,
    ):
        """Track a failed retry attempt."""
        if self.db_manager and session_id and tool_call_id:
            try:
                self.db_manager.update_tool_call(
                    tool_call_id,
                    retry_count=attempt_num,
                    success=False,
                    error_message=result.error,
                )
            except Exception as e:
                logger.error(f"Failed to update tool call retry count: {e}")

    def add_tool_retry_rule(self, tool_name: str, error_types: list[ErrorType]):
        """Add a tool-specific retry rule.

        Args:
            tool_name: Name of the tool
            error_types: List of error types that are retryable for this tool
        """
        self.tool_retry_rules[tool_name] = set(error_types)

    def set_model_retry_policy(self, model: str, policy: RetryPolicy):
        """Set retry policy for a specific model.

        Args:
            model: Model name
            policy: Retry policy to use
        """
        self.model_retry_settings[model] = policy

    def get_model_retry_policy(self, model: str) -> RetryPolicy:
        """Get retry policy for a specific model.

        Args:
            model: Model name

        Returns:
            RetryPolicy for the model or default global policy
        """
        return self.model_retry_settings.get(model, self.global_policy)

    def get_statistics(self) -> dict:
        """Get retry statistics.

        Returns:
            Dictionary with retry statistics
        """
        return self.statistics.to_dict()

    def get_attempt_history(self, tool_name: str | None = None, limit: int = 100) -> list[dict]:
        """Get retry attempt history.

        Args:
            tool_name: Optional filter by tool name
            limit: Maximum number of attempts to return

        Returns:
            List of retry attempt dictionaries
        """
        attempts = self.attempt_history

        if tool_name:
            attempts = [a for a in attempts if a.tool_name == tool_name]

        attempts = attempts[-limit:]

        return [a.to_dict() for a in attempts]

    def reset_statistics(self):
        """Reset retry statistics."""
        self.statistics = RetryStatistics()
        self.attempt_history.clear()

    def calculate_optimal_retry_limits(self) -> dict[str, int]:
        """Calculate optimal retry limits based on statistics.

        Returns:
            Dictionary of tool names to optimal retry limits
        """
        optimal_limits = {}

        for tool_name, count in self.statistics.retries_by_tool.items():
            if count < 10:
                optimal_limits[tool_name] = 2
            elif count < 50:
                optimal_limits[tool_name] = 3
            elif count < 100:
                optimal_limits[tool_name] = 4
            else:
                optimal_limits[tool_name] = 5

        return optimal_limits

    def get_recovery_rate(self) -> float:
        """Get the overall error recovery rate.

        Returns:
            Recovery rate as a percentage (0-100)
        """
        total = self.statistics.successful_retries + self.statistics.failed_retries
        if total == 0:
            return 0.0
        return (self.statistics.successful_retries / total) * 100

    def retry_decorator(self, tool_name: str = "", session_id: int | None = None):
        """Decorator to add retry logic to a function.

        Args:
            tool_name: Name of the tool
            session_id: Optional database session ID

        Returns:
            Decorated function
        """

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_retry(
                    func,
                    tool_name,
                    *args,
                    session_id=session_id,
                    **kwargs,
                )

            return wrapper

        return decorator


def retry_with_manager(
    retry_manager: RetryManager,
    tool_name: str = "",
    session_id: int | None = None,
):
    """Convenience function to create a retry decorator.

    Args:
        retry_manager: RetryManager instance
        tool_name: Name of the tool
        session_id: Optional database session ID

    Returns:
        Decorator function
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_manager.execute_with_retry(
                func,
                tool_name,
                *args,
                session_id=session_id,
                **kwargs,
            )

        return wrapper

    return decorator
