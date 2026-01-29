"""Example usage of RetryManager."""


from core.database_manager import DatabaseManager
from core.retry_manager import (
    BackoffStrategy,
    ErrorType,
    RetryManager,
    RetryPolicy,
)
from core.skill import SkillResult


def example_basic_usage():
    """Basic retry manager usage."""
    print("=== Basic Usage ===")

    # Create retry manager
    retry_manager = RetryManager()

    # Define a flaky function
    call_count = 0

    def flaky_api_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Connection failed")
        return SkillResult(success=True, data={"result": "success"})

    # Execute with retry
    result = retry_manager.execute_with_retry(
        flaky_api_call,
        tool_name="api_call",
    )

    print(f"Result: {result.success}")
    print(f"Total calls: {call_count}")
    print(f"Statistics: {retry_manager.get_statistics()}")


def example_custom_policy():
    """Using custom retry policies."""
    print("\n=== Custom Policy ===")

    # Create retry manager with custom policy
    custom_policy = RetryPolicy(
        max_retries=5,
        base_delay_ms=2000,
        max_delay_ms=60000,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        jitter_enabled=True,
        jitter_factor=0.2,
    )

    retry_manager = RetryManager()
    retry_manager.global_policy = custom_policy

    # Set tool-specific policy
    arxiv_policy = RetryPolicy(
        max_retries=7,
        base_delay_ms=3000,
        max_delay_ms=120000,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
    )
    retry_manager.TOOL_SPECIFIC_POLICIES["arxiv_search_papers"] = arxiv_policy

    print(f"Global policy: {retry_manager.global_policy.to_dict()}")
    print(f"ArXiv policy: {retry_manager.get_retry_policy('arxiv_search_papers').to_dict()}")


def example_error_classification():
    """Error classification and retryable detection."""
    print("\n=== Error Classification ===")

    retry_manager = RetryManager()

    # Test different error types
    errors = [
        Exception("Connection failed"),
        Exception("Request timed out"),
        Exception("429 Too Many Requests"),
        Exception("Invalid parameter value"),
        Exception("403 Forbidden"),
        Exception("404 Not Found"),
    ]

    for error in errors:
        error_type = retry_manager.classify_error(error)
        is_retryable = retry_manager.is_retryable(error)
        print(f"Error: {str(error):40} | Type: {error_type.value:15} | Retryable: {is_retryable}")


def example_backoff_strategies():
    """Different backoff strategies."""
    print("\n=== Backoff Strategies ===")

    retry_manager = RetryManager()
    policy = RetryPolicy(
        base_delay_ms=1000,
        max_delay_ms=30000,
        jitter_enabled=False,
    )

    strategies = [
        BackoffStrategy.EXPONENTIAL,
        BackoffStrategy.LINEAR,
        BackoffStrategy.FIXED,
    ]

    for strategy in strategies:
        print(f"\n{strategy.value.upper()} Backoff:")
        for attempt in range(5):
            delay = retry_manager.calculate_backoff(attempt, policy, strategy)
            print(f"  Attempt {attempt}: {delay}ms")


def example_database_integration():
    """Integration with DatabaseManager."""
    print("\n=== Database Integration ===")

    # Create database manager
    db_manager = DatabaseManager()

    # Create retry manager with database
    retry_manager = RetryManager(db_manager=db_manager)

    # Define a function that fails then succeeds
    call_count = 0

    def database_tracked_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("Network error")
        return SkillResult(success=True, data={"count": call_count})

    # Execute with database tracking
    result = retry_manager.execute_with_retry(
        database_tracked_function,
        tool_name="db_tracked_tool",
        session_id=1,
        tool_call_id=1,
    )

    print(f"Result: {result.success}")
    print("Retry attempts tracked in database")
    print(f"Statistics: {retry_manager.get_statistics()}")


def example_tool_specific_rules():
    """Tool-specific retry rules."""
    print("\n=== Tool-Specific Rules ===")

    retry_manager = RetryManager()

    # Add custom retry rules for a tool
    retry_manager.add_tool_retry_rule(
        "custom_tool",
        [ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.TEMPORARY],
    )

    # Test custom rules
    error = Exception("Connection failed")
    is_retryable = retry_manager.is_retryable(error, "custom_tool")
    print(f"Custom tool retryable: {is_retryable}")

    # Test with different tool
    is_retryable_default = retry_manager.is_retryable(error, "default_tool")
    print(f"Default tool retryable: {is_retryable_default}")


def example_model_specific_policies():
    """Model-specific retry policies."""
    print("\n=== Model-Specific Policies ===")

    retry_manager = RetryManager()

    # Set different policies for different models
    gpt4_policy = RetryPolicy(max_retries=5, base_delay_ms=2000)
    claude_policy = RetryPolicy(max_retries=3, base_delay_ms=1000)

    retry_manager.set_model_retry_policy("gpt-4", gpt4_policy)
    retry_manager.set_model_retry_policy("claude-3", claude_policy)

    print(f"GPT-4 policy: {retry_manager.get_model_retry_policy('gpt-4').to_dict()}")
    print(f"Claude-3 policy: {retry_manager.get_model_retry_policy('claude-3').to_dict()}")
    print(f"Default policy: {retry_manager.get_model_retry_policy('unknown').to_dict()}")


def example_decorator_usage():
    """Using retry decorator."""
    print("\n=== Decorator Usage ===")

    retry_manager = RetryManager()

    call_count = 0

    @retry_manager.retry_decorator(tool_name="decorated_tool")
    def decorated_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("Temporary failure")
        return SkillResult(success=True, data={"attempts": call_count})

    result = decorated_function()
    print(f"Result: {result.success}")
    print(f"Data: {result.data}")
    print(f"Statistics: {retry_manager.get_statistics()}")


def example_statistics_analysis():
    """Statistics and analysis."""
    print("\n=== Statistics Analysis ===")

    retry_manager = RetryManager()

    # Simulate multiple retry scenarios
    for i in range(10):
        call_count = 0

        def simulated_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Network error")
            return SkillResult(success=True, data={"result": "success"})

        retry_manager.execute_with_retry(
            simulated_function,
            tool_name=f"tool_{i % 3}",
        )

    # Get statistics
    stats = retry_manager.get_statistics()
    print(f"Total retries: {stats['total_retries']}")
    print(f"Successful retries: {stats['successful_retries']}")
    print(f"Failed retries: {stats['failed_retries']}")
    print(f"Retry success rate: {stats['retry_success_rate']:.2%}")
    print(f"Retries by tool: {stats['retries_by_tool']}")
    print(f"Retries by error type: {stats['retries_by_error_type']}")

    # Get recovery rate
    recovery_rate = retry_manager.get_recovery_rate()
    print(f"Overall recovery rate: {recovery_rate:.2f}%")

    # Calculate optimal retry limits
    optimal_limits = retry_manager.calculate_optimal_retry_limits()
    print(f"Optimal retry limits: {optimal_limits}")


def example_attempt_history():
    """Retrieve retry attempt history."""
    print("\n=== Attempt History ===")

    retry_manager = RetryManager()

    # Execute some retries
    call_count = 0

    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Network error")
        return SkillResult(success=True, data={"result": "success"})

    retry_manager.execute_with_retry(flaky_function, tool_name="test_tool")

    # Get attempt history
    history = retry_manager.get_attempt_history(tool_name="test_tool", limit=10)
    print("Attempt history for 'test_tool':")
    for attempt in history:
        print(
            f"  Attempt {attempt['attempt_number']}: "
            f"{attempt['error_type']} - {attempt['success']}"
        )


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_custom_policy()
    example_error_classification()
    example_backoff_strategies()
    # example_database_integration()  # Uncomment if database is set up
    example_tool_specific_rules()
    example_model_specific_policies()
    example_decorator_usage()
    example_statistics_analysis()
    example_attempt_history()

    print("\n=== All examples completed ===")
