"""Tests for self-correction system."""

import pytest
from core.self_corrector import (
    CorrectionStrategy,
    CorrectionTrigger,
    CorrectionType,
    ErrorPattern,
    SelfCorrector,
    SelfCorrectorConfig,
)
from core.skill import SkillResult


class TestCorrectionTriggerDetection:
    """Test correction trigger detection."""

    def test_should_correct_repeated_failures(self):
        """Test correction triggered by repeated failures."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(consecutive_error_threshold=2))

        context = {
            "tool_result": SkillResult(success=False, error="API error"),
            "consecutive_errors": 3,
            "iteration": 5,
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is True
        assert trigger == CorrectionTrigger.REPEATED_FAILURES

    def test_should_not_correct_single_failure(self):
        """Test no correction for single failure."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(consecutive_error_threshold=2))

        context = {
            "tool_result": SkillResult(success=False, error="API error"),
            "consecutive_errors": 1,
            "iteration": 2,
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is False
        assert trigger is None

    def test_should_correct_confidence_drop(self):
        """Test correction triggered by confidence drop."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(confidence_threshold=0.6))

        context = {
            "tool_result": SkillResult(success=True, data={"result": "ok"}),
            "consecutive_errors": 0,
            "iteration": 3,
            "confidence": 0.4,
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is True
        assert trigger == CorrectionTrigger.CONFIDENCE_DROP

    def test_should_correct_stagnation(self):
        """Test correction triggered by stagnation."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(stagnation_threshold=3))

        corrector._last_success_iteration = 0

        context = {
            "tool_result": SkillResult(success=True, data={"result": "ok"}),
            "consecutive_errors": 0,
            "iteration": 5,
            "confidence": 0.8,
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is True
        assert trigger == CorrectionTrigger.STAGNATION


class TestErrorPatternRecognition:
    """Test error pattern recognition."""

    def test_network_error_pattern(self):
        """Test network error pattern detection."""
        corrector = SelfCorrector()

        errors = [
            "Network connection refused",
            "Connection reset by peer",
            "Host unreachable",
        ]

        for error in errors:
            pattern = corrector.get_error_pattern(error)
            assert pattern == ErrorPattern.NETWORK_ERROR

    def test_timeout_error_pattern(self):
        """Test timeout error pattern detection."""
        corrector = SelfCorrector()

        errors = [
            "Request timed out after 30s",
            "Operation timed out",
            "Timeout waiting for response",
        ]

        for error in errors:
            pattern = corrector.get_error_pattern(error)
            assert pattern == ErrorPattern.TIMEOUT_ERROR

    def test_rate_limit_pattern(self):
        """Test rate limit pattern detection."""
        corrector = SelfCorrector()

        errors = [
            "Rate limit exceeded (429)",
            "Too many requests",
            "API quota exceeded",
        ]

        for error in errors:
            pattern = corrector.get_error_pattern(error)
            assert pattern == ErrorPattern.RATE_LIMIT

    def test_validation_error_pattern(self):
        """Test validation error pattern detection."""
        corrector = SelfCorrector()

        errors = [
            "Invalid input parameter",
            "Validation failed for field 'name'",
            "Bad request: missing required field",
        ]

        for error in errors:
            pattern = corrector.get_error_pattern(error)
            assert pattern == ErrorPattern.INVALID_INPUT

    def test_unknown_error_pattern(self):
        """Test unknown error pattern."""
        corrector = SelfCorrector()

        error = "Some unknown error occurred"
        pattern = corrector.get_error_pattern(error)

        assert pattern == ErrorPattern.UNKNOWN


class TestReflectionEngine:
    """Test reflection engine."""

    def test_reflect_on_failure(self):
        """Test failure reflection."""
        corrector = SelfCorrector()

        error = "Connection timeout after 30 seconds"
        context = {
            "tool_name": "api_fetcher",
            "iteration": 3,
            "consecutive_errors": 2,
        }

        reflection = corrector.reflect_on_failure(error, context)

        assert "error" in reflection
        assert reflection["error"] == error
        assert "error_pattern" in reflection
        assert "root_causes" in reflection
        assert len(reflection["root_causes"]) > 0
        assert "context_analysis" in reflection

    def test_reflect_on_result_success(self):
        """Test result reflection for success."""
        corrector = SelfCorrector()

        result = SkillResult(success=True, data={"result": "success"})
        context = {}

        reflection = corrector.reflect_on_result(result, context)

        assert reflection["success"] is True
        assert "result_quality" in reflection
        assert "confidence_metrics" in reflection

    def test_reflect_on_result_failure(self):
        """Test result reflection for failure."""
        corrector = SelfCorrector()

        result = SkillResult(success=False, error="API error")
        context = {}

        reflection = corrector.reflect_on_result(result, context)

        assert reflection["success"] is False
        assert reflection["result_quality"] == 0.0


class TestAlternativeStrategyGeneration:
    """Test alternative strategy generation."""

    def test_generate_alternatives_for_timeout(self):
        """Test generating alternatives for timeout error."""
        corrector = SelfCorrector()

        context = {
            "error": "Request timed out after 30s",
            "tool_result": SkillResult(success=False, error="Timeout"),
            "tool_name": "api_fetcher",
            "parameters": {"url": "https://api.example.com"},
            "error_pattern": "timeout_error",
        }

        alternatives = corrector.generate_alternatives(context)

        assert len(alternatives) > 0

        retry_strategy = next(
            (s for s in alternatives if s.strategy_type == CorrectionType.RETRY_WITH_DELAY),
            None,
        )
        assert retry_strategy is not None

    def test_generate_alternatives_for_validation_error(self):
        """Test generating alternatives for validation error."""
        corrector = SelfCorrector()

        context = {
            "error": "Invalid input: missing required field",
            "tool_result": SkillResult(success=False, error="Invalid input"),
            "tool_name": "data_validator",
            "parameters": {"name": "test"},
            "error_pattern": "invalid_input",
        }

        alternatives = corrector.generate_alternatives(context)

        assert len(alternatives) > 0

        param_strategy = next(
            (s for s in alternatives if s.strategy_type == CorrectionType.MODIFIED_PARAMETERS),
            None,
        )
        assert param_strategy is not None

    def test_score_strategies(self):
        """Test strategy scoring."""
        corrector = SelfCorrector()

        strategies = [
            CorrectionStrategy(
                strategy_type=CorrectionType.RETRY_WITH_DELAY,
                description="Retry with delay",
                confidence_score=0.7,
            ),
            CorrectionStrategy(
                strategy_type=CorrectionType.DIFFERENT_TOOL,
                description="Use different tool",
                confidence_score=0.5,
            ),
        ]

        context = {"consecutive_failures": 1}

        scored = corrector._score_and_rank_strategies(strategies, context)

        assert len(scored) == 2
        assert scored[0].confidence_score >= scored[1].confidence_score

    def test_select_best_strategy(self):
        """Test selecting best strategy."""
        corrector = SelfCorrector()

        strategies = [
            CorrectionStrategy(
                strategy_type=CorrectionType.RETRY_WITH_DELAY,
                description="Retry with delay",
                confidence_score=0.8,
            ),
            CorrectionStrategy(
                strategy_type=CorrectionType.DIFFERENT_TOOL,
                description="Use different tool",
                confidence_score=0.6,
            ),
        ]

        best = corrector.select_best_strategy(strategies)

        assert best is not None
        assert best.strategy_type == CorrectionType.RETRY_WITH_DELAY
        assert best.confidence_score == 0.8


class TestCorrectionExecution:
    """Test correction execution."""

    def test_apply_retry_with_delay(self):
        """Test applying retry with delay correction."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(timeout_correction_ms=100))

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.RETRY_WITH_DELAY,
            description="Retry with delay",
            delay_ms=100,
        )

        context = {"iteration": 3}

        result = corrector.apply_correction(strategy, context)

        assert result.success is True
        assert len(corrector._correction_attempts) == 1

    def test_apply_modified_parameters(self):
        """Test applying modified parameters correction."""
        corrector = SelfCorrector()

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.MODIFIED_PARAMETERS,
            description="Fix parameters",
            new_parameters={"id": "123"},
        )

        context = {"parameters": {"name": "test"}}

        result = corrector.apply_correction(strategy, context)

        assert result.success is True
        assert "id" in context["parameters"]

    def test_apply_different_tool(self):
        """Test applying different tool correction."""
        corrector = SelfCorrector()

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.DIFFERENT_TOOL,
            description="Use alternative tool",
            new_tool="alternative_api",
        )

        context = {"tool_name": "original_api"}

        result = corrector.apply_correction(strategy, context)

        assert result.success is True
        assert context["tool_name"] == "alternative_api"

    def test_correction_tracking(self):
        """Test correction attempt tracking."""
        corrector = SelfCorrector()

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.RETRY_WITH_DELAY,
            description="Retry with delay",
        )

        context = {"iteration": 1}

        corrector.apply_correction(strategy, context)

        assert len(corrector._correction_attempts) == 1
        attempt = corrector._correction_attempts[0]
        assert attempt.attempt_number == 1
        assert attempt.strategy == strategy


class TestConfidenceScoring:
    """Test confidence scoring."""

    def test_calculate_confidence_high(self):
        """Test confidence calculation for high confidence scenario."""
        corrector = SelfCorrector()

        context = {
            "consecutive_errors": 0,
            "total_errors": 0,
            "iteration": 1,
        }

        confidence = corrector.calculate_confidence(context)

        assert confidence > 0.7

    def test_calculate_confidence_low(self):
        """Test confidence calculation for low confidence scenario."""
        corrector = SelfCorrector()

        context = {
            "consecutive_errors": 3,
            "total_errors": 5,
            "iteration": 6,
        }

        confidence = corrector.calculate_confidence(context)

        assert confidence < 0.5

    def test_confidence_metrics_update_on_success(self):
        """Test confidence metrics update on success."""
        corrector = SelfCorrector()

        result = SkillResult(success=True, data={"result": "ok"})
        context = {}

        initial_confidence = corrector._confidence_metrics.current_confidence

        corrector.reflect_on_result(result, context)

        assert corrector._confidence_metrics.current_confidence >= initial_confidence

    def test_confidence_metrics_update_on_failure(self):
        """Test confidence metrics update on failure."""
        corrector = SelfCorrector()

        result = SkillResult(success=False, error="API error")
        context = {}

        initial_confidence = corrector._confidence_metrics.current_confidence

        corrector.reflect_on_result(result, context)

        assert corrector._confidence_metrics.current_confidence <= initial_confidence


class TestPatternLearning:
    """Test pattern learning."""

    def test_learn_from_successful_correction(self):
        """Test learning from successful correction."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(enable_pattern_learning=True))

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.RETRY_WITH_DELAY,
            description="Retry with delay",
        )

        attempt = CorrectionAttempt(
            attempt_number=1,
            trigger=CorrectionTrigger.REPEATED_FAILURES,
            strategy=strategy,
            success=True,
        )

        context = {
            "error_pattern": "timeout_error",
        }

        initial_cache = corrector._pattern_success_cache.copy()

        corrector._learn_from_correction(attempt, context)

        pattern_key = "retry_with_delay_timeout_error"
        if pattern_key in corrector._pattern_success_cache:
            assert corrector._pattern_success_cache[pattern_key] >= initial_cache.get(
                pattern_key, 0.5
            )

    def test_learn_from_failed_correction(self):
        """Test learning from failed correction."""
        corrector = SelfCorrector(config=SelfCorrectorConfig(enable_pattern_learning=True))

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.DIFFERENT_TOOL,
            description="Use different tool",
        )

        attempt = CorrectionAttempt(
            attempt_number=1,
            trigger=CorrectionTrigger.REPEATED_FAILURES,
            strategy=strategy,
            success=False,
        )

        context = {
            "error_pattern": "network_error",
        }

        corrector._learn_from_correction(attempt, context)

        pattern_key = "different_tool_network_error"
        if pattern_key in corrector._pattern_success_cache:
            assert corrector._pattern_success_cache[pattern_key] <= 0.5


class TestSimilarPastSolutions:
    """Test finding similar past solutions."""

    def test_get_similar_past_solutions(self):
        """Test retrieving similar past solutions."""
        corrector = SelfCorrector()

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.RETRY_WITH_DELAY,
            description="Retry with delay",
        )

        attempt = CorrectionAttempt(
            attempt_number=1,
            trigger=CorrectionTrigger.REPEATED_FAILURES,
            strategy=strategy,
            success=True,
            original_error="Connection timeout",
        )

        corrector._correction_attempts.append(attempt)

        solutions = corrector.get_similar_past_solutions("Request timed out")

        assert len(solutions) > 0
        assert solutions[0]["strategy_type"] == "retry_with_delay"


class TestCorrectionStatistics:
    """Test correction statistics."""

    def test_get_correction_statistics_empty(self):
        """Test statistics when no corrections attempted."""
        corrector = SelfCorrector()

        stats = corrector.get_correction_statistics()

        assert stats["total_attempts"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_correction_statistics_with_attempts(self):
        """Test statistics with correction attempts."""
        corrector = SelfCorrector()

        for i in range(3):
            strategy = CorrectionStrategy(
                strategy_type=CorrectionType.RETRY_WITH_DELAY,
                description="Retry with delay",
            )

            attempt = CorrectionAttempt(
                attempt_number=i + 1,
                trigger=CorrectionTrigger.REPEATED_FAILURES,
                strategy=strategy,
                success=(i < 2),
            )

            corrector._correction_attempts.append(attempt)

        stats = corrector.get_correction_statistics()

        assert stats["total_attempts"] == 3
        assert stats["successful_corrections"] == 2
        assert stats["failed_corrections"] == 1
        assert stats["success_rate"] == 2 / 3


class TestSelfCorrectorReset:
    """Test self-corrector reset."""

    def test_reset(self):
        """Test resetting self-corrector state."""
        corrector = SelfCorrector()

        strategy = CorrectionStrategy(
            strategy_type=CorrectionType.RETRY_WITH_DELAY,
            description="Retry with delay",
        )

        attempt = CorrectionAttempt(
            attempt_number=1,
            trigger=CorrectionTrigger.REPEATED_FAILURES,
            strategy=strategy,
            success=True,
        )

        corrector._correction_attempts.append(attempt)
        corrector._consecutive_failures = 3
        corrector._iteration_count = 5

        corrector.reset()

        assert len(corrector._correction_attempts) == 0
        assert corrector._consecutive_failures == 0
        assert corrector._iteration_count == 0


class TestIntegration:
    """Integration tests."""

    def test_full_correction_flow(self):
        """Test complete correction flow."""
        corrector = SelfCorrector(
            config=SelfCorrectorConfig(
                consecutive_error_threshold=2,
                max_correction_attempts=3,
            )
        )

        context = {
            "tool_result": SkillResult(success=False, error="Connection timeout"),
            "tool_name": "api_fetcher",
            "iteration": 3,
            "consecutive_errors": 2,
            "parameters": {"url": "https://api.example.com"},
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is True

        reflection = corrector.reflect_on_failure(context["tool_result"].error, context)

        assert reflection is not None

        alternatives = corrector.generate_alternatives(context)

        assert len(alternatives) > 0

        best_strategy = corrector.select_best_strategy(alternatives)

        assert best_strategy is not None

        result = corrector.apply_correction(best_strategy, context)

        assert result is not None

        stats = corrector.get_correction_statistics()

        assert stats["total_attempts"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
