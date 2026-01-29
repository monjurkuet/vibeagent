"""Self-correction system for intelligent error recovery and strategy adaptation."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .skill import SkillResult

logger = logging.getLogger(__name__)


class CorrectionTrigger(Enum):
    """Triggers for self-correction."""

    REPEATED_FAILURES = "repeated_failures"
    UNEXPECTED_RESULT = "unexpected_result"
    CONFIDENCE_DROP = "confidence_drop"
    STAGNATION = "stagnation"
    ERROR_PATTERN = "error_pattern"
    TIMEOUT = "timeout"
    VALIDATION_FAILURE = "validation_failure"


class CorrectionType(Enum):
    """Types of corrections that can be applied."""

    DIFFERENT_TOOL = "different_tool"
    MODIFIED_PARAMETERS = "modified_parameters"
    DIFFERENT_ORDER = "different_order"
    ASK_USER = "ask_user"
    RETRY_WITH_DELAY = "retry_with_delay"
    FALLBACK_STRATEGY = "fallback_strategy"
    SKIP_STEP = "skip_step"
    BREAK_DOWN_TASK = "break_down_task"


class ErrorPattern(Enum):
    """Common error patterns for recognition."""

    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


@dataclass
class CorrectionStrategy:
    """A potential correction strategy."""

    strategy_type: CorrectionType
    description: str
    tool_name: str | None = None
    new_parameters: dict | None = None
    new_tool: str | None = None
    delay_ms: int | None = None
    confidence_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert strategy to dictionary."""
        return {
            "strategy_type": self.strategy_type.value,
            "description": self.description,
            "tool_name": self.tool_name,
            "new_parameters": self.new_parameters,
            "new_tool": self.new_tool,
            "delay_ms": self.delay_ms,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }


@dataclass
class CorrectionAttempt:
    """Record of a correction attempt."""

    attempt_number: int
    trigger: CorrectionTrigger
    strategy: CorrectionStrategy
    original_error: str | None = None
    success: bool = False
    execution_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    reflection_summary: str | None = None
    result_before: dict | None = None
    result_after: dict | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert attempt to dictionary."""
        return {
            "attempt_number": self.attempt_number,
            "trigger": self.trigger.value,
            "strategy": self.strategy.to_dict(),
            "original_error": self.original_error,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "reflection_summary": self.reflection_summary,
            "result_before": self.result_before,
            "result_after": self.result_after,
            "metadata": self.metadata,
        }


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for current approach."""

    current_confidence: float = 0.8
    historical_confidence: float = 0.8
    confidence_trend: float = 0.0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "current_confidence": self.current_confidence,
            "historical_confidence": self.historical_confidence,
            "confidence_trend": self.confidence_trend,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SelfCorrectorConfig:
    """Configuration for self-corrector."""

    max_correction_attempts: int = 3
    consecutive_error_threshold: int = 2
    stagnation_threshold: int = 5
    confidence_threshold: float = 0.5
    confidence_drop_threshold: float = 0.2
    enable_llm_guidance: bool = True
    enable_pattern_learning: bool = True
    strategy_diversity_factor: float = 0.3
    min_confidence_for_correction: float = 0.3
    timeout_correction_ms: int = 30000
    reflection_depth: int = 3
    auto_apply_confidence: float = 0.7

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "max_correction_attempts": self.max_correction_attempts,
            "consecutive_error_threshold": self.consecutive_error_threshold,
            "stagnation_threshold": self.stagnation_threshold,
            "confidence_threshold": self.confidence_threshold,
            "confidence_drop_threshold": self.confidence_drop_threshold,
            "enable_llm_guidance": self.enable_llm_guidance,
            "enable_pattern_learning": self.enable_pattern_learning,
            "strategy_diversity_factor": self.strategy_diversity_factor,
            "min_confidence_for_correction": self.min_confidence_for_correction,
            "timeout_correction_ms": self.timeout_correction_ms,
            "reflection_depth": self.reflection_depth,
            "auto_apply_confidence": self.auto_apply_confidence,
        }


class SelfCorrector:
    """Self-correction system for intelligent error recovery."""

    ERROR_PATTERNS = {
        ErrorPattern.NETWORK_ERROR: [
            r"network.*error",
            r"connection.*refused",
            r"connection.*reset",
            r"host.*unreachable",
            r"dns.*error",
        ],
        ErrorPattern.TIMEOUT_ERROR: [
            r"timeout",
            r"timed out",
            r"operation.*timed",
        ],
        ErrorPattern.RATE_LIMIT: [
            r"rate.*limit",
            r"too.*many.*requests",
            r"429",
        ],
        ErrorPattern.INVALID_INPUT: [
            r"invalid.*input",
            r"invalid.*parameter",
            r"bad.*request",
            r"400",
        ],
        ErrorPattern.AUTHENTICATION_ERROR: [
            r"authentication.*error",
            r"unauthorized",
            r"401",
            r"invalid.*token",
        ],
        ErrorPattern.VALIDATION_ERROR: [
            r"validation.*error",
            r"validation.*failed",
        ],
        ErrorPattern.NOT_FOUND: [
            r"not.*found",
            r"404",
            r"does.*not.*exist",
        ],
        ErrorPattern.PERMISSION_DENIED: [
            r"permission.*denied",
            r"access.*denied",
            r"403",
            r"forbidden",
        ],
        ErrorPattern.INTERNAL_ERROR: [
            r"internal.*error",
            r"500",
            r"server.*error",
        ],
    }

    def __init__(self, llm_skill=None, db_manager=None, config=None):
        """Initialize the self-corrector.

        Args:
            llm_skill: Optional LLMSkill for LLM-guided corrections
            db_manager: Optional DatabaseManager for tracking corrections
            config: Optional SelfCorrectorConfig
        """
        self.llm_skill = llm_skill
        self.db_manager = db_manager
        self.config = config or SelfCorrectorConfig()

        self._correction_attempts: list[CorrectionAttempt] = []
        self._error_history: dict[str, list[dict]] = {}
        self._pattern_success_cache: dict[str, float] = {}
        self._confidence_metrics = ConfidenceMetrics()
        self._iteration_count = 0
        self._last_success_iteration = 0
        self._consecutive_failures = 0

    def should_self_correct(
        self,
        context: dict,
    ) -> tuple[bool, CorrectionTrigger | None]:
        """Determine if self-correction is needed.

        Args:
            context: Context dictionary containing:
                - tool_result: SkillResult from last execution
                - iteration: Current iteration number
                - error_count: Dict of error counts
                - consecutive_errors: Number of consecutive errors
                - confidence: Current confidence score
                - result: Last result (if any)

        Returns:
            Tuple of (should_correct, trigger)
        """
        tool_result = context.get("tool_result")
        iteration = context.get("iteration", 0)
        consecutive_errors = context.get("consecutive_errors", 0)
        confidence = context.get("confidence", 1.0)

        if tool_result and not tool_result.success:
            if consecutive_errors >= self.config.consecutive_error_threshold:
                return True, CorrectionTrigger.REPEATED_FAILURES

            error_pattern = self.get_error_pattern(tool_result.error)
            if error_pattern != ErrorPattern.UNKNOWN:
                return True, CorrectionTrigger.ERROR_PATTERN

        if confidence < self.config.confidence_threshold:
            return True, CorrectionTrigger.CONFIDENCE_DROP

        confidence_drop = self._confidence_metrics.historical_confidence - confidence
        if confidence_drop >= self.config.confidence_drop_threshold:
            return True, CorrectionTrigger.CONFIDENCE_DROP

        if iteration - self._last_success_iteration >= self.config.stagnation_threshold:
            return True, CorrectionTrigger.STAGNATION

        result = context.get("result")
        if result and self._is_unexpected_result(result, context):
            return True, CorrectionTrigger.UNEXPECTED_RESULT

        return False, None

    def _is_unexpected_result(self, result: dict, context: dict) -> bool:
        """Check if result is unexpected.

        Args:
            result: Result to check
            context: Execution context

        Returns:
            True if result is unexpected
        """
        if not result:
            return False

        if result.get("error"):
            return False

        data = result.get("data", {})
        if not data:
            return True

        expected_type = context.get("expected_type")
        if expected_type and not isinstance(data, expected_type):
            return True

        expected_keys = context.get("expected_keys")
        if expected_keys and not all(k in data for k in expected_keys):
            return True

        return False

    def reflect_on_failure(self, error: str, context: dict) -> dict[str, Any]:
        """Analyze failure and generate reflection summary.

        Args:
            error: Error message
            context: Execution context

        Returns:
            Reflection dictionary with analysis
        """
        error_pattern = self.get_error_pattern(error)
        tool_name = context.get("tool_name", "unknown")
        iteration = context.get("iteration", 0)

        reflection = {
            "error": error,
            "error_pattern": error_pattern.value,
            "tool_name": tool_name,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "root_causes": [],
            "context_analysis": {},
            "suggested_corrections": [],
        }

        reflection["context_analysis"] = self._analyze_context(context)
        reflection["root_causes"] = self._identify_root_causes(error, context)

        if self.config.enable_llm_guidance and self.llm_skill:
            llm_reflection = self._get_llm_reflection(error, context)
            if llm_reflection:
                reflection["llm_analysis"] = llm_reflection

        return reflection

    def reflect_on_result(self, result: SkillResult, context: dict) -> dict[str, Any]:
        """Evaluate results and generate reflection.

        Args:
            result: Result to evaluate
            context: Execution context

        Returns:
            Reflection dictionary with analysis
        """
        reflection = {
            "success": result.success,
            "result_quality": self._evaluate_result_quality(result, context),
            "timestamp": datetime.now().isoformat(),
            "improvement_suggestions": [],
        }

        if result.success:
            self._confidence_metrics.success_rate = (
                self._confidence_metrics.success_rate * 0.9 + 0.1
            )
            self._confidence_metrics.current_confidence = min(
                self._confidence_metrics.current_confidence * 1.1, 1.0
            )
        else:
            self._confidence_metrics.success_rate = self._confidence_metrics.success_rate * 0.9
            self._confidence_metrics.current_confidence = max(
                self._confidence_metrics.current_confidence * 0.9, 0.0
            )

        reflection["confidence_metrics"] = self._confidence_metrics.to_dict()

        return reflection

    def _analyze_context(self, context: dict) -> dict:
        """Analyze execution context.

        Args:
            context: Context to analyze

        Returns:
            Context analysis
        """
        analysis = {
            "iteration": context.get("iteration", 0),
            "total_errors": context.get("total_errors", 0),
            "consecutive_errors": context.get("consecutive_errors", 0),
            "previous_attempts": len(self._correction_attempts),
        }

        tool_name = context.get("tool_name")
        if tool_name:
            analysis["tool_history"] = self._error_history.get(tool_name, [])

        return analysis

    def _identify_root_causes(self, error: str, context: dict) -> list[str]:
        """Identify potential root causes of error.

        Args:
            error: Error message
            context: Execution context

        Returns:
            List of potential root causes
        """
        root_causes = []

        error_pattern = self.get_error_pattern(error)

        if error_pattern == ErrorPattern.NETWORK_ERROR:
            root_causes.extend(
                [
                    "Network connectivity issue",
                    "Service unavailable",
                    "DNS resolution failure",
                ]
            )
        elif error_pattern == ErrorPattern.TIMEOUT_ERROR:
            root_causes.extend(
                [
                    "Operation taking too long",
                    "Service overloaded",
                    "Inefficient algorithm",
                ]
            )
        elif error_pattern == ErrorPattern.RATE_LIMIT:
            root_causes.extend(
                [
                    "Too many requests",
                    "API quota exceeded",
                    "Need to implement backoff",
                ]
            )
        elif error_pattern == ErrorPattern.INVALID_INPUT:
            root_causes.extend(
                [
                    "Incorrect parameter format",
                    "Missing required field",
                    "Invalid data type",
                ]
            )
        elif error_pattern == ErrorPattern.AUTHENTICATION_ERROR:
            root_causes.extend(
                [
                    "Invalid credentials",
                    "Expired token",
                    "Missing authentication",
                ]
            )
        elif error_pattern == ErrorPattern.VALIDATION_ERROR:
            root_causes.extend(
                [
                    "Data validation failed",
                    "Constraint violation",
                    "Business rule violation",
                ]
            )
        elif error_pattern == ErrorPattern.NOT_FOUND:
            root_causes.extend(
                [
                    "Resource does not exist",
                    "Incorrect identifier",
                    "Resource was deleted",
                ]
            )
        elif error_pattern == ErrorPattern.PERMISSION_DENIED:
            root_causes.extend(
                [
                    "Insufficient permissions",
                    "Access control violation",
                    "Authorization required",
                ]
            )
        else:
            root_causes.append("Unknown error - needs investigation")

        consecutive_errors = context.get("consecutive_errors", 0)
        if consecutive_errors > 1:
            root_causes.append(f"Repeated failure ({consecutive_errors} attempts)")

        return root_causes

    def _get_llm_reflection(self, error: str, context: dict) -> dict | None:
        """Get reflection from LLM.

        Args:
            error: Error message
            context: Execution context

        Returns:
            LLM reflection or None
        """
        if not self.llm_skill:
            return None

        try:
            prompt = f"""Analyze this error and provide insights:

Error: {error}
Context: {json.dumps(context, indent=2)}

Provide:
1. Root cause analysis
2. Suggested corrections
3. Alternative approaches

Format as JSON with keys: root_cause, suggestions, alternatives."""

            result = self.llm_skill.execute(prompt=prompt)

            if result.success and result.data:
                try:
                    return json.loads(result.data)
                except json.JSONDecodeError:
                    return {"analysis": result.data}

        except Exception as e:
            logger.warning(f"LLM reflection failed: {e}")

        return None

    def _evaluate_result_quality(self, result: SkillResult, context: dict) -> float:
        """Evaluate quality of result.

        Args:
            result: Result to evaluate
            context: Execution context

        Returns:
            Quality score (0-1)
        """
        if not result.success:
            return 0.0

        quality = 0.5

        data = result.data
        if data:
            quality += 0.3

        if not result.error:
            quality += 0.2

        return min(quality, 1.0)

    def generate_alternatives(self, context: dict) -> list[CorrectionStrategy]:
        """Generate alternative correction strategies.

        Args:
            context: Execution context with:
                - tool_result: Last tool result
                - tool_name: Current tool name
                - parameters: Current parameters
                - error: Error message

        Returns:
            List of alternative strategies
        """
        strategies = []
        tool_result = context.get("tool_result")
        tool_name = context.get("tool_name", "unknown")
        parameters = context.get("parameters", {})
        error = context.get("error", "")

        error_pattern = self.get_error_pattern(error) if error else ErrorPattern.UNKNOWN

        similar_solutions = self.get_similar_past_solutions(error)
        if similar_solutions:
            for solution in similar_solutions[:2]:
                strategies.append(
                    CorrectionStrategy(
                        strategy_type=CorrectionType.FALLBACK_STRATEGY,
                        description=f"Try previously successful approach: {solution['description']}",
                        confidence_score=solution.get("success_rate", 0.6),
                        metadata={"from_pattern": True},
                    )
                )

        if error_pattern in [ErrorPattern.NETWORK_ERROR, ErrorPattern.TIMEOUT_ERROR]:
            strategies.append(
                CorrectionStrategy(
                    strategy_type=CorrectionType.RETRY_WITH_DELAY,
                    description=f"Retry with exponential backoff ({self.config.timeout_correction_ms}ms)",
                    delay_ms=self.config.timeout_correction_ms,
                    confidence_score=0.7,
                )
            )

        if error_pattern == ErrorPattern.INVALID_INPUT:
            strategies.append(
                CorrectionStrategy(
                    strategy_type=CorrectionType.MODIFIED_PARAMETERS,
                    description="Modify parameters to fix validation issues",
                    tool_name=tool_name,
                    new_parameters=self._suggest_parameter_fixes(parameters, error),
                    confidence_score=0.6,
                )
            )

        if error_pattern == ErrorPattern.RATE_LIMIT:
            strategies.append(
                CorrectionStrategy(
                    strategy_type=CorrectionType.DIFFERENT_ORDER,
                    description="Reorder operations to reduce concurrent requests",
                    confidence_score=0.5,
                )
            )

        if self._consecutive_failures >= self.config.max_correction_attempts:
            strategies.append(
                CorrectionStrategy(
                    strategy_type=CorrectionType.ASK_USER,
                    description="Request user guidance for this situation",
                    confidence_score=0.8,
                )
            )

        if self.config.enable_llm_guidance and self.llm_skill:
            llm_strategies = self._get_llm_strategies(context)
            strategies.extend(llm_strategies)

        strategies = self._score_and_rank_strategies(strategies, context)

        return strategies

    def _suggest_parameter_fixes(self, parameters: dict, error: str) -> dict:
        """Suggest fixes for invalid parameters.

        Args:
            parameters: Current parameters
            error: Error message

        Returns:
            Suggested parameter fixes
        """
        fixes = {}

        for key, value in parameters.items():
            if value is None:
                fixes[key] = ""

            if isinstance(value, str) and not value.strip():
                fixes[key] = "default_value"

        if "required" in error.lower() or "missing" in error.lower():
            for key in parameters:
                if parameters[key] is None:
                    fixes[key] = ""

        return fixes

    def _get_llm_strategies(self, context: dict) -> list[CorrectionStrategy]:
        """Get alternative strategies from LLM.

        Args:
            context: Execution context

        Returns:
            List of LLM-suggested strategies
        """
        if not self.llm_skill:
            return []

        try:
            prompt = f"""Given this error and context, suggest 3 alternative strategies:

Error: {context.get("error", "Unknown")}
Tool: {context.get("tool_name", "Unknown")}
Context: {json.dumps(context, indent=2)}

For each strategy provide:
- type (different_tool, modified_parameters, different_order, retry_with_delay, fallback_strategy)
- description
- confidence score (0-1)

Format as JSON array."""

            result = self.llm_skill.execute(prompt=prompt)

            if result.success and result.data:
                try:
                    suggestions = json.loads(result.data)
                    strategies = []
                    for s in suggestions:
                        strategies.append(
                            CorrectionStrategy(
                                strategy_type=CorrectionType(s.get("type", "fallback_strategy")),
                                description=s.get("description", ""),
                                confidence_score=s.get("confidence_score", 0.5),
                                metadata={"from_llm": True},
                            )
                        )
                    return strategies
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.warning(f"LLM strategy generation failed: {e}")

        return []

    def _score_and_rank_strategies(
        self, strategies: list[CorrectionStrategy], context: dict
    ) -> list[CorrectionStrategy]:
        """Score and rank strategies by likelihood of success.

        Args:
            strategies: List of strategies
            context: Execution context

        Returns:
            Ranked list of strategies
        """
        for strategy in strategies:
            strategy.confidence_score = self.score_strategy(strategy, context)

        strategies.sort(key=lambda s: s.confidence_score, reverse=True)

        return strategies

    def score_strategy(self, strategy: CorrectionStrategy, context: dict) -> float:
        """Score strategy by likelihood of success.

        Args:
            strategy: Strategy to score
            context: Execution context

        Returns:
            Confidence score (0-1)
        """
        score = strategy.confidence_score

        error_pattern = context.get("error_pattern")
        if error_pattern:
            pattern_key = f"{strategy.strategy_type.value}_{error_pattern}"
            if pattern_key in self._pattern_success_cache:
                historical_score = self._pattern_success_cache[pattern_key]
                score = (score + historical_score) / 2

        if strategy.metadata.get("from_pattern"):
            score *= 1.2

        if strategy.metadata.get("from_llm"):
            score *= 1.1

        tool_name = context.get("tool_name")
        if tool_name and strategy.tool_name == tool_name:
            score *= 0.9

        consecutive_failures = context.get("consecutive_failures", 0)
        if consecutive_failures > 0:
            score *= max(0.5, 1.0 - consecutive_failures * 0.1)

        return min(score, 1.0)

    def select_best_strategy(
        self, strategies: list[CorrectionStrategy]
    ) -> CorrectionStrategy | None:
        """Select best strategy from alternatives.

        Args:
            strategies: List of strategies

        Returns:
            Best strategy or None
        """
        if not strategies:
            return None

        strategies = self._score_and_rank_strategies(strategies, {})

        best = strategies[0]

        if best.confidence_score < self.config.min_confidence_for_correction:
            return None

        return best

    def apply_correction(self, correction: CorrectionStrategy, context: dict) -> SkillResult:
        """Apply chosen correction.

        Args:
            correction: Correction to apply
            context: Execution context

        Returns:
            Result of correction application
        """
        attempt_number = len(self._correction_attempts) + 1
        start_time = time.time()

        result_before = context.get("tool_result")
        original_error = context.get("error")

        def skill_result_to_dict(result):
            """Convert SkillResult to dict."""
            if result is None:
                return None
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "metadata": result.metadata,
            }

        try:
            if correction.strategy_type == CorrectionType.DIFFERENT_TOOL:
                result = self._apply_different_tool(correction, context)
            elif correction.strategy_type == CorrectionType.MODIFIED_PARAMETERS:
                result = self._apply_modified_parameters(correction, context)
            elif correction.strategy_type == CorrectionType.DIFFERENT_ORDER:
                result = self._apply_different_order(context)
            elif correction.strategy_type == CorrectionType.ASK_USER:
                result = self._apply_ask_user(context)
            elif correction.strategy_type == CorrectionType.RETRY_WITH_DELAY:
                result = self._apply_retry_with_delay(correction, context)
            elif correction.strategy_type == CorrectionType.FALLBACK_STRATEGY:
                result = self._apply_fallback_strategy(correction, context)
            else:
                result = SkillResult(
                    success=False,
                    error=f"Unknown correction type: {correction.strategy_type}",
                )

        except Exception as e:
            result = SkillResult(success=False, error=f"Correction failed: {str(e)}")

        execution_time_ms = int((time.time() - start_time) * 1000)

        attempt = CorrectionAttempt(
            attempt_number=attempt_number,
            trigger=CorrectionTrigger.REPEATED_FAILURES,
            strategy=correction,
            original_error=original_error,
            success=result.success,
            execution_time_ms=execution_time_ms,
            result_before=skill_result_to_dict(result_before),
            result_after=skill_result_to_dict(result),
        )

        self._correction_attempts.append(attempt)

        if result.success:
            self._consecutive_failures = 0
            self._last_success_iteration = context.get("iteration", 0)
        else:
            self._consecutive_failures += 1

        if self.config.enable_pattern_learning:
            self._learn_from_correction(attempt, context)

        if self.db_manager:
            self._track_correction(attempt, context)

        return result

    def _apply_different_tool(self, correction: CorrectionStrategy, context: dict) -> SkillResult:
        """Apply different tool correction.

        Args:
            correction: Correction to apply
            context: Execution context

        Returns:
            Result
        """
        new_tool = correction.new_tool
        if not new_tool:
            return SkillResult(success=False, error="No alternative tool specified")

        context["tool_name"] = new_tool

        return SkillResult(
            success=True,
            data={"message": f"Switched to tool: {new_tool}"},
        )

    def _apply_modified_parameters(
        self, correction: CorrectionStrategy, context: dict
    ) -> SkillResult:
        """Apply modified parameters correction.

        Args:
            correction: Correction to apply
            context: Execution context

        Returns:
            Result
        """
        new_parameters = correction.new_parameters or {}
        context["parameters"] = {**context.get("parameters", {}), **new_parameters}

        return SkillResult(
            success=True,
            data={"message": "Parameters modified", "new_parameters": new_parameters},
        )

    def _apply_different_order(self, context: dict) -> SkillResult:
        """Apply different order correction.

        Args:
            context: Execution context

        Returns:
            Result
        """
        return SkillResult(
            success=True,
            data={"message": "Operation order changed"},
        )

    def _apply_ask_user(self, context: dict) -> SkillResult:
        """Apply ask user correction.

        Args:
            context: Execution context

        Returns:
            Result
        """
        return SkillResult(
            success=True,
            data={"message": "User guidance requested"},
        )

    def _apply_retry_with_delay(self, correction: CorrectionStrategy, context: dict) -> SkillResult:
        """Apply retry with delay correction.

        Args:
            correction: Correction to apply
            context: Execution context

        Returns:
            Result
        """
        delay_ms = correction.delay_ms or self.config.timeout_correction_ms
        time.sleep(delay_ms / 1000.0)

        return SkillResult(
            success=True,
            data={"message": f"Retried after {delay_ms}ms delay"},
        )

    def _apply_fallback_strategy(
        self, correction: CorrectionStrategy, context: dict
    ) -> SkillResult:
        """Apply fallback strategy correction.

        Args:
            correction: Correction to apply
            context: Execution context

        Returns:
            Result
        """
        return SkillResult(
            success=True,
            data={"message": f"Applied fallback: {correction.description}"},
        )

    def _learn_from_correction(self, attempt: CorrectionAttempt, context: dict):
        """Learn from correction attempt.

        Args:
            attempt: Correction attempt
            context: Execution context
        """
        error_pattern = context.get("error_pattern", "unknown")
        pattern_key = f"{attempt.strategy.strategy_type.value}_{error_pattern}"

        if pattern_key not in self._pattern_success_cache:
            self._pattern_success_cache[pattern_key] = 0.5

        learning_rate = 0.1
        if attempt.success:
            self._pattern_success_cache[pattern_key] += learning_rate * (
                1.0 - self._pattern_success_cache[pattern_key]
            )
        else:
            self._pattern_success_cache[pattern_key] -= learning_rate * (
                self._pattern_success_cache[pattern_key] - 0.0
            )

        self._pattern_success_cache[pattern_key] = max(
            0.0, min(1.0, self._pattern_success_cache[pattern_key])
        )

    def _track_correction(self, attempt: CorrectionAttempt, context: dict):
        """Track correction in database.

        Args:
            attempt: Correction attempt
            context: Execution context
        """
        if not self.db_manager:
            return

        try:
            session_id = context.get("session_id")
            if not session_id:
                return

            self.db_manager.add_self_correction(
                session_id=session_id,
                attempt_number=attempt.attempt_number,
                trigger=attempt.trigger.value,
                strategy_type=attempt.strategy.strategy_type.value,
                strategy_description=attempt.strategy.description,
                original_error=attempt.original_error,
                success=attempt.success,
                execution_time_ms=attempt.execution_time_ms,
                reflection_summary=attempt.reflection_summary,
                metadata=json.dumps(attempt.to_dict()),
            )
        except Exception as e:
            logger.error(f"Failed to track correction: {e}")

    def get_error_pattern(self, error: str | None) -> ErrorPattern:
        """Identify error pattern from error message.

        Args:
            error: Error message

        Returns:
            Error pattern
        """
        if not error:
            return ErrorPattern.UNKNOWN

        error_lower = error.lower()

        for pattern, patterns in self.ERROR_PATTERNS.items():
            for pattern_str in patterns:
                if re.search(pattern_str, error_lower):
                    return pattern

        return ErrorPattern.UNKNOWN

    def get_similar_past_solutions(self, error: str) -> list[dict]:
        """Find similar past solutions from history.

        Args:
            error: Current error

        Returns:
            List of similar past solutions
        """
        error_pattern = self.get_error_pattern(error)
        solutions = []

        for attempt in self._correction_attempts:
            if attempt.success and attempt.original_error:
                attempt_pattern = self.get_error_pattern(attempt.original_error)
                if attempt_pattern == error_pattern:
                    solutions.append(
                        {
                            "description": attempt.strategy.description,
                            "strategy_type": attempt.strategy.strategy_type.value,
                            "success_rate": 1.0,
                        }
                    )

        return solutions

    def calculate_confidence(self, context: dict) -> float:
        """Calculate confidence in current approach.

        Args:
            context: Execution context

        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.8

        consecutive_errors = context.get("consecutive_errors", 0)
        base_confidence *= max(0.3, 1.0 - consecutive_errors * 0.2)

        total_errors = context.get("total_errors", 0)
        base_confidence *= max(0.5, 1.0 - total_errors * 0.05)

        iteration = context.get("iteration", 0)
        if iteration > self.config.stagnation_threshold:
            base_confidence *= 0.8

        tool_name = context.get("tool_name")
        if tool_name:
            tool_errors = self._error_history.get(tool_name, [])
            if tool_errors:
                error_rate = len([e for e in tool_errors if not e.get("success", True)]) / len(
                    tool_errors
                )
                base_confidence *= max(0.3, 1.0 - error_rate)

        return max(0.0, min(1.0, base_confidence))

    def get_correction_statistics(self) -> dict:
        """Get statistics about correction attempts.

        Returns:
            Statistics dictionary
        """
        if not self._correction_attempts:
            return {
                "total_attempts": 0,
                "successful_corrections": 0,
                "failed_corrections": 0,
                "success_rate": 0.0,
                "by_trigger": {},
                "by_strategy": {},
            }

        stats = {
            "total_attempts": len(self._correction_attempts),
            "successful_corrections": sum(1 for a in self._correction_attempts if a.success),
            "failed_corrections": sum(1 for a in self._correction_attempts if not a.success),
            "by_trigger": {},
            "by_strategy": {},
        }

        stats["success_rate"] = (
            stats["successful_corrections"] / stats["total_attempts"]
            if stats["total_attempts"] > 0
            else 0.0
        )

        for attempt in self._correction_attempts:
            trigger = attempt.trigger.value
            stats["by_trigger"][trigger] = stats["by_trigger"].get(trigger, 0) + 1

            strategy = attempt.strategy.strategy_type.value
            stats["by_strategy"][strategy] = stats["by_strategy"].get(strategy, 0) + 1

        return stats

    def reset(self):
        """Reset self-corrector state."""
        self._correction_attempts = []
        self._error_history = {}
        self._iteration_count = 0
        self._last_success_iteration = 0
        self._consecutive_failures = 0
        self._confidence_metrics = ConfidenceMetrics()
