"""Enhanced error feedback system for VibeAgent."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Error type classification."""

    VALIDATION = "validation"
    EXECUTION = "execution"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"


class SeverityLevel(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Retryability(str, Enum):
    """Error retryability classification."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    CONDITIONAL = "conditional"


class RecoveryStrategy(str, Enum):
    """Recovery strategy types."""

    RETRY = "retry"
    MODIFY_PARAMS = "modify_params"
    TRY_ALTERNATIVE = "try_alternative"
    SKIP = "skip"
    ASK_USER = "ask_user"
    ABORT = "abort"


@dataclass
class ErrorClassification:
    """Error classification result."""

    error_type: ErrorType
    severity: SeverityLevel
    retryability: Retryability
    confidence: float
    description: str


@dataclass
class ErrorContext:
    """Rich error context."""

    tool_name: str
    parameters: dict[str, Any]
    attempt_number: int
    previous_attempts: list[dict[str, Any]] = field(default_factory=list)
    similar_errors: list[dict[str, Any]] = field(default_factory=list)
    successful_patterns: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecoverySuggestion:
    """Recovery strategy suggestion."""

    strategy: RecoveryStrategy
    description: str
    parameter_modifications: dict[str, Any] = field(default_factory=dict)
    alternative_tools: list[str] = field(default_factory=list)
    estimated_success_rate: float = 0.0
    confidence: float = 0.0


@dataclass
class ErrorPattern:
    """Error pattern for learning."""

    fingerprint: str
    error_type: ErrorType
    pattern_key: str
    recovery_strategies: dict[str, float]
    total_occurrences: int
    successful_recoveries: int
    last_seen: datetime
    created_at: datetime

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_occurrences == 0:
            return 0.0
        return (self.successful_recoveries / self.total_occurrences) * 100


class ErrorClassifier:
    """Classifies errors by type, severity, and retryability."""

    ERROR_PATTERNS = {
        "validation": {
            "keywords": ["invalid", "validation", "malformed", "schema", "format"],
            "severity": SeverityLevel.LOW,
            "retryability": Retryability.NON_RETRYABLE,
        },
        "execution": {
            "keywords": ["execution", "runtime", "exception", "failed"],
            "severity": SeverityLevel.MEDIUM,
            "retryability": Retryability.CONDITIONAL,
        },
        "network": {
            "keywords": ["network", "connection", "dns", "unreachable", "refused"],
            "severity": SeverityLevel.MEDIUM,
            "retryability": Retryability.RETRYABLE,
        },
        "timeout": {
            "keywords": ["timeout", "timed out", "deadline", "expired"],
            "severity": SeverityLevel.MEDIUM,
            "retryability": Retryability.RETRYABLE,
        },
        "permission": {
            "keywords": [
                "permission",
                "unauthorized",
                "forbidden",
                "access denied",
                "403",
            ],
            "severity": SeverityLevel.HIGH,
            "retryability": Retryability.NON_RETRYABLE,
        },
        "not_found": {
            "keywords": ["not found", "404", "does not exist", "no such"],
            "severity": SeverityLevel.LOW,
            "retryability": Retryability.NON_RETRYABLE,
        },
        "rate_limit": {
            "keywords": ["rate limit", "429", "too many requests", "quota"],
            "severity": SeverityLevel.MEDIUM,
            "retryability": Retryability.CONDITIONAL,
        },
        "internal": {
            "keywords": ["internal", "server", "500", "unexpected"],
            "severity": SeverityLevel.CRITICAL,
            "retryability": Retryability.RETRYABLE,
        },
    }

    def classify(
        self, error: Exception, context: ErrorContext | None = None
    ) -> ErrorClassification:
        """Classify an error.

        Args:
            error: The exception to classify
            context: Optional error context

        Returns:
            ErrorClassification with type, severity, and retryability
        """
        error_message = str(error).lower()

        best_match = None
        best_score = 0.0

        for error_type_name, pattern in self.ERROR_PATTERNS.items():
            score = 0.0
            keywords = pattern["keywords"]

            for keyword in keywords:
                if keyword in error_message:
                    score += 1.0

            if score > best_score:
                best_score = score
                best_match = (
                    ErrorType(error_type_name),
                    pattern["severity"],
                    pattern["retryability"],
                )

        if not best_match:
            best_match = (
                ErrorType.INTERNAL,
                SeverityLevel.MEDIUM,
                Retryability.RETRYABLE,
            )

        confidence = min(
            best_score / len(self.ERROR_PATTERNS.get(best_match[0].value, {}).get("keywords", [])),
            1.0,
        )

        description = f"Error classified as {best_match[0].value} with {best_match[1]} severity"

        return ErrorClassification(
            error_type=best_match[0],
            severity=best_match[1],
            retryability=best_match[2],
            confidence=confidence,
            description=description,
        )


class ErrorPatternDatabase:
    """Database for storing and learning from error patterns."""

    def __init__(self, db_manager: DatabaseManager | None = None):
        """Initialize pattern database.

        Args:
            db_manager: Optional DatabaseManager for persistence
        """
        self.db_manager = db_manager
        self._initialize_tables()
        self._memory_patterns: dict[str, ErrorPattern] = {}

    def _initialize_tables(self):
        """Initialize database tables for error patterns."""
        if not self.db_manager:
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS error_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fingerprint TEXT UNIQUE NOT NULL,
                        error_type TEXT NOT NULL,
                        pattern_key TEXT NOT NULL,
                        recovery_strategies TEXT NOT NULL,
                        total_occurrences INTEGER DEFAULT 1,
                        successful_recoveries INTEGER DEFAULT 0,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_error_patterns_fingerprint
                    ON error_patterns(fingerprint)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_error_patterns_type
                    ON error_patterns(error_type)
                """
                )

                logger.info("Error pattern database tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize error pattern tables: {e}")

    def get_pattern(self, fingerprint: str) -> ErrorPattern | None:
        """Get error pattern by fingerprint.

        Args:
            fingerprint: Error fingerprint

        Returns:
            ErrorPattern or None if not found
        """
        if fingerprint in self._memory_patterns:
            return self._memory_patterns[fingerprint]

        if not self.db_manager:
            return None

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT fingerprint, error_type, pattern_key, recovery_strategies,
                           total_occurrences, successful_recoveries, last_seen, created_at
                    FROM error_patterns
                    WHERE fingerprint = ?
                    """,
                    (fingerprint,),
                )
                row = cursor.fetchone()
                if row:
                    pattern = ErrorPattern(
                        fingerprint=row["fingerprint"],
                        error_type=ErrorType(row["error_type"]),
                        pattern_key=row["pattern_key"],
                        recovery_strategies=json.loads(row["recovery_strategies"]),
                        total_occurrences=row["total_occurrences"],
                        successful_recoveries=row["successful_recoveries"],
                        last_seen=datetime.fromisoformat(row["last_seen"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    self._memory_patterns[fingerprint] = pattern
                    return pattern
        except Exception as e:
            logger.error(f"Failed to get error pattern: {e}")

        return None

    def update_pattern(
        self,
        fingerprint: str,
        error_type: ErrorType,
        pattern_key: str,
        recovery_strategy: str,
        success: bool,
    ):
        """Update error pattern with new occurrence.

        Args:
            fingerprint: Error fingerprint
            error_type: Type of error
            pattern_key: Pattern key (e.g., tool_name:parameter_hash)
            recovery_strategy: Strategy attempted
            success: Whether recovery was successful
        """
        if not self.db_manager:
            if fingerprint in self._memory_patterns:
                pattern = self._memory_patterns[fingerprint]
                pattern.total_occurrences += 1
                if success:
                    pattern.successful_recoveries += 1
                if recovery_strategy not in pattern.recovery_strategies:
                    pattern.recovery_strategies[recovery_strategy] = 0.0
                pattern.recovery_strategies[recovery_strategy] = (
                    pattern.recovery_strategies[recovery_strategy] + (1.0 if success else 0.0)
                ) / pattern.total_occurrences
            else:
                self._memory_patterns[fingerprint] = ErrorPattern(
                    fingerprint=fingerprint,
                    error_type=error_type,
                    pattern_key=pattern_key,
                    recovery_strategies={recovery_strategy: 1.0 if success else 0.0},
                    total_occurrences=1,
                    successful_recoveries=1 if success else 0,
                    last_seen=datetime.now(),
                    created_at=datetime.now(),
                )
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT recovery_strategies, total_occurrences, successful_recoveries
                    FROM error_patterns
                    WHERE fingerprint = ?
                    """,
                    (fingerprint,),
                )
                row = cursor.fetchone()

                if row:
                    recovery_strategies = json.loads(row["recovery_strategies"])
                    total_occurrences = row["total_occurrences"] + 1
                    successful_recoveries = row["successful_recoveries"] + (1 if success else 0)

                    if recovery_strategy not in recovery_strategies:
                        recovery_strategies[recovery_strategy] = 0.0
                    recovery_strategies[recovery_strategy] = (
                        recovery_strategies[recovery_strategy] + (1.0 if success else 0.0)
                    ) / total_occurrences

                    cursor.execute(
                        """
                        UPDATE error_patterns
                        SET recovery_strategies = ?, total_occurrences = ?,
                            successful_recoveries = ?, last_seen = CURRENT_TIMESTAMP
                        WHERE fingerprint = ?
                        """,
                        (
                            json.dumps(recovery_strategies),
                            total_occurrences,
                            successful_recoveries,
                            fingerprint,
                        ),
                    )
                else:
                    recovery_strategies = {recovery_strategy: 1.0 if success else 0.0}
                    cursor.execute(
                        """
                        INSERT INTO error_patterns
                        (fingerprint, error_type, pattern_key, recovery_strategies,
                         total_occurrences, successful_recoveries)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fingerprint,
                            error_type.value,
                            pattern_key,
                            json.dumps(recovery_strategies),
                            1,
                            1 if success else 0,
                        ),
                    )

            logger.debug(f"Updated error pattern: {fingerprint}")
        except Exception as e:
            logger.error(f"Failed to update error pattern: {e}")

    def get_similar_patterns(self, error_type: ErrorType, limit: int = 5) -> list[ErrorPattern]:
        """Get similar error patterns by type.

        Args:
            error_type: Type of error
            limit: Maximum number of patterns to return

        Returns:
            List of similar ErrorPattern objects
        """
        if not self.db_manager:
            return [p for p in self._memory_patterns.values() if p.error_type == error_type][:limit]

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT fingerprint, error_type, pattern_key, recovery_strategies,
                           total_occurrences, successful_recoveries, last_seen, created_at
                    FROM error_patterns
                    WHERE error_type = ?
                    ORDER BY successful_recoveries DESC, total_occurrences DESC
                    LIMIT ?
                    """,
                    (error_type.value, limit),
                )
                patterns = []
                for row in cursor.fetchall():
                    patterns.append(
                        ErrorPattern(
                            fingerprint=row["fingerprint"],
                            error_type=ErrorType(row["error_type"]),
                            pattern_key=row["pattern_key"],
                            recovery_strategies=json.loads(row["recovery_strategies"]),
                            total_occurrences=row["total_occurrences"],
                            successful_recoveries=row["successful_recoveries"],
                            last_seen=datetime.fromisoformat(row["last_seen"]),
                            created_at=datetime.fromisoformat(row["created_at"]),
                        )
                    )
                return patterns
        except Exception as e:
            logger.error(f"Failed to get similar patterns: {e}")
            return []


class ErrorHandler:
    """Enhanced error handling system with classification and recovery."""

    def __init__(self, db_manager: DatabaseManager | None = None):
        """Initialize error handler.

        Args:
            db_manager: Optional DatabaseManager for persistence
        """
        self.classifier = ErrorClassifier()
        self.pattern_db = ErrorPatternDatabase(db_manager)
        self.db_manager = db_manager

    def get_error_fingerprint(self, error: Exception, context: ErrorContext) -> str:
        """Create unique error fingerprint.

        Args:
            error: The exception
            context: Error context

        Returns:
            Unique fingerprint string
        """
        error_str = f"{type(error).__name__}:{str(error)}:{context.tool_name}"
        if context.parameters:
            param_str = json.dumps(context.parameters, sort_keys=True)
            error_str += f":{param_str}"

        return hashlib.md5(error_str.encode()).hexdigest()[:16]

    def build_error_context(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        attempt_number: int = 1,
        previous_attempts: list[dict[str, Any]] | None = None,
    ) -> ErrorContext:
        """Build rich error context.

        Args:
            tool_name: Name of the tool being called
            parameters: Parameters used in the call
            attempt_number: Current attempt number
            previous_attempts: List of previous attempt details

        Returns:
            ErrorContext with rich information
        """
        context = ErrorContext(
            tool_name=tool_name,
            parameters=parameters,
            attempt_number=attempt_number,
            previous_attempts=previous_attempts or [],
        )

        if self.db_manager:
            similar_errors = self._find_similar_errors(tool_name, parameters)
            context.similar_errors = similar_errors

        return context

    def _find_similar_errors(
        self, tool_name: str, parameters: dict[str, Any], limit: int = 3
    ) -> list[dict[str, Any]]:
        """Find similar historical errors.

        Args:
            tool_name: Tool name
            parameters: Parameters used
            limit: Maximum number of errors to return

        Returns:
            List of similar error records
        """
        if not self.db_manager:
            return []

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT tc.tool_name, tc.parameters, tc.error_message, tc.error_type,
                           er.recovery_strategy, er.success, tc.created_at
                    FROM tool_calls tc
                    LEFT JOIN error_recovery er ON tc.id = er.tool_call_id
                    WHERE tc.tool_name = ? AND tc.success = 0
                    ORDER BY tc.created_at DESC
                    LIMIT ?
                    """,
                    (tool_name, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to find similar errors: {e}")
            return []

    def format_error_for_llm(
        self,
        error: Exception,
        classification: ErrorClassification,
        context: ErrorContext,
    ) -> str:
        """Format error for LLM understanding.

        Args:
            error: The exception
            classification: Error classification
            context: Error context

        Returns:
            Formatted error message for LLM
        """
        recovery_suggestions = self.get_recovery_strategy(classification.error_type, context)

        message = f"""Error Information:
- Type: {classification.error_type.value}
- Severity: {classification.severity.value}
- Tool: {context.tool_name}
- Description: {str(error)}

Likely Causes:
{self._get_likely_causes(classification.error_type, error)}

Recovery Suggestions:
{self._format_recovery_suggestions(recovery_suggestions)}

Next Steps:
{self._get_next_steps(classification, context)}

Context:
- Attempt: {context.attempt_number}
- Parameters: {json.dumps(context.parameters, indent=2)}
"""
        return message

    def _get_likely_causes(self, error_type: ErrorType, error: Exception) -> str:
        """Get likely causes for error type.

        Args:
            error_type: Type of error
            error: The exception

        Returns:
            String of likely causes
        """
        causes = {
            ErrorType.VALIDATION: [
                "Invalid parameter format or type",
                "Missing required parameters",
                "Parameter value outside allowed range",
            ],
            ErrorType.EXECUTION: [
                "Tool execution logic error",
                "External service failure",
                "Unexpected runtime condition",
            ],
            ErrorType.NETWORK: [
                "Network connectivity issues",
                "Service unavailable",
                "DNS resolution failure",
            ],
            ErrorType.TIMEOUT: [
                "Request processing exceeded time limit",
                "External service slow response",
                "Resource constraints",
            ],
            ErrorType.PERMISSION: [
                "Insufficient permissions for operation",
                "Authentication required or expired",
                "Access denied to resource",
            ],
            ErrorType.NOT_FOUND: [
                "Requested resource does not exist",
                "Invalid identifier or path",
                "Resource was deleted or moved",
            ],
            ErrorType.RATE_LIMIT: [
                "API rate limit exceeded",
                "Too many concurrent requests",
                "Quota exhausted",
            ],
            ErrorType.INTERNAL: [
                "Server-side error",
                "Internal service failure",
                "Unexpected system error",
            ],
        }

        return "\n".join(f"- {cause}" for cause in causes.get(error_type, ["Unknown cause"]))

    def _format_recovery_suggestions(self, suggestions: list[RecoverySuggestion]) -> str:
        """Format recovery suggestions.

        Args:
            suggestions: List of recovery suggestions

        Returns:
            Formatted suggestions string
        """
        if not suggestions:
            return "- No specific recovery suggestions available"

        formatted = []
        for suggestion in suggestions:
            formatted.append(f"- {suggestion.strategy.value}: {suggestion.description}")
            if suggestion.parameter_modifications:
                formatted.append(
                    f"  Parameter changes: {json.dumps(suggestion.parameter_modifications)}"
                )
            if suggestion.alternative_tools:
                formatted.append(f"  Alternative tools: {', '.join(suggestion.alternative_tools)}")

        return "\n".join(formatted)

    def _get_next_steps(self, classification: ErrorClassification, context: ErrorContext) -> str:
        """Get next steps based on classification and context.

        Args:
            classification: Error classification
            context: Error context

        Returns:
            Next steps string
        """
        steps = []

        if classification.retryability == Retryability.RETRYABLE:
            steps.append("1. Retry the operation with exponential backoff")
        elif classification.retryability == Retryability.CONDITIONAL:
            steps.append("1. Review and modify parameters before retrying")
        else:
            steps.append("1. Do not retry - parameters or approach need modification")

        if classification.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            steps.append("2. Consider escalating to human intervention")
        else:
            steps.append("2. Try alternative tools or approaches")

        steps.append("3. Log error details for analysis and pattern learning")

        return "\n".join(steps)

    def get_recovery_strategy(
        self, error_type: ErrorType, context: ErrorContext
    ) -> list[RecoverySuggestion]:
        """Get recovery strategy suggestions.

        Args:
            error_type: Type of error
            context: Error context

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        fingerprint = self.get_error_fingerprint(Exception(error_type.value), context)
        pattern = self.pattern_db.get_pattern(fingerprint)

        if pattern and pattern.recovery_strategies:
            for strategy, success_rate in pattern.recovery_strategies.items():
                suggestions.append(
                    RecoverySuggestion(
                        strategy=RecoveryStrategy(strategy),
                        description=self._get_strategy_description(RecoveryStrategy(strategy)),
                        estimated_success_rate=success_rate,
                        confidence=min(pattern.total_occurrences / 10.0, 1.0),
                    )
                )

        default_strategies = self._get_default_strategies(error_type)
        for strategy in default_strategies:
            if not any(s.strategy == strategy for s in suggestions):
                suggestions.append(
                    RecoverySuggestion(
                        strategy=strategy,
                        description=self._get_strategy_description(strategy),
                        estimated_success_rate=0.5,
                        confidence=0.3,
                    )
                )

        suggestions.sort(key=lambda x: (x.estimated_success_rate * x.confidence), reverse=True)
        return suggestions[:3]

    def _get_default_strategies(self, error_type: ErrorType) -> list[RecoveryStrategy]:
        """Get default recovery strategies for error type.

        Args:
            error_type: Type of error

        Returns:
            List of default strategies
        """
        strategies = {
            ErrorType.VALIDATION: [
                RecoveryStrategy.MODIFY_PARAMS,
                RecoveryStrategy.ASK_USER,
            ],
            ErrorType.EXECUTION: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.MODIFY_PARAMS,
            ],
            ErrorType.NETWORK: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.TRY_ALTERNATIVE,
            ],
            ErrorType.TIMEOUT: [RecoveryStrategy.RETRY, RecoveryStrategy.MODIFY_PARAMS],
            ErrorType.PERMISSION: [RecoveryStrategy.ASK_USER, RecoveryStrategy.ABORT],
            ErrorType.NOT_FOUND: [
                RecoveryStrategy.MODIFY_PARAMS,
                RecoveryStrategy.TRY_ALTERNATIVE,
            ],
            ErrorType.RATE_LIMIT: [RecoveryStrategy.RETRY, RecoveryStrategy.SKIP],
            ErrorType.INTERNAL: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.TRY_ALTERNATIVE,
            ],
        }
        return strategies.get(error_type, [RecoveryStrategy.ABORT])

    def _get_strategy_description(self, strategy: RecoveryStrategy) -> str:
        """Get description for recovery strategy.

        Args:
            strategy: Recovery strategy

        Returns:
            Strategy description
        """
        descriptions = {
            RecoveryStrategy.RETRY: "Retry the operation with exponential backoff",
            RecoveryStrategy.MODIFY_PARAMS: "Modify parameters and try again",
            RecoveryStrategy.TRY_ALTERNATIVE: "Try using an alternative tool or method",
            RecoveryStrategy.SKIP: "Skip this operation and continue",
            RecoveryStrategy.ASK_USER: "Request user intervention or clarification",
            RecoveryStrategy.ABORT: "Abort the current operation",
        }
        return descriptions.get(strategy, "Unknown strategy")

    def is_retryable_error(self, error: Exception, context: ErrorContext | None = None) -> bool:
        """Check if error can be retried.

        Args:
            error: The exception
            context: Optional error context

        Returns:
            True if retryable
        """
        classification = self.classifier.classify(error, context)
        return classification.retryability in [
            Retryability.RETRYABLE,
            Retryability.CONDITIONAL,
        ]

    def get_retry_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number
            error_type: Type of error

        Returns:
            Delay in seconds
        """
        base_delays = {
            ErrorType.NETWORK: 2.0,
            ErrorType.TIMEOUT: 1.0,
            ErrorType.RATE_LIMIT: 5.0,
            ErrorType.INTERNAL: 1.0,
        }

        base_delay = base_delays.get(error_type, 1.0)
        return min(base_delay * (2 ** (attempt - 1)), 60.0)

    def should_abort(self, error: Exception, context: ErrorContext | None = None) -> bool:
        """Check if execution should abort.

        Args:
            error: The exception
            context: Optional error context

        Returns:
            True if should abort
        """
        classification = self.classifier.classify(error, context)

        if classification.severity == SeverityLevel.CRITICAL:
            return True

        if classification.retryability == Retryability.NON_RETRYABLE:
            if context and context.attempt_number >= 2:
                return True

        if context and context.attempt_number >= 3:
            return True

        return False

    def record_error_recovery(
        self,
        session_id: int,
        tool_call_id: int,
        error: Exception,
        context: ErrorContext,
        recovery_strategy: str,
        success: bool,
    ):
        """Record error recovery attempt in database.

        Args:
            session_id: Database session ID
            tool_call_id: Database tool call ID
            error: The exception
            context: Error context
            recovery_strategy: Strategy used
            success: Whether recovery was successful
        """
        classification = self.classifier.classify(error, context)
        fingerprint = self.get_error_fingerprint(error, context)
        pattern_key = f"{context.tool_name}:{hashlib.md5(json.dumps(context.parameters, sort_keys=True).encode()).hexdigest()[:8]}"

        if self.db_manager:
            try:
                self.db_manager.add_error_recovery(
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    error_type=classification.error_type.value,
                    recovery_strategy=recovery_strategy,
                    attempt_number=context.attempt_number,
                    success=success,
                    original_error=str(error),
                    recovery_details={
                        "fingerprint": fingerprint,
                        "parameters": context.parameters,
                        "severity": classification.severity.value,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to record error recovery: {e}")

        self.pattern_db.update_pattern(
            fingerprint=fingerprint,
            error_type=classification.error_type,
            pattern_key=pattern_key,
            recovery_strategy=recovery_strategy,
            success=success,
        )

    def handle_error(
        self,
        error: Exception,
        tool_name: str,
        parameters: dict[str, Any],
        session_id: int | None = None,
        tool_call_id: int | None = None,
        attempt_number: int = 1,
    ) -> tuple[str, list[RecoverySuggestion]]:
        """Handle error with full classification and recovery suggestions.

        Args:
            error: The exception
            tool_name: Name of the tool
            parameters: Parameters used
            session_id: Optional database session ID
            tool_call_id: Optional database tool call ID
            attempt_number: Current attempt number

        Returns:
            Tuple of (formatted error message, recovery suggestions)
        """
        context = self.build_error_context(tool_name, parameters, attempt_number)
        classification = self.classifier.classify(error, context)

        formatted_message = self.format_error_for_llm(error, classification, context)
        recovery_suggestions = self.get_recovery_strategy(classification.error_type, context)

        logger.warning(
            f"Error handled: {classification.error_type.value} - {classification.severity.value} - {str(error)[:100]}"
        )

        return formatted_message, recovery_suggestions
