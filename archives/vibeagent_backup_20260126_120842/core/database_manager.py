import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for VibeAgent."""

    def __init__(self, db_path: str = "data/vibeagent.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database schema if not exists."""
        with self.get_connection() as conn:
            schema_file = Path(__file__).parent.parent / "config" / "schema.sql"
            if schema_file.exists():
                with open(schema_file) as f:
                    schema = f.read()
                conn.executescript(schema)
                logger.info("Database schema initialized")

    def create_session(
        self,
        session_id: str,
        session_type: str,
        model: str,
        orchestrator_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Create a new session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions
                (session_id, session_type, model, orchestrator_type, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    session_type,
                    model,
                    orchestrator_type,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def update_session(self, session_id: int, **kwargs):
        """Update session fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = []
            values = []
            for key, value in kwargs.items():
                if key == "metadata":
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value) if value else None)
                else:
                    fields.append(f"{key} = ?")
                    values.append(value)
            values.append(session_id)
            cursor.execute(
                f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", values
            )

    def get_session(self, session_id: int) -> Optional[Dict]:
        """Get session by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        message_index: int,
        raw_content: Optional[str] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_message_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a message to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages
                (session_id, role, content, raw_content, message_index,
                 tokens_input, tokens_output, model, temperature, max_tokens,
                 parent_message_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    raw_content,
                    message_index,
                    tokens_input,
                    tokens_output,
                    model,
                    temperature,
                    max_tokens,
                    parent_message_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def get_session_messages(self, session_id: int) -> List[Dict]:
        """Get all messages for a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY message_index",
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def add_llm_response(
        self,
        session_id: int,
        message_id: Optional[int],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        response_time_ms: int,
        finish_reason: Optional[str] = None,
        raw_response: Optional[Dict] = None,
        reasoning_content: Optional[str] = None,
        tool_calls_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add an LLM response to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO llm_responses
                (session_id, message_id, model, prompt_tokens, completion_tokens,
                 total_tokens, response_time_ms, finish_reason, raw_response,
                 reasoning_content, tool_calls_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message_id,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    response_time_ms,
                    finish_reason,
                    json.dumps(raw_response) if raw_response else None,
                    reasoning_content,
                    tool_calls_count,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def add_tool_call(
        self,
        session_id: int,
        call_index: int,
        tool_name: str,
        parameters: Dict,
        execution_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        retry_count: int = 0,
        is_parallel: bool = False,
        parallel_batch_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a tool call to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_calls
                (session_id, call_index, tool_name, parameters, execution_time_ms,
                 success, error_message, error_type, retry_count, is_parallel,
                 parallel_batch_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    call_index,
                    tool_name,
                    json.dumps(parameters),
                    execution_time_ms,
                    success,
                    error_message,
                    error_type,
                    retry_count,
                    is_parallel,
                    parallel_batch_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def add_tool_result(
        self,
        tool_call_id: int,
        success: bool,
        data: Optional[Dict] = None,
        error: Optional[str] = None,
        result_size_bytes: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a tool result to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_results
                (tool_call_id, success, data, error, result_size_bytes, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    success,
                    json.dumps(data) if data else None,
                    error,
                    result_size_bytes,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def update_tool_call(self, tool_call_id: int, **kwargs):
        """Update tool call fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = []
            values = []
            for key, value in kwargs.items():
                if key == "metadata":
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value) if value else None)
                else:
                    fields.append(f"{key} = ?")
                    values.append(value)
            values.append(tool_call_id)
            cursor.execute(
                f"UPDATE tool_calls SET {', '.join(fields)} WHERE id = ?", values
            )

    def create_test_case(
        self,
        name: str,
        category: str,
        description: Optional[str],
        messages: List[Dict],
        tools: List[Dict],
        expected_tools: Optional[List[Dict]] = None,
        expected_parameters: Optional[Dict] = None,
        expect_no_tools: bool = False,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Create a test case."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_cases
                (name, category, description, messages, tools, expected_tools,
                 expected_parameters, expect_no_tools, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    category,
                    description,
                    json.dumps(messages),
                    json.dumps(tools),
                    json.dumps(expected_tools) if expected_tools else None,
                    json.dumps(expected_parameters) if expected_parameters else None,
                    expect_no_tools,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def create_test_run(
        self,
        test_case_id: int,
        run_number: int,
        session_id: Optional[int] = None,
        status: str = "running",
    ) -> int:
        """Create a test run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_runs
                (test_case_id, session_id, run_number, status, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (test_case_id, session_id, run_number, status, datetime.now()),
            )
            return cursor.lastrowid

    def update_test_run(self, test_run_id: int, **kwargs):
        """Update test run fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = []
            values = []
            for key, value in kwargs.items():
                if key in ["metadata"]:
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value) if value else None)
                else:
                    fields.append(f"{key} = ?")
                    values.append(value)
            values.append(test_run_id)
            cursor.execute(
                f"UPDATE test_runs SET {', '.join(fields)} WHERE id = ?", values
            )

    def add_judge_evaluation(
        self,
        test_run_id: int,
        judge_model: str,
        passed: bool,
        confidence: float,
        reasoning: str,
        evaluation_type: str = "semantic",
        criteria: Optional[Dict] = None,
        details: Optional[Dict] = None,
        evaluation_time_ms: Optional[int] = None,
        tool_call_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a judge evaluation to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO judge_evaluations
                (test_run_id, tool_call_id, judge_model, passed, confidence,
                 reasoning, evaluation_type, criteria, details, evaluation_time_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test_run_id,
                    tool_call_id,
                    judge_model,
                    passed,
                    confidence,
                    reasoning,
                    evaluation_type,
                    json.dumps(criteria) if criteria else None,
                    json.dumps(details) if details else None,
                    evaluation_time_ms,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def add_reasoning_step(
        self,
        session_id: int,
        iteration: int,
        step_type: str,
        content: str,
        tool_call_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a reasoning step (for ReAct/ToT)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO reasoning_steps
                (session_id, iteration, step_type, content, tool_call_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    iteration,
                    step_type,
                    content,
                    tool_call_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def add_error_recovery(
        self,
        session_id: int,
        tool_call_id: int,
        error_type: str,
        recovery_strategy: str,
        attempt_number: int,
        success: bool,
        original_error: str,
        recovery_details: Optional[Dict] = None,
    ) -> int:
        """Add an error recovery attempt."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO error_recovery
                (session_id, tool_call_id, error_type, recovery_strategy,
                 attempt_number, success, original_error, recovery_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tool_call_id,
                    error_type,
                    recovery_strategy,
                    attempt_number,
                    success,
                    original_error,
                    json.dumps(recovery_details) if recovery_details else None,
                ),
            )
            return cursor.lastrowid

    def add_self_correction(
        self,
        session_id: int,
        attempt_number: int,
        trigger: str,
        strategy_type: str,
        strategy_description: str,
        original_error: Optional[str] = None,
        success: bool = False,
        execution_time_ms: int = 0,
        reflection_summary: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> int:
        """Add a self-correction attempt."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO self_corrections
                (session_id, attempt_number, trigger, strategy_type, strategy_description,
                 original_error, success, execution_time_ms, reflection_summary, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    attempt_number,
                    trigger,
                    strategy_type,
                    strategy_description,
                    original_error,
                    success,
                    execution_time_ms,
                    reflection_summary,
                    metadata,
                ),
            )
            return cursor.lastrowid

    def add_performance_metric(
        self,
        session_id: int,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a performance metric."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics
                (session_id, metric_name, metric_value, metric_unit, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    metric_name,
                    metric_value,
                    metric_unit,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def get_test_performance(self, days: int = 30) -> List[Dict]:
        """Get test performance for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM test_performance
                WHERE last_run >= datetime('now', '-' || ? || ' days')
                ORDER BY last_run DESC
                """,
                (days,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_tool_success_rate(self) -> List[Dict]:
        """Get tool success rate statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tool_success_rate")
            return [dict(row) for row in cursor.fetchall()]

    def get_model_comparison(self) -> List[Dict]:
        """Get model comparison statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_comparison")
            return [dict(row) for row in cursor.fetchall()]

    def export_to_json(
        self,
        query: str,
        params: Optional[tuple] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """Export query results to JSON."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            rows = [dict(row) for row in cursor.fetchall()]

        if file_path:
            with open(file_path, "w") as f:
                json.dump(rows, f, indent=2)
            return file_path
        else:
            return json.dumps(rows, indent=2)
