import pytest
import sqlite3
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from core.database_manager import DatabaseManager


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_vibeagent.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def db_manager(temp_db_path):
    """Create a DatabaseManager instance with temporary database."""
    return DatabaseManager(db_path=temp_db_path)


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "session_id": "test-session-123",
        "session_type": "test",
        "model": "gpt-4",
        "orchestrator_type": "basic",
        "metadata": {"test_key": "test_value"},
    }


@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "role": "user",
        "content": "Hello, world!",
        "message_index": 0,
        "tokens_input": 10,
        "tokens_output": 5,
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_tool_call_data():
    """Sample tool call data for testing."""
    return {
        "call_index": 0,
        "tool_name": "search",
        "parameters": {"query": "test"},
        "execution_time_ms": 100,
        "success": True,
    }


@pytest.fixture
def sample_test_case_data():
    """Sample test case data for testing."""
    return {
        "name": "Test Case 1",
        "category": "functional",
        "description": "A test case",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"name": "search", "description": "search tool"}],
    }


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_database_file_creation(self, temp_db_path):
        """Test that database file is created."""
        db_path = Path(temp_db_path)
        assert not db_path.exists()

        DatabaseManager(db_path=temp_db_path)

        assert db_path.exists()
        assert db_path.stat().st_size > 0

    def test_schema_initialization(self, db_manager):
        """Test that database schema is properly initialized."""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            tables = [
                "sessions",
                "messages",
                "llm_responses",
                "tool_calls",
                "tool_results",
                "test_cases",
                "test_runs",
                "judge_evaluations",
                "reasoning_steps",
                "error_recovery",
                "performance_metrics",
            ]

            for table in tables:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                result = cursor.fetchone()
                assert result is not None, f"Table {table} not found"

    def test_views_initialization(self, db_manager):
        """Test that database views are properly initialized."""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            views = ["test_performance", "tool_success_rate", "model_comparison"]

            for view in views:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='view' AND name=?",
                    (view,),
                )
                result = cursor.fetchone()
                assert result is not None, f"View {view} not found"

    def test_indexes_initialization(self, db_manager):
        """Test that database indexes are properly initialized."""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            indexes = cursor.fetchall()
            assert len(indexes) > 0, "No indexes found"


class TestSessionOperations:
    """Tests for session CRUD operations."""

    def test_create_session_basic(self, db_manager, sample_session_data):
        """Test creating a basic session."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        assert isinstance(session_id, int)
        assert session_id > 0

    def test_create_session_with_all_fields(self, db_manager, sample_session_data):
        """Test creating a session with all fields."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
            orchestrator_type=sample_session_data["orchestrator_type"],
            metadata=sample_session_data["metadata"],
        )

        session = db_manager.get_session(session_id)
        assert session["session_id"] == sample_session_data["session_id"]
        assert session["session_type"] == sample_session_data["session_type"]
        assert session["model"] == sample_session_data["model"]
        assert session["orchestrator_type"] == sample_session_data["orchestrator_type"]
        assert json.loads(session["metadata"]) == sample_session_data["metadata"]

    def test_create_session_duplicate_id(self, db_manager, sample_session_data):
        """Test that duplicate session IDs raise an error."""
        db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        with pytest.raises(sqlite3.IntegrityError):
            db_manager.create_session(
                session_id=sample_session_data["session_id"],
                session_type="test2",
                model="gpt-3.5",
            )

    def test_get_session_existing(self, db_manager, sample_session_data):
        """Test getting an existing session."""
        created_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        session = db_manager.get_session(created_id)

        assert session is not None
        assert session["id"] == created_id
        assert session["session_id"] == sample_session_data["session_id"]

    def test_get_session_nonexistent(self, db_manager):
        """Test getting a non-existent session."""
        session = db_manager.get_session(99999)
        assert session is None

    def test_update_session_single_field(self, db_manager, sample_session_data):
        """Test updating a single session field."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db_manager.update_session(session_id, total_iterations=5)

        session = db_manager.get_session(session_id)
        assert session["total_iterations"] == 5

    def test_update_session_multiple_fields(self, db_manager, sample_session_data):
        """Test updating multiple session fields."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db_manager.update_session(
            session_id, total_iterations=10, total_tool_calls=3, final_status="success"
        )

        session = db_manager.get_session(session_id)
        assert session["total_iterations"] == 10
        assert session["total_tool_calls"] == 3
        assert session["final_status"] == "success"

    def test_update_session_metadata(self, db_manager, sample_session_data):
        """Test updating session metadata."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        new_metadata = {"updated_key": "updated_value", "count": 42}
        db_manager.update_session(session_id, metadata=new_metadata)

        session = db_manager.get_session(session_id)
        assert json.loads(session["metadata"]) == new_metadata


class TestMessageOperations:
    """Tests for message operations."""

    def test_add_message_basic(
        self, db_manager, sample_session_data, sample_message_data
    ):
        """Test adding a basic message."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id,
            role=sample_message_data["role"],
            content=sample_message_data["content"],
            message_index=sample_message_data["message_index"],
        )

        assert isinstance(message_id, int)
        assert message_id > 0

    def test_add_message_with_all_fields(
        self, db_manager, sample_session_data, sample_message_data
    ):
        """Test adding a message with all fields."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id,
            role=sample_message_data["role"],
            content=sample_message_data["content"],
            message_index=sample_message_data["message_index"],
            raw_content="raw content",
            tokens_input=sample_message_data["tokens_input"],
            tokens_output=sample_message_data["tokens_output"],
            model=sample_message_data["model"],
            temperature=sample_message_data["temperature"],
            max_tokens=sample_message_data["max_tokens"],
            metadata={"custom": "data"},
        )

        messages = db_manager.get_session_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["id"] == message_id
        assert messages[0]["role"] == sample_message_data["role"]
        assert messages[0]["tokens_input"] == sample_message_data["tokens_input"]

    def test_get_session_messages_ordered(self, db_manager, sample_session_data):
        """Test that messages are returned in correct order."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db_manager.add_message(session_id, "user", "first", 2)
        db_manager.add_message(session_id, "assistant", "second", 0)
        db_manager.add_message(session_id, "user", "third", 1)

        messages = db_manager.get_session_messages(session_id)

        assert len(messages) == 3
        assert messages[0]["message_index"] == 0
        assert messages[1]["message_index"] == 1
        assert messages[2]["message_index"] == 2

    def test_get_session_messages_empty(self, db_manager, sample_session_data):
        """Test getting messages for a session with no messages."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        messages = db_manager.get_session_messages(session_id)
        assert messages == []

    def test_add_message_with_parent(
        self, db_manager, sample_session_data, sample_message_data
    ):
        """Test adding a message with a parent message."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        parent_id = db_manager.add_message(
            session_id=session_id, role="user", content="parent", message_index=0
        )

        child_id = db_manager.add_message(
            session_id=session_id,
            role="assistant",
            content="child",
            message_index=1,
            parent_message_id=parent_id,
        )

        messages = db_manager.get_session_messages(session_id)
        assert messages[1]["parent_message_id"] == parent_id


class TestLLMResponseOperations:
    """Tests for LLM response operations."""

    def test_add_llm_response_basic(self, db_manager, sample_session_data):
        """Test adding a basic LLM response."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        response_id = db_manager.add_llm_response(
            session_id=session_id,
            message_id=None,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=2000,
        )

        assert isinstance(response_id, int)
        assert response_id > 0

    def test_add_llm_response_with_all_fields(self, db_manager, sample_session_data):
        """Test adding an LLM response with all fields."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        response_id = db_manager.add_llm_response(
            session_id=session_id,
            message_id=None,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=2000,
            finish_reason="stop",
            raw_response={"content": "response"},
            reasoning_content="thinking process",
            tool_calls_count=2,
            metadata={"custom": "data"},
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM llm_responses WHERE id = ?", (response_id,))
            response = dict(cursor.fetchone())

            assert response["finish_reason"] == "stop"
            assert response["reasoning_content"] == "thinking process"
            assert response["tool_calls_count"] == 2

    def test_add_llm_response_with_message_link(self, db_manager, sample_session_data):
        """Test adding an LLM response linked to a message."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id, role="user", content="test", message_index=0
        )

        response_id = db_manager.add_llm_response(
            session_id=session_id,
            message_id=message_id,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=2000,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM llm_responses WHERE id = ?", (response_id,))
            response = dict(cursor.fetchone())

            assert response["message_id"] == message_id


class TestToolCallOperations:
    """Tests for tool call operations."""

    def test_add_tool_call_basic(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test adding a basic tool call."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=sample_tool_call_data["call_index"],
            tool_name=sample_tool_call_data["tool_name"],
            parameters=sample_tool_call_data["parameters"],
            execution_time_ms=sample_tool_call_data["execution_time_ms"],
            success=sample_tool_call_data["success"],
        )

        assert isinstance(tool_call_id, int)
        assert tool_call_id > 0

    def test_add_tool_call_with_all_fields(self, db_manager, sample_session_data):
        """Test adding a tool call with all fields."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="search",
            parameters={"query": "test", "limit": 10},
            execution_time_ms=150,
            success=False,
            error_message="API error",
            error_type="RateLimitError",
            retry_count=2,
            is_parallel=True,
            parallel_batch_id=1,
            metadata={"attempt": "retry"},
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,))
            tool_call = dict(cursor.fetchone())

            assert tool_call["success"] == False
            assert tool_call["error_message"] == "API error"
            assert tool_call["error_type"] == "RateLimitError"
            assert tool_call["retry_count"] == 2
            assert tool_call["is_parallel"] == 1

    def test_add_tool_result_basic(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test adding a basic tool result."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=sample_tool_call_data["call_index"],
            tool_name=sample_tool_call_data["tool_name"],
            parameters=sample_tool_call_data["parameters"],
            execution_time_ms=sample_tool_call_data["execution_time_ms"],
            success=sample_tool_call_data["success"],
        )

        result_id = db_manager.add_tool_result(
            tool_call_id=tool_call_id, success=True, data={"result": "success"}
        )

        assert isinstance(result_id, int)
        assert result_id > 0

    def test_add_tool_result_with_error(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test adding a tool result with an error."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=sample_tool_call_data["call_index"],
            tool_name=sample_tool_call_data["tool_name"],
            parameters=sample_tool_call_data["parameters"],
            execution_time_ms=sample_tool_call_data["execution_time_ms"],
            success=False,
        )

        result_id = db_manager.add_tool_result(
            tool_call_id=tool_call_id,
            success=False,
            error="Tool failed",
            result_size_bytes=0,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tool_results WHERE id = ?", (result_id,))
            result = dict(cursor.fetchone())

            assert result["success"] == False
            assert result["error"] == "Tool failed"
            assert result["result_size_bytes"] == 0


class TestTestCaseOperations:
    """Tests for test case operations."""

    def test_create_test_case_basic(self, db_manager, sample_test_case_data):
        """Test creating a basic test case."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        assert isinstance(test_case_id, int)
        assert test_case_id > 0

    def test_create_test_case_with_all_fields(self, db_manager):
        """Test creating a test case with all fields."""
        test_case_id = db_manager.create_test_case(
            name="Complex Test",
            category="integration",
            description="Complex test case",
            messages=[{"role": "user", "content": "test"}],
            tools=[{"name": "tool1"}, {"name": "tool2"}],
            expected_tools=[{"name": "tool1"}],
            expected_parameters={"param1": "value1"},
            expect_no_tools=False,
            metadata={"priority": "high"},
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test_cases WHERE id = ?", (test_case_id,))
            test_case = dict(cursor.fetchone())

            assert test_case["name"] == "Complex Test"
            assert test_case["expect_no_tools"] == 0

    def test_create_test_run_basic(self, db_manager, sample_test_case_data):
        """Test creating a basic test run."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1
        )

        assert isinstance(test_run_id, int)
        assert test_run_id > 0

    def test_create_test_run_with_session(
        self, db_manager, sample_session_data, sample_test_case_data
    ):
        """Test creating a test run linked to a session."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1, session_id=session_id
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test_runs WHERE id = ?", (test_run_id,))
            test_run = dict(cursor.fetchone())

            assert test_run["session_id"] == session_id

    def test_update_test_run(self, db_manager, sample_test_case_data):
        """Test updating a test run."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1
        )

        db_manager.update_test_run(
            test_run_id,
            status="completed",
            final_status="success",
            total_iterations=5,
            total_tool_calls=2,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test_runs WHERE id = ?", (test_run_id,))
            test_run = dict(cursor.fetchone())

            assert test_run["status"] == "completed"
            assert test_run["final_status"] == "success"
            assert test_run["total_iterations"] == 5


class TestJudgeEvaluationOperations:
    """Tests for judge evaluation operations."""

    def test_add_judge_evaluation_basic(self, db_manager, sample_test_case_data):
        """Test adding a basic judge evaluation."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1
        )

        evaluation_id = db_manager.add_judge_evaluation(
            test_run_id=test_run_id,
            judge_model="gpt-4",
            passed=True,
            confidence=0.95,
            reasoning="Correct tool usage",
        )

        assert isinstance(evaluation_id, int)
        assert evaluation_id > 0

    def test_add_judge_evaluation_with_all_fields(
        self, db_manager, sample_test_case_data
    ):
        """Test adding a judge evaluation with all fields."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1
        )

        evaluation_id = db_manager.add_judge_evaluation(
            test_run_id=test_run_id,
            tool_call_id=None,
            judge_model="gpt-4",
            passed=True,
            confidence=0.95,
            reasoning="Correct tool usage",
            evaluation_type="semantic",
            criteria={"accuracy": 0.9},
            details={"score": 0.95},
            evaluation_time_ms=500,
            metadata={"version": "1.0"},
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM judge_evaluations WHERE id = ?", (evaluation_id,)
            )
            evaluation = dict(cursor.fetchone())

            assert evaluation["evaluation_type"] == "semantic"
            assert evaluation["evaluation_time_ms"] == 500


class TestReasoningStepOperations:
    """Tests for reasoning step operations."""

    def test_add_reasoning_step_basic(self, db_manager, sample_session_data):
        """Test adding a basic reasoning step."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        step_id = db_manager.add_reasoning_step(
            session_id=session_id,
            iteration=1,
            step_type="thought",
            content="I need to search for information",
        )

        assert isinstance(step_id, int)
        assert step_id > 0

    def test_add_reasoning_step_with_tool_call(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test adding a reasoning step linked to a tool call."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=sample_tool_call_data["call_index"],
            tool_name=sample_tool_call_data["tool_name"],
            parameters=sample_tool_call_data["parameters"],
            execution_time_ms=sample_tool_call_data["execution_time_ms"],
            success=sample_tool_call_data["success"],
        )

        step_id = db_manager.add_reasoning_step(
            session_id=session_id,
            iteration=1,
            step_type="action",
            content="Executing search tool",
            tool_call_id=tool_call_id,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reasoning_steps WHERE id = ?", (step_id,))
            step = dict(cursor.fetchone())

            assert step["tool_call_id"] == tool_call_id


class TestErrorRecoveryOperations:
    """Tests for error recovery operations."""

    def test_add_error_recovery_basic(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test adding a basic error recovery."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=sample_tool_call_data["call_index"],
            tool_name=sample_tool_call_data["tool_name"],
            parameters=sample_tool_call_data["parameters"],
            execution_time_ms=sample_tool_call_data["execution_time_ms"],
            success=False,
        )

        recovery_id = db_manager.add_error_recovery(
            session_id=session_id,
            tool_call_id=tool_call_id,
            error_type="RateLimitError",
            recovery_strategy="retry_with_backoff",
            attempt_number=1,
            success=True,
            original_error="Rate limit exceeded",
        )

        assert isinstance(recovery_id, int)
        assert recovery_id > 0

    def test_add_error_recovery_with_details(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test adding an error recovery with details."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=sample_tool_call_data["call_index"],
            tool_name=sample_tool_call_data["tool_name"],
            parameters=sample_tool_call_data["parameters"],
            execution_time_ms=sample_tool_call_data["execution_time_ms"],
            success=False,
        )

        recovery_id = db_manager.add_error_recovery(
            session_id=session_id,
            tool_call_id=tool_call_id,
            error_type="RateLimitError",
            recovery_strategy="retry_with_backoff",
            attempt_number=2,
            success=False,
            original_error="Rate limit exceeded",
            recovery_details={"backoff_ms": 5000, "max_retries": 3},
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM error_recovery WHERE id = ?", (recovery_id,))
            recovery = dict(cursor.fetchone())

            assert recovery["attempt_number"] == 2
            assert recovery["success"] == False


class TestPerformanceMetricsOperations:
    """Tests for performance metrics operations."""

    def test_add_performance_metric_basic(self, db_manager, sample_session_data):
        """Test adding a basic performance metric."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        metric_id = db_manager.add_performance_metric(
            session_id=session_id, metric_name="total_time", metric_value=1234.56
        )

        assert isinstance(metric_id, int)
        assert metric_id > 0

    def test_add_performance_metric_with_unit(self, db_manager, sample_session_data):
        """Test adding a performance metric with unit."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        metric_id = db_manager.add_performance_metric(
            session_id=session_id,
            metric_name="memory_usage",
            metric_value=512.0,
            metric_unit="MB",
            metadata={"peak": True},
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM performance_metrics WHERE id = ?", (metric_id,)
            )
            metric = dict(cursor.fetchone())

            assert metric["metric_name"] == "memory_usage"
            assert metric["metric_unit"] == "MB"


class TestQueryMethods:
    """Tests for query methods."""

    def test_get_test_performance_empty(self, db_manager):
        """Test getting test performance when no data exists."""
        performance = db_manager.get_test_performance(days=30)
        assert performance == []

    def test_get_test_performance_with_data(self, db_manager, sample_test_case_data):
        """Test getting test performance with data."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1
        )

        db_manager.update_test_run(
            test_run_id,
            status="completed",
            final_status="success",
            completed_at=datetime.now(),
        )

        performance = db_manager.get_test_performance(days=30)
        assert len(performance) > 0
        assert performance[0]["test_case_name"] == sample_test_case_data["name"]

    def test_get_tool_success_rate_empty(self, db_manager):
        """Test getting tool success rate when no data exists."""
        success_rate = db_manager.get_tool_success_rate()
        assert success_rate == []

    def test_get_tool_success_rate_with_data(
        self, db_manager, sample_session_data, sample_tool_call_data
    ):
        """Test getting tool success rate with data."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="search",
            parameters={"query": "test1"},
            execution_time_ms=100,
            success=True,
        )

        db_manager.add_tool_call(
            session_id=session_id,
            call_index=1,
            tool_name="search",
            parameters={"query": "test2"},
            execution_time_ms=150,
            success=False,
        )

        success_rate = db_manager.get_tool_success_rate()
        assert len(success_rate) > 0
        assert success_rate[0]["tool_name"] == "search"
        assert success_rate[0]["total_calls"] == 2
        assert success_rate[0]["successful_calls"] == 1

    def test_get_model_comparison_empty(self, db_manager):
        """Test getting model comparison when no data exists."""
        comparison = db_manager.get_model_comparison()
        assert comparison == []

    def test_get_model_comparison_with_data(self, db_manager, sample_session_data):
        """Test getting model comparison with data."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db_manager.add_llm_response(
            session_id=session_id,
            message_id=None,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_ms=2000,
        )

        comparison = db_manager.get_model_comparison()
        assert len(comparison) > 0
        assert comparison[0]["model"] == "gpt-4"
        assert comparison[0]["total_tokens"] == 150


class TestExportFunctionality:
    """Tests for export functionality."""

    def test_export_to_json_string(self, db_manager, sample_session_data):
        """Test exporting to JSON string."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        json_str = db_manager.export_to_json("SELECT * FROM sessions")

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["id"] == session_id

    def test_export_to_json_file(self, db_manager, sample_session_data, temp_db_path):
        """Test exporting to JSON file."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        file_path = Path(temp_db_path).parent / "export.json"
        result_path = db_manager.export_to_json(
            "SELECT * FROM sessions", file_path=str(file_path)
        )

        assert result_path == str(file_path)
        assert file_path.exists()

        with open(file_path) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["id"] == session_id

    def test_export_to_json_with_params(self, db_manager, sample_session_data):
        """Test exporting to JSON with query parameters."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        json_str = db_manager.export_to_json(
            "SELECT * FROM sessions WHERE id = ?", params=(session_id,)
        )

        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["id"] == session_id


class TestEdgeCases:
    """Tests for edge cases."""

    def test_foreign_key_constraint_message_invalid_session(self, db_manager):
        """Test foreign key constraint for invalid session in message."""
        with pytest.raises(sqlite3.IntegrityError):
            db_manager.add_message(
                session_id=99999, role="user", content="test", message_index=0
            )

    def test_foreign_key_constraint_tool_result_invalid_tool_call(self, db_manager):
        """Test foreign key constraint for invalid tool call in tool result."""
        with pytest.raises(sqlite3.IntegrityError):
            db_manager.add_tool_result(
                tool_call_id=99999, success=True, data={"result": "test"}
            )

    def test_json_serialization_complex_metadata(self, db_manager, sample_session_data):
        """Test JSON serialization of complex metadata."""
        complex_metadata = {
            "nested": {"deeply": {"nested": "value"}},
            "array": [1, 2, 3, {"key": "value"}],
            "mixed": {"number": 42, "string": "test", "bool": True, "null": None},
        }

        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
            metadata=complex_metadata,
        )

        session = db_manager.get_session(session_id)
        retrieved_metadata = json.loads(session["metadata"])
        assert retrieved_metadata == complex_metadata

    def test_null_handling_optional_fields(self, db_manager, sample_session_data):
        """Test NULL handling for optional fields."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
            orchestrator_type=None,
            metadata=None,
        )

        session = db_manager.get_session(session_id)
        assert session["orchestrator_type"] is None
        assert session["metadata"] is None

    def test_duplicate_message_indices(self, db_manager, sample_session_data):
        """Test that duplicate message indices are allowed (no unique constraint)."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db_manager.add_message(session_id, "user", "first", 0)
        db_manager.add_message(session_id, "assistant", "second", 0)

        messages = db_manager.get_session_messages(session_id)
        assert len(messages) == 2

    def test_empty_dict_parameters(self, db_manager, sample_session_data):
        """Test handling of empty dictionary parameters."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="test",
            parameters={},
            execution_time_ms=100,
            success=True,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT parameters FROM tool_calls WHERE id = ?", (tool_call_id,)
            )
            params = json.loads(cursor.fetchone()["parameters"])
            assert params == {}

    def test_large_text_content(self, db_manager, sample_session_data):
        """Test handling of large text content."""
        large_content = "x" * 10000

        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id, role="user", content=large_content, message_index=0
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM messages WHERE id = ?", (message_id,))
            content = cursor.fetchone()["content"]
            assert content == large_content

    def test_special_characters_in_content(self, db_manager, sample_session_data):
        """Test handling of special characters in content."""
        special_content = (
            "Test with 'quotes', \"double quotes\", \n newlines, \t tabs, and emoji 🎉"
        )

        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id, role="user", content=special_content, message_index=0
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM messages WHERE id = ?", (message_id,))
            content = cursor.fetchone()["content"]
            assert content == special_content

    def test_zero_values(self, db_manager, sample_session_data):
        """Test handling of zero values."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        response_id = db_manager.add_llm_response(
            session_id=session_id,
            message_id=None,
            model="gpt-4",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_time_ms=0,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM llm_responses WHERE id = ?", (response_id,))
            response = dict(cursor.fetchone())
            assert response["prompt_tokens"] == 0
            assert response["response_time_ms"] == 0

    def test_negative_values_allowed(self, db_manager, sample_session_data):
        """Test that negative values are stored (no validation in DB layer)."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        metric_id = db_manager.add_performance_metric(
            session_id=session_id, metric_name="temperature_change", metric_value=-5.5
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metric_value FROM performance_metrics WHERE id = ?",
                (metric_id,),
            )
            value = cursor.fetchone()["metric_value"]
            assert value == -5.5


class TestTransactionRollback:
    """Tests for transaction rollback on errors."""

    def test_rollback_on_sql_error(self, db_manager, sample_session_data):
        """Test that transaction rolls back on SQL error."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        initial_session = db_manager.get_session(session_id)

        with pytest.raises(sqlite3.IntegrityError):
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (session_id, session_type, model) VALUES (?, ?, ?)",
                    (sample_session_data["session_id"], "test", "gpt-4"),
                )
                cursor.execute("INVALID SQL STATEMENT")

        session_after = db_manager.get_session(session_id)
        assert session_after == initial_session

    def test_context_manager_commit_on_success(self, db_manager, sample_session_data):
        """Test that context manager commits on success."""
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, session_type, model) VALUES (?, ?, ?)",
                ("test-context-commit", "test", "gpt-4"),
            )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE session_id = ?", ("test-context-commit",)
            )
            result = cursor.fetchone()
            assert result is not None

    def test_context_manager_rollback_on_exception(self, db_manager):
        """Test that context manager rolls back on exception."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (session_id, session_type, model) VALUES (?, ?, ?)",
                    ("test-context-rollback", "test", "gpt-4"),
                )
                raise ValueError("Test exception")
        except ValueError:
            pass

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                ("test-context-rollback",),
            )
            result = cursor.fetchone()
            assert result is None


class TestParameterizedTests:
    """Parameterized tests for various scenarios."""

    @pytest.mark.parametrize(
        "session_type,model",
        [
            ("test", "gpt-4"),
            ("production", "gpt-3.5-turbo"),
            ("benchmark", "claude-3"),
            ("debug", "local-model"),
        ],
    )
    def test_create_session_various_types(self, db_manager, session_type, model):
        """Test creating sessions with various types and models."""
        session_id = db_manager.create_session(
            session_id=f"test-{session_type}", session_type=session_type, model=model
        )

        session = db_manager.get_session(session_id)
        assert session["session_type"] == session_type
        assert session["model"] == model

    @pytest.mark.parametrize(
        "role,expected_valid",
        [
            ("user", True),
            ("assistant", True),
            ("system", True),
            ("function", True),
            ("custom", True),
        ],
    )
    def test_add_message_various_roles(
        self, db_manager, sample_session_data, role, expected_valid
    ):
        """Test adding messages with various roles."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id, role=role, content="test content", message_index=0
        )

        assert message_id > 0

    @pytest.mark.parametrize(
        "success,error_message,expected_success",
        [
            (True, None, True),
            (False, "API Error", False),
            (False, None, False),
            (True, "", True),
        ],
    )
    def test_add_tool_call_various_outcomes(
        self, db_manager, sample_session_data, success, error_message, expected_success
    ):
        """Test adding tool calls with various outcomes."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="test",
            parameters={},
            execution_time_ms=100,
            success=success,
            error_message=error_message,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT success, error_message FROM tool_calls WHERE id = ?",
                (tool_call_id,),
            )
            result = cursor.fetchone()
            assert result["success"] == expected_success

    @pytest.mark.parametrize(
        "passed,confidence",
        [(True, 1.0), (True, 0.5), (False, 0.0), (True, 0.99), (False, 0.01)],
    )
    def test_add_judge_evaluation_various_scores(
        self, db_manager, sample_test_case_data, passed, confidence
    ):
        """Test adding judge evaluations with various scores."""
        test_case_id = db_manager.create_test_case(
            name=sample_test_case_data["name"],
            category=sample_test_case_data["category"],
            description=sample_test_case_data["description"],
            messages=sample_test_case_data["messages"],
            tools=sample_test_case_data["tools"],
        )

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1
        )

        evaluation_id = db_manager.add_judge_evaluation(
            test_run_id=test_run_id,
            judge_model="gpt-4",
            passed=passed,
            confidence=confidence,
            reasoning="Test reasoning",
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT passed, confidence FROM judge_evaluations WHERE id = ?",
                (evaluation_id,),
            )
            result = cursor.fetchone()
            assert result["passed"] == passed
            assert result["confidence"] == confidence


class TestPerformanceAndConcurrency:
    """Tests for performance and basic concurrency scenarios."""

    def test_bulk_insert_performance(self, db_manager, sample_session_data):
        """Test bulk insert performance with multiple messages."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        import time

        start_time = time.time()

        for i in range(100):
            db_manager.add_message(
                session_id=session_id,
                role="user",
                content=f"Message {i}",
                message_index=i,
            )

        elapsed = time.time() - start_time
        assert elapsed < 5.0, f"Bulk insert took {elapsed} seconds, expected < 5.0"

        messages = db_manager.get_session_messages(session_id)
        assert len(messages) == 100

    def test_multiple_connections_same_db(self, temp_db_path):
        """Test multiple connections to the same database."""
        db1 = DatabaseManager(db_path=temp_db_path)
        db2 = DatabaseManager(db_path=temp_db_path)

        session_id_1 = db1.create_session("session1", "test", "gpt-4")
        session_id_2 = db2.create_session("session2", "test", "gpt-4")

        assert db1.get_session(session_id_1) is not None
        assert db2.get_session(session_id_2) is not None

    def test_large_dataset_query_performance(self, db_manager, sample_session_data):
        """Test query performance with large dataset."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        for i in range(500):
            db_manager.add_tool_call(
                session_id=session_id,
                call_index=i,
                tool_name="tool" + str(i % 10),
                parameters={"index": i},
                execution_time_ms=100,
                success=i % 2 == 0,
            )

        import time

        start_time = time.time()
        success_rate = db_manager.get_tool_success_rate()
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Query took {elapsed} seconds, expected < 1.0"
        assert len(success_rate) == 10


class TestDataIntegrity:
    """Tests for data integrity."""

    def test_cascade_delete_session(self, db_manager, sample_session_data):
        """Test that deleting a session cascades to related records."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        message_id = db_manager.add_message(
            session_id=session_id, role="user", content="test", message_index=0
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="test",
            parameters={},
            execution_time_ms=100,
            success=True,
        )

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
            assert cursor.fetchone() is None

            cursor.execute("SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,))
            assert cursor.fetchone() is None

    def test_data_persistence_across_connections(
        self, db_manager, sample_session_data, temp_db_path
    ):
        """Test that data persists across database connections."""
        session_id = db_manager.create_session(
            session_id=sample_session_data["session_id"],
            session_type=sample_session_data["session_type"],
            model=sample_session_data["model"],
        )

        db2 = DatabaseManager(db_path=temp_db_path)
        session = db2.get_session(session_id)

        assert session is not None
        assert session["session_id"] == sample_session_data["session_id"]
