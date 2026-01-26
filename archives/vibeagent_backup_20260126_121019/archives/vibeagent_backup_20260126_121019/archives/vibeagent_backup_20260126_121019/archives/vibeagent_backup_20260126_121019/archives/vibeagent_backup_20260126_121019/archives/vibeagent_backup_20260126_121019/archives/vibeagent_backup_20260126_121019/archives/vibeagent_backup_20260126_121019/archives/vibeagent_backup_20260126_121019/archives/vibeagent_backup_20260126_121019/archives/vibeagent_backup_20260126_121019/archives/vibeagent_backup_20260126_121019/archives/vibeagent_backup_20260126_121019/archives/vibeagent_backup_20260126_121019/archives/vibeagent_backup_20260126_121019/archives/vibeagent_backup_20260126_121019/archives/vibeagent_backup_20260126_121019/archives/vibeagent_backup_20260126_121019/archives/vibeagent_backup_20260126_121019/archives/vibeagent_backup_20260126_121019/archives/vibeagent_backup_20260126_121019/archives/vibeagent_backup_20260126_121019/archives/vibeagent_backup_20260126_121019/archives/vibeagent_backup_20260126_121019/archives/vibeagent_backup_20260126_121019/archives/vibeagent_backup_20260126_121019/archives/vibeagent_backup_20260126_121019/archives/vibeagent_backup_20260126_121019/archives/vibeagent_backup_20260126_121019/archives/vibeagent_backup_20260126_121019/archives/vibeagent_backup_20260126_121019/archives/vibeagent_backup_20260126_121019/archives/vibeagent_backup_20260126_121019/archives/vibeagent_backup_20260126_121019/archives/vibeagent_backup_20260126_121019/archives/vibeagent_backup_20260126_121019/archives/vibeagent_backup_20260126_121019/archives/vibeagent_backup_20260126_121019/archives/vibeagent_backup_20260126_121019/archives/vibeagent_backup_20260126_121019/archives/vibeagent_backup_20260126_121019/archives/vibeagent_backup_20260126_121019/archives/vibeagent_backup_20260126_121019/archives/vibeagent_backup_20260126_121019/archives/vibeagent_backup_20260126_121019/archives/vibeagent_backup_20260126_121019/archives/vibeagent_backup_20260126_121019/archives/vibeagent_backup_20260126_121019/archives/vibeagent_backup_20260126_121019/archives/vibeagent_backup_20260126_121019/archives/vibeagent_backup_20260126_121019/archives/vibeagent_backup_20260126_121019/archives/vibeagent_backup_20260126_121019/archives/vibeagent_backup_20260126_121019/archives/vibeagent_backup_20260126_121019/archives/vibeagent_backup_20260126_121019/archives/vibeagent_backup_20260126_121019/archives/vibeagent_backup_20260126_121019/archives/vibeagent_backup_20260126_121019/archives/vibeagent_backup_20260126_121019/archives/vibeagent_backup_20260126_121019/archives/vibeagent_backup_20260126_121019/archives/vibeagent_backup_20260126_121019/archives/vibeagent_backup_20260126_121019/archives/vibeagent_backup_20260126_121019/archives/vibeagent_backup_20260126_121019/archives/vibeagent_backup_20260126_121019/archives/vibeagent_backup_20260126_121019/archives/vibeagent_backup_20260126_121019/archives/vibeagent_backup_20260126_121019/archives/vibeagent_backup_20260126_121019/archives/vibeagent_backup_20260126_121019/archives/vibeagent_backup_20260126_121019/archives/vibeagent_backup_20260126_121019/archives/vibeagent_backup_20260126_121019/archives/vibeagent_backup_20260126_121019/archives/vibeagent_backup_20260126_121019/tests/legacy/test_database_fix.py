"""Quick test to verify database operations work correctly."""

import pytest
import tempfile
import os
from pathlib import Path

from core.database_manager import DatabaseManager


@pytest.fixture
def test_db_path():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def db_manager(test_db_path):
    """Create a database manager for testing."""
    return DatabaseManager(db_path=test_db_path)


def test_database_session_operations(db_manager):
    """Test that session creation and update work with correct parameters."""

    # Create a session
    session_id = db_manager.create_session(
        session_id="test-session-1",
        session_type="test",
        model="gpt-4",
        orchestrator_type="ToolOrchestrator",
        metadata={"test": True},
    )

    assert session_id > 0

    # Update session with correct column names
    db_manager.update_session(
        session_id,
        final_status="completed",
        total_iterations=5,
        total_tool_calls=3,
        total_duration_ms=1500,
    )

    # Verify the update worked
    session = db_manager.get_session(session_id)
    assert session is not None
    assert session["final_status"] == "completed"
    assert session["total_iterations"] == 5
    assert session["total_tool_calls"] == 3
    assert session["total_duration_ms"] == 1500

    print("✓ Database session operations work correctly")


def test_database_message_operations(db_manager):
    """Test that message operations work."""

    session_id = db_manager.create_session(
        session_id="test-session-2",
        session_type="test",
        model="gpt-4",
    )

    # Add a message
    message_id = db_manager.add_message(
        session_id=session_id,
        role="user",
        content="Hello, world!",
        message_index=0,
        model="gpt-4",
    )

    assert message_id > 0

    # Get messages
    messages = db_manager.get_session_messages(session_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello, world!"

    print("✓ Database message operations work correctly")


def test_database_tool_call_operations(db_manager):
    """Test that tool call operations work."""

    session_id = db_manager.create_session(
        session_id="test-session-3",
        session_type="test",
        model="gpt-4",
    )

    # Add a tool call
    tool_call_id = db_manager.add_tool_call(
        session_id=session_id,
        call_index=0,
        tool_name="test_tool",
        parameters={"arg1": "value1"},
        execution_time_ms=100,
        success=True,
    )

    assert tool_call_id > 0

    # Add a tool result
    result_id = db_manager.add_tool_result(
        tool_call_id=tool_call_id,
        success=True,
        data={"result": "success"},
        result_size_bytes=100,
    )

    assert result_id > 0

    print("✓ Database tool call operations work correctly")


def test_database_reasoning_steps(db_manager):
    """Test that reasoning steps work."""

    session_id = db_manager.create_session(
        session_id="test-session-4",
        session_type="test",
        model="gpt-4",
    )

    # Add a reasoning step
    reasoning_id = db_manager.add_reasoning_step(
        session_id=session_id,
        iteration=1,
        step_type="thought",
        content="This is a test reasoning step",
    )

    assert reasoning_id > 0

    print("✓ Database reasoning steps work correctly")


def test_database_performance_metrics(db_manager):
    """Test that performance metrics work."""

    session_id = db_manager.create_session(
        session_id="test-session-5",
        session_type="test",
        model="gpt-4",
    )

    # Add a performance metric
    metric_id = db_manager.add_performance_metric(
        session_id=session_id,
        metric_name="test_metric",
        metric_value=1.5,
        metric_unit="ms",
    )

    assert metric_id > 0

    print("✓ Database performance metrics work correctly")


if __name__ == "__main__":
    import sys

    # Run tests manually
    test_db_path = tempfile.mktemp(suffix=".db")
    try:
        db_manager = DatabaseManager(db_path=test_db_path)

        test_database_session_operations(db_manager)
        test_database_message_operations(db_manager)
        test_database_tool_call_operations(db_manager)
        test_database_reasoning_steps(db_manager)
        test_database_performance_metrics(db_manager)

        print("\n✅ All database operations work correctly!")

    finally:
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
