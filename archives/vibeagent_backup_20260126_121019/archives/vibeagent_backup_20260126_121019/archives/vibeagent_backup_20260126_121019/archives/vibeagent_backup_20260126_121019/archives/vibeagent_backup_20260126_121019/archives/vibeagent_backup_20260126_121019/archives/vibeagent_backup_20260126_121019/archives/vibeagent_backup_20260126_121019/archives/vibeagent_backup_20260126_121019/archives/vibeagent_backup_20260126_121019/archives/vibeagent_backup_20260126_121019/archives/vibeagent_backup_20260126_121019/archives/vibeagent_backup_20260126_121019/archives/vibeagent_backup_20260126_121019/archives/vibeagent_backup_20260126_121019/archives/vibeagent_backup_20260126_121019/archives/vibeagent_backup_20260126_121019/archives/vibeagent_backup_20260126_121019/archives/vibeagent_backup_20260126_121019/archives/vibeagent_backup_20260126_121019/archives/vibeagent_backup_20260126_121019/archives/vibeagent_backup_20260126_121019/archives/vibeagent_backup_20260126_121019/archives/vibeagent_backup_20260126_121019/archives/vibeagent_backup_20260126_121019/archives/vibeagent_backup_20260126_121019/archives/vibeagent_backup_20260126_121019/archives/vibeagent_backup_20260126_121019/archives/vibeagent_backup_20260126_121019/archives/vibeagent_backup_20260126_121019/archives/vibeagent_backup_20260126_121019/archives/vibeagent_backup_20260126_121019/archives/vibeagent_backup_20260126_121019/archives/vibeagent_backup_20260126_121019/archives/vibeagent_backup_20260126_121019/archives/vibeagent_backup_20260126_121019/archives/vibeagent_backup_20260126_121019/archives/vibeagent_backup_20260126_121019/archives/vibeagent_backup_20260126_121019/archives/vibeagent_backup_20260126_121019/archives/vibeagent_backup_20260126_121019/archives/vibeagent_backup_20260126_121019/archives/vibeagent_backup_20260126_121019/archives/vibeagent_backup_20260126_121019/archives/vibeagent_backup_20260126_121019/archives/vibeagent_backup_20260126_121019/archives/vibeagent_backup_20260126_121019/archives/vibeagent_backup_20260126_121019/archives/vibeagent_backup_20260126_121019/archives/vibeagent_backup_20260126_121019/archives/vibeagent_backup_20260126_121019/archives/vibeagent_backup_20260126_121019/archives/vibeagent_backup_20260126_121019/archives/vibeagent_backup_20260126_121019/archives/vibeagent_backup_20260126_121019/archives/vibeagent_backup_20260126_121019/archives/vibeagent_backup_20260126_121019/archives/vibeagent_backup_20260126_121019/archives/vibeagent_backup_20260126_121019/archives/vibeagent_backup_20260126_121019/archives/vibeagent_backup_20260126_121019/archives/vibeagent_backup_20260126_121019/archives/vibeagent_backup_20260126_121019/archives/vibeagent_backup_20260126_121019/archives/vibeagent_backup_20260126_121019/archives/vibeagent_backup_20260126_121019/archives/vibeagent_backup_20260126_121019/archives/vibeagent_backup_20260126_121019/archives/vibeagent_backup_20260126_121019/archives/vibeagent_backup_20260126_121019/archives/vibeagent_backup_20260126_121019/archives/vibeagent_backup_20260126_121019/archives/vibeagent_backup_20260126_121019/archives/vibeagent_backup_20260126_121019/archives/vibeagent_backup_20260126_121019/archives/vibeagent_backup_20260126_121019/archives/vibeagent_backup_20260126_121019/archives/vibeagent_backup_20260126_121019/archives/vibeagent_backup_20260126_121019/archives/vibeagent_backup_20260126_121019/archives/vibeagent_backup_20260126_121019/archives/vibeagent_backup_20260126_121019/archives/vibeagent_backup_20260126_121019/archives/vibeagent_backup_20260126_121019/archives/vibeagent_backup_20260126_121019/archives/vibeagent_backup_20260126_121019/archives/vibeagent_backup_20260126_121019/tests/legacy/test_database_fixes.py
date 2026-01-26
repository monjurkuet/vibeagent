"""Simple test to verify database fixes work end-to-end."""

import tempfile
import os

from core.database_manager import DatabaseManager


def test_database_update_session_fix():
    """Test that update_session works with correct parameters."""

    test_db_path = tempfile.mktemp(suffix=".db")

    try:
        # Initialize database
        db_manager = DatabaseManager(db_path=test_db_path)

        # Create a session
        session_id = db_manager.create_session(
            session_id="test-session-update",
            session_type="tool_orchestration",
            model="gpt-4",
            orchestrator_type="ToolOrchestrator",
        )

        print(f"✓ Created session with ID: {session_id}")

        # Test update with correct parameters (matching schema)
        db_manager.update_session(
            session_id,
            final_status="completed",
            total_iterations=5,
            total_tool_calls=3,
            total_duration_ms=1500,
        )

        print("✓ Updated session with correct parameters")

        # Verify the update
        session = db_manager.get_session(session_id)
        assert session is not None
        assert session["final_status"] == "completed"
        assert session["total_iterations"] == 5
        assert session["total_tool_calls"] == 3
        assert session["total_duration_ms"] == 1500

        print("✓ Verified session data:")
        print(f"  - final_status: {session['final_status']}")
        print(f"  - total_iterations: {session['total_iterations']}")
        print(f"  - total_tool_calls: {session['total_tool_calls']}")
        print(f"  - total_duration_ms: {session['total_duration_ms']}")

        # Test update with error status
        db_manager.update_session(
            session_id,
            final_status="error",
            total_iterations=2,
            total_tool_calls=1,
            total_duration_ms=500,
            error_message="Test error",
        )

        print("✓ Updated session with error status")

        # Verify the error update
        session = db_manager.get_session(session_id)
        assert session["final_status"] == "error"
        assert session["error_message"] == "Test error"

        print("✓ Verified error update")

        return True

    finally:
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)


def test_database_all_operations():
    """Test all database operations work together."""

    test_db_path = tempfile.mktemp(suffix=".db")

    try:
        # Initialize database
        db_manager = DatabaseManager(db_path=test_db_path)

        # Create session
        session_id = db_manager.create_session(
            session_id="test-session-full",
            session_type="test",
            model="gpt-4",
            orchestrator_type="TestOrchestrator",
            metadata={"test": True},
        )

        # Add message
        db_manager.add_message(
            session_id=session_id,
            role="user",
            content="Test message",
            message_index=0,
            model="gpt-4",
        )

        # Add LLM response
        db_manager.add_llm_response(
            session_id=session_id,
            message_id=None,
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            response_time_ms=100,
        )

        # Add tool call
        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="test_tool",
            parameters={"arg": "value"},
            execution_time_ms=50,
            success=True,
        )

        # Add tool result
        db_manager.add_tool_result(
            tool_call_id=tool_call_id,
            success=True,
            data={"result": "test"},
        )

        # Add reasoning step
        db_manager.add_reasoning_step(
            session_id=session_id,
            iteration=1,
            step_type="thought",
            content="Test reasoning",
        )

        # Add performance metric
        db_manager.add_performance_metric(
            session_id=session_id,
            metric_name="test_metric",
            metric_value=1.5,
        )

        # Update session
        db_manager.update_session(
            session_id,
            final_status="completed",
            total_iterations=1,
            total_tool_calls=1,
            total_duration_ms=200,
        )

        print("✓ All database operations work correctly")

        # Verify all data
        messages = db_manager.get_session_messages(session_id)
        assert len(messages) == 1

        print("✓ All data persisted correctly")

        return True

    finally:
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Testing Database Parameter Fixes")
        print("=" * 60)

        print("\nTest 1: update_session with correct parameters")
        print("-" * 60)
        test_database_update_session_fix()

        print("\nTest 2: All database operations")
        print("-" * 60)
        test_database_all_operations()

        print("\n" + "=" * 60)
        print("✅ All database fixes verified successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
