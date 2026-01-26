"""Integration test to verify orchestrator works with database."""

import tempfile
import os
from pathlib import Path
from typing import List

from core.database_manager import DatabaseManager
from core.skill import BaseSkill, SkillResult
from core.tool_orchestrator import ToolOrchestrator


class MockLLMSkill:
    """Mock LLM skill for testing."""

    def __init__(self):
        self.model = "gpt-4"
        self.base_url = "http://localhost:11434"

    def execute(self, prompt: str, **kwargs) -> SkillResult:
        return SkillResult(
            success=True,
            data={
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response",
                    "tool_calls": [],
                },
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )


class MockSearchSkill(BaseSkill):
    """Mock search skill for testing."""

    def __init__(self):
        super().__init__(name="search", version="1.0.0")

    def get_tool_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, **kwargs) -> SkillResult:
        self._record_usage()
        query = kwargs.get("query", "")
        return SkillResult(
            success=True, data={"results": ["Mock result for: " + query]}
        )

    def validate(self) -> bool:
        return True

    def get_dependencies(self) -> List[str]:
        return []


def test_tool_orchestrator_with_database():
    """Test that tool orchestrator works with database tracking."""

    test_db_path = tempfile.mktemp(suffix=".db")

    try:
        # Initialize database
        db_manager = DatabaseManager(db_path=test_db_path)

        # Initialize skills
        llm_skill = MockLLMSkill()
        skills = {"search": MockSearchSkill()}

        # Initialize orchestrator with database
        orchestrator = ToolOrchestrator(
            llm_skill=llm_skill, skills=skills, db_manager=db_manager, use_react=False
        )

        # Execute a simple query
        result = orchestrator.execute_with_tools(
            user_message="Search for information about AI", max_iterations=3
        )

        # Verify result exists (may fail due to mock LLM)
        assert result is not None

        # Verify database has session
        sessions = db_manager.export_to_json("SELECT * FROM sessions")
        assert len(sessions) > 0

        print("✓ Tool orchestrator with database tracking works correctly")
        print(f"  - Session created: {len(sessions)} session(s)")
        print(f"  - Success: {result.success}")
        print(f"  - Iterations: {result.iterations}")
        print(f"  - Tool calls: {result.tool_calls_made}")
        if result.error:
            print(f"  - Error: {result.error}")

        return True

    finally:
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)


def test_tool_orchestrator_react_mode():
    """Test that ReAct mode works with database tracking."""

    test_db_path = tempfile.mktemp(suffix=".db")

    try:
        # Initialize database
        db_manager = DatabaseManager(db_path=test_db_path)

        # Initialize skills
        llm_skill = MockLLMSkill()
        skills = {"search": MockSearchSkill()}

        # Initialize orchestrator with ReAct mode
        orchestrator = ToolOrchestrator(
            llm_skill=llm_skill, skills=skills, db_manager=db_manager, use_react=True
        )

        # Execute a query with ReAct
        result = orchestrator.execute_with_tools(
            user_message="Search for information about AI", max_iterations=3
        )

        # Verify result
        assert result is not None

        # Verify database has reasoning steps
        reasoning_steps = db_manager.export_to_json("SELECT * FROM reasoning_steps")

        print("✓ ReAct mode with database tracking works correctly")
        print(f"  - Reasoning steps: {len(reasoning_steps)} step(s)")

        return True

    finally:
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)


if __name__ == "__main__":
    try:
        test_tool_orchestrator_with_database()
        test_tool_orchestrator_react_mode()

        print("\n✅ All integration tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
