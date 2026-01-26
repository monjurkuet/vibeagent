#!/usr/bin/env python3
"""
Test tool calling with real LLM to verify multi-step and parallel tool calling works.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def test_simple_tool_call():
    """Test simple tool call."""
    logger.info("=" * 80)
    logger.info("TEST 1: Simple Tool Call")
    logger.info("=" * 80)

    from core.database_manager import DatabaseManager
    from core.tool_orchestrator import ToolOrchestrator
    from skills import LLMSkill, ArxivSkill
    from core.skill import BaseSkill, SkillResult
    from typing import List, Dict, Any

    # Setup database
    db_path = Path("data/test_tool_calling.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_manager = DatabaseManager(db_path=str(db_path))
    logger.info(f"✅ Database initialized at {db_path}")

    # Create mock LLM skill that returns tool calls
    class MockLLMSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="mock_llm", version="1.0.0")
            self.model = "mock-model"
            self.base_url = "http://mock"
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1
            prompt = kwargs.get("prompt", "")
            messages = kwargs.get("messages", [])

            # First call - return tool call
            if self.call_count == 1:
                logger.info(f"   LLM Call #{self.call_count}: Returning tool call")
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "I'll search for that information.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "test"}',
                                    },
                                }
                            ],
                        },
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                )
            # Second call - return final answer
            else:
                logger.info(f"   LLM Call #{self.call_count}: Returning final answer")
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "Here's the result: test result",
                        },
                        "usage": {
                            "prompt_tokens": 30,
                            "completion_tokens": 10,
                            "total_tokens": 40,
                        },
                    },
                )

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

    # Create mock search skill
    class MockSearchSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="search", version="1.0.0")
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1
            query = kwargs.get("query", "")
            logger.info(f"   Search Skill executed: query='{query}'")
            return SkillResult(
                success=True,
                data={"results": [f"Result for: {query}"]},
            )

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

        def get_tool_schema(self) -> Dict:
            return {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            }

    # Create orchestrator
    llm_skill = MockLLMSkill()
    search_skill = MockSearchSkill()
    skills = {"search": search_skill}

    orchestrator = ToolOrchestrator(
        llm_skill=llm_skill,
        skills=skills,
        db_manager=db_manager,
        use_react=False,
    )

    logger.info("✅ Orchestrator initialized")
    logger.info("   Skills: " + ", ".join(skills.keys()))

    # Execute
    logger.info("\n🚀 Executing: 'Search for information about test'")
    result = orchestrator.execute_with_tools(
        "Search for information about test", max_iterations=5
    )

    logger.info(f"\n📊 Result:")
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Iterations: {result.iterations}")
    logger.info(f"   Tool calls made: {result.tool_calls_made}")
    logger.info(f"   Final response: {result.final_response}")

    # Check database
    logger.info(f"\n🗄️  Database:")
    sessions = db_manager.get_test_performance(days=1)
    logger.info(f"   Sessions: {len(sessions)}")

    tool_calls = db_manager.get_tool_success_rate()
    logger.info(f"   Tool calls tracked: {len(tool_calls)}")
    for tc in tool_calls:
        logger.info(
            f"      - {tc['tool_name']}: {tc['total_calls']} calls, {tc['success_rate']:.1%} success"
        )

    assert result.success is True, "Expected success"
    assert result.tool_calls_made == 1, (
        f"Expected 1 tool call, got {result.tool_calls_made}"
    )
    assert search_skill.call_count == 1, (
        f"Expected 1 search execution, got {search_skill.call_count}"
    )
    assert llm_skill.call_count == 2, (
        f"Expected 2 LLM calls, got {llm_skill.call_count}"
    )

    logger.info("\n✅ TEST 1 PASSED\n")


def test_multi_tool_chaining():
    """Test multi-tool call with chaining."""
    logger.info("=" * 80)
    logger.info("TEST 2: Multi-Tool Chaining")
    logger.info("=" * 80)

    from core.database_manager import DatabaseManager
    from core.tool_orchestrator import ToolOrchestrator
    from core.skill import BaseSkill, SkillResult
    from typing import List, Dict, Any

    # Setup database
    db_path = Path("data/test_tool_calling.db")
    db_manager = DatabaseManager(db_path=str(db_path))

    # Create mock LLM skill
    class MockLLMSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="mock_llm", version="1.0.0")
            self.model = "mock-model"
            self.base_url = "http://mock"
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1

            # First call - search
            if self.call_count == 1:
                logger.info(f"   LLM Call #{self.call_count}: Requesting search")
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "First step - searching",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "test"}',
                                    },
                                }
                            ],
                        },
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                )
            # Second call - calculate
            elif self.call_count == 2:
                logger.info(f"   LLM Call #{self.call_count}: Requesting calculation")
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "Second step - calculating",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"value": 42}',
                                    },
                                }
                            ],
                        },
                        "usage": {
                            "prompt_tokens": 30,
                            "completion_tokens": 20,
                            "total_tokens": 50,
                        },
                    },
                )
            # Third call - final answer
            else:
                logger.info(f"   LLM Call #{self.call_count}: Returning final answer")
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "Final answer: result calculated",
                        },
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 10,
                            "total_tokens": 60,
                        },
                    },
                )

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

    # Create mock skills
    class MockSearchSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="search", version="1.0.0")
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1
            query = kwargs.get("query", "")
            logger.info(f"   Search executed: query='{query}'")
            return SkillResult(success=True, data={"results": [f"Result: {query}"]})

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

        def get_tool_schema(self) -> Dict:
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

    class MockCalculateSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="calculate", version="1.0.0")
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1
            value = kwargs.get("value", 0)
            logger.info(f"   Calculate executed: value={value}")
            return SkillResult(success=True, data={"result": value * 2})

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

        def get_tool_schema(self) -> Dict:
            return {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate something",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "integer",
                                "description": "Value to calculate",
                            }
                        },
                        "required": ["value"],
                    },
                },
            }

    # Create orchestrator
    llm_skill = MockLLMSkill()
    skills = {"search": MockSearchSkill(), "calculate": MockCalculateSkill()}

    orchestrator = ToolOrchestrator(
        llm_skill=llm_skill,
        skills=skills,
        db_manager=db_manager,
        use_react=False,
    )

    logger.info("✅ Orchestrator initialized")
    logger.info("   Skills: " + ", ".join(skills.keys()))

    # Execute
    logger.info("\n🚀 Executing: 'Search and calculate'")
    result = orchestrator.execute_with_tools("Search and calculate", max_iterations=5)

    logger.info(f"\n📊 Result:")
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Iterations: {result.iterations}")
    logger.info(f"   Tool calls made: {result.tool_calls_made}")
    logger.info(f"   Final response: {result.final_response}")

    assert result.success is True, "Expected success"
    assert result.tool_calls_made == 2, (
        f"Expected 2 tool calls, got {result.tool_calls_made}"
    )
    assert result.iterations == 2, f"Expected 2 iterations, got {result.iterations}"
    assert skills["search"].call_count == 1, (
        f"Expected 1 search, got {skills['search'].call_count}"
    )
    assert skills["calculate"].call_count == 1, (
        f"Expected 1 calculate, got {skills['calculate'].call_count}"
    )
    assert llm_skill.call_count == 3, (
        f"Expected 3 LLM calls, got {llm_skill.call_count}"
    )

    logger.info("\n✅ TEST 2 PASSED\n")


def test_parallel_tool_calls():
    """Test parallel tool execution."""
    logger.info("=" * 80)
    logger.info("TEST 3: Parallel Tool Calls")
    logger.info("=" * 80)

    from core.database_manager import DatabaseManager
    from core.tool_orchestrator import ToolOrchestrator
    from core.skill import BaseSkill, SkillResult
    from typing import List, Dict, Any

    # Setup database
    db_path = Path("data/test_tool_calling.db")
    db_manager = DatabaseManager(db_path=str(db_path))

    # Create mock LLM skill
    class MockLLMSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="mock_llm", version="1.0.0")
            self.model = "mock-model"
            self.base_url = "http://mock"
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1

            # First call - return parallel tool calls
            if self.call_count == 1:
                logger.info(
                    f"   LLM Call #{self.call_count}: Requesting parallel tools"
                )
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "Running parallel searches",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "test1"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "test2"}',
                                    },
                                },
                            ],
                        },
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 30,
                            "total_tokens": 40,
                        },
                    },
                )
            # Second call - final answer
            else:
                logger.info(f"   LLM Call #{self.call_count}: Returning final answer")
                return SkillResult(
                    success=True,
                    data={
                        "message": {
                            "role": "assistant",
                            "content": "Complete: found results for both queries",
                        },
                        "usage": {
                            "prompt_tokens": 40,
                            "completion_tokens": 10,
                            "total_tokens": 50,
                        },
                    },
                )

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

    # Create mock search skill
    class MockSearchSkill(BaseSkill):
        def __init__(self):
            super().__init__(name="search", version="1.0.0")
            self.call_count = 0

        def execute(self, **kwargs) -> SkillResult:
            self.call_count += 1
            query = kwargs.get("query", "")
            logger.info(f"   Search executed: query='{query}'")
            return SkillResult(success=True, data={"results": [f"Result: {query}"]})

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> List[str]:
            return []

        def get_tool_schema(self) -> Dict:
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

    # Create orchestrator
    llm_skill = MockLLMSkill()
    search_skill = MockSearchSkill()
    skills = {"search": search_skill}

    orchestrator = ToolOrchestrator(
        llm_skill=llm_skill,
        skills=skills,
        db_manager=db_manager,
        use_react=False,
    )

    logger.info("✅ Orchestrator initialized")

    # Execute
    logger.info("\n🚀 Executing: 'Search for test1 and test2'")
    result = orchestrator.execute_with_tools(
        "Search for test1 and test2", max_iterations=5
    )

    logger.info(f"\n📊 Result:")
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Iterations: {result.iterations}")
    logger.info(f"   Tool calls made: {result.tool_calls_made}")
    logger.info(f"   Final response: {result.final_response}")

    assert result.success is True, "Expected success"
    assert result.tool_calls_made == 2, (
        f"Expected 2 tool calls, got {result.tool_calls_made}"
    )
    assert search_skill.call_count == 2, (
        f"Expected 2 search executions, got {search_skill.call_count}"
    )
    assert llm_skill.call_count == 2, (
        f"Expected 2 LLM calls, got {llm_skill.call_count}"
    )

    logger.info("\n✅ TEST 3 PASSED\n")


def main():
    """Run all tests."""
    logger.info("🧪 Starting Tool Calling Tests\n")

    try:
        test_simple_tool_call()
        test_multi_tool_chaining()
        test_parallel_tool_calls()

        logger.info("=" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)
        return 0
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
