"""Comprehensive integration tests for the VibeAgent system.

These tests validate the entire system working together end-to-end, covering:
- End-to-end orchestration with database tracking
- ReAct integration with reasoning
- Parallel execution
- Self-correction
- Tree of Thoughts
- Plan-and-Execute
- Context management
- Analytics
- Model configuration
- Error handling
- Performance
"""

import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List
from datetime import datetime

from core.database_manager import DatabaseManager
from core.tool_orchestrator import ToolOrchestrator, OrchestratorResult
from core.parallel_executor import ParallelExecutor, ParallelExecutorConfig
from core.self_corrector import SelfCorrector, SelfCorrectorConfig, CorrectionTrigger
from core.tot_orchestrator import (
    TreeOfThoughtsOrchestrator,
    ToTConfig,
    ExplorationStrategy,
)
from core.plan_execute_orchestrator import (
    PlanExecuteOrchestrator,
    PlanExecuteOrchestratorConfig,
    Plan,
    PlanStep,
    StepStatus,
)
from core.context_manager import ContextManager, ContextConfig, ContextType
from core.analytics_engine import AnalyticsEngine
from core.skill import BaseSkill, SkillResult
from core.retry_manager import RetryManager


class MockLLMSkill:
    """Mock LLM skill for testing."""

    def __init__(self, base_url="http://localhost:8000", model="gpt-4"):
        self.base_url = base_url
        self.model = model
        self.call_count = 0
        self.responses = []

    def execute(self, **kwargs) -> SkillResult:
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return SkillResult(
            success=True,
            data={"content": "Mock LLM response", "model": self.model},
        )


class MockSkill(BaseSkill):
    """Mock skill for testing."""

    def __init__(self, name="mock_skill", should_fail=False):
        super().__init__(name=name)
        self.should_fail = should_fail
        self.execute_count = 0

    def validate(self) -> bool:
        return True

    def execute(self, **kwargs) -> SkillResult:
        self.execute_count += 1
        if self.should_fail:
            return SkillResult(success=False, error=f"Mock skill {self.name} failed")
        return SkillResult(success=True, data={"result": f"Executed {self.name}"})

    def get_tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"Mock {self.name} skill",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }

    def get_dependencies(self) -> List[str]:
        return []


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_integration.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def db_manager(temp_db_path):
    """Create a DatabaseManager instance with temporary database."""
    return DatabaseManager(db_path=temp_db_path)


@pytest.fixture
def mock_llm():
    """Create a mock LLM skill."""
    return MockLLMSkill()


@pytest.fixture
def mock_skills():
    """Create mock skills for testing."""
    return {
        "search": MockSkill(name="search"),
        "calculate": MockSkill(name="calculate"),
        "validate": MockSkill(name="validate"),
        "process": MockSkill(name="process"),
    }


@pytest.fixture
def context_manager(db_manager, mock_llm):
    """Create a ContextManager instance."""
    return ContextManager(
        config=ContextConfig(max_context_tokens=4000),
        db_manager=db_manager,
        llm_skill=mock_llm,
    )


@pytest.fixture
def analytics_engine(db_manager):
    """Create an AnalyticsEngine instance."""
    return AnalyticsEngine(db_manager=db_manager)


class TestEndToEndOrchestration:
    """Tests for end-to-end orchestration with database tracking."""

    def test_simple_tool_call_with_database_tracking(
        self, db_manager, mock_llm, mock_skills
    ):
        """Test simple tool call with complete database tracking."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=False,
        )

        mock_llm.responses.append(
            SkillResult(
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
                                    "arguments": json.dumps({"query": "test"}),
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
        )

        mock_llm.responses.append(
            SkillResult(
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
        )

        result = orchestrator.execute_with_tools(
            "Search for information about test", max_iterations=5
        )

        assert result.success is True
        assert result.tool_calls_made == 1

        sessions = db_manager.get_test_performance(days=1)
        assert len(sessions) > 0

        tool_calls = db_manager.get_tool_success_rate()
        assert len(tool_calls) > 0
        assert tool_calls[0]["tool_name"] == "search"

    def test_multi_tool_call_with_chaining(self, db_manager, mock_llm, mock_skills):
        """Test multi-tool call with chaining."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=False,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "First step",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": "test"}),
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
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Second step",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": json.dumps({"value": 42}),
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
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {"role": "assistant", "content": "Final answer"},
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                        "total_tokens": 60,
                    },
                },
            )
        )

        result = orchestrator.execute_with_tools(
            "Search and calculate", max_iterations=5
        )

        assert result.success is True
        assert result.tool_calls_made == 2
        assert result.iterations == 2

    def test_parallel_tool_execution(self, db_manager, mock_llm, mock_skills):
        """Test parallel tool execution."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=False,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Running parallel",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": "test1"}),
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": json.dumps({"value": 1}),
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
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {"role": "assistant", "content": "Complete"},
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 10,
                        "total_tokens": 50,
                    },
                },
            )
        )

        result = orchestrator.execute_with_tools("Do multiple things", max_iterations=5)

        assert result.success is True
        assert result.tool_calls_made == 2

    def test_error_recovery_with_retry(self, db_manager, mock_llm, mock_skills):
        """Test error recovery with retry logic."""
        failing_skill = MockSkill(name="failing_skill", should_fail=True)
        mock_skills["failing"] = failing_skill

        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=False,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Trying failing tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "failing",
                                    "arguments": json.dumps({}),
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
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Failed but continuing",
                    },
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 10,
                        "total_tokens": 40,
                    },
                },
            )
        )

        result = orchestrator.execute_with_tools(
            "Try failing operation", max_iterations=5
        )

        assert result.tool_calls_made == 1
        assert failing_skill.execute_count > 0

    def test_self_correction_on_failure(self, db_manager, mock_llm, mock_skills):
        """Test self-correction mechanism on failure."""
        corrector = SelfCorrector(
            llm_skill=mock_llm, db_manager=db_manager, config=SelfCorrectorConfig()
        )

        failing_skill = MockSkill(name="failing_skill", should_fail=True)
        mock_skills["failing"] = failing_skill

        context = {
            "tool_result": SkillResult(success=False, error="Tool execution failed"),
            "iteration": 1,
            "consecutive_errors": 2,
            "confidence": 0.3,
            "tool_name": "failing",
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is True
        assert trigger == CorrectionTrigger.REPEATED_FAILURES

        alternatives = corrector.generate_alternatives(context)

        assert len(alternatives) > 0

    def test_react_loop_with_reasoning(self, db_manager, mock_llm, mock_skills):
        """Test ReAct loop with reasoning tracking."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=True,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Thought: I need to search first\nAction: search",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": "test"}),
                                },
                            }
                        ],
                    },
                    "reasoning_content": "Thought: I need to search first",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 30,
                        "total_tokens": 40,
                    },
                },
            )
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Thought: Found results\nFinal Answer: Here are the results",
                    },
                    "reasoning_content": "Thought: Found results",
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 20,
                        "total_tokens": 60,
                    },
                },
            )
        )

        result = orchestrator.execute_with_tools(
            "Search and analyze", max_iterations=5, use_react=True
        )

        assert result.success is True
        assert len(result.reasoning_trace) > 0


class TestDatabaseIntegration:
    """Tests for complete database integration."""

    def test_session_creation_and_tracking(self, db_manager):
        """Test session creation and tracking."""
        session_id = db_manager.create_session(
            session_id="test-session-1",
            session_type="integration_test",
            model="gpt-4",
            orchestrator_type="ToolOrchestrator",
            metadata={"test": True},
        )

        assert session_id > 0

        session = db_manager.get_session(session_id)

        assert session is not None
        assert session["session_id"] == "test-session-1"
        assert session["session_type"] == "integration_test"

    def test_message_storage_and_retrieval(self, db_manager):
        """Test message storage and retrieval."""
        session_id = db_manager.create_session(
            session_id="test-session-2", session_type="test", model="gpt-4"
        )

        db_manager.add_message(
            session_id=session_id,
            role="user",
            content="Hello",
            message_index=0,
            tokens_input=5,
            model="gpt-4",
        )

        db_manager.add_message(
            session_id=session_id,
            role="assistant",
            content="Hi there!",
            message_index=1,
            tokens_output=8,
            model="gpt-4",
        )

        messages = db_manager.get_session_messages(session_id)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_tool_call_tracking(self, db_manager):
        """Test tool call tracking."""
        session_id = db_manager.create_session(
            session_id="test-session-3", session_type="test", model="gpt-4"
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="search",
            parameters={"query": "test"},
            execution_time_ms=100,
            success=True,
        )

        assert tool_call_id > 0

        db_manager.add_tool_result(
            tool_call_id=tool_call_id,
            success=True,
            data={"results": ["result1", "result2"]},
        )

        tool_calls = db_manager.get_tool_success_rate()

        assert len(tool_calls) > 0
        assert tool_calls[0]["tool_name"] == "search"
        assert tool_calls[0]["total_calls"] == 1
        assert tool_calls[0]["successful_calls"] == 1

    def test_test_case_and_run_tracking(self, db_manager):
        """Test test case and run tracking."""
        test_case_id = db_manager.create_test_case(
            name="Integration Test Case",
            category="integration",
            description="Test case for integration",
            messages=[{"role": "user", "content": "test"}],
            tools=[{"name": "search"}],
            expected_tools=[{"name": "search"}],
        )

        assert test_case_id > 0

        test_run_id = db_manager.create_test_run(
            test_case_id=test_case_id, run_number=1, status="completed"
        )

        assert test_run_id > 0

        from datetime import datetime

        db_manager.update_test_run(
            test_run_id,
            status="completed",
            final_status="success",
            total_iterations=3,
            total_tool_calls=2,
            completed_at=datetime.now(),
        )

        performance = db_manager.get_test_performance(days=1)

        assert len(performance) > 0
        assert performance[0]["test_case_name"] == "Integration Test Case"

    def test_judge_evaluation_storage(self, db_manager):
        """Test judge evaluation storage."""
        test_case_id = db_manager.create_test_case(
            name="Eval Test",
            category="evaluation",
            description="Test case for evaluation",
            messages=[{"role": "user", "content": "test"}],
            tools=[{"name": "search"}],
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
            evaluation_type="semantic",
            criteria={"accuracy": 0.9},
            evaluation_time_ms=500,
        )

        assert evaluation_id > 0

    def test_performance_metrics_storage(self, db_manager):
        """Test performance metrics storage."""
        session_id = db_manager.create_session(
            session_id="test-session-metrics",
            session_type="performance_test",
            model="gpt-4",
        )

        metric_id = db_manager.add_performance_metric(
            session_id=session_id,
            metric_name="execution_time",
            metric_value=1234.56,
            metric_unit="milliseconds",
            metadata={"phase": "planning"},
        )

        assert metric_id > 0


class TestReActIntegration:
    """Tests for ReAct integration."""

    def test_react_mode_execution(self, db_manager, mock_llm, mock_skills):
        """Test ReAct mode execution."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=True,
            react_config={"max_reasoning_steps": 10},
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Thought: Need to search\nAction: search",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": "test"}),
                                },
                            }
                        ],
                    },
                    "reasoning_content": "Thought: Need to search",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            )
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Thought: Got results\nFinal Answer: Results found",
                    },
                    "reasoning_content": "Thought: Got results",
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 10,
                        "total_tokens": 40,
                    },
                },
            )
        )

        result = orchestrator.execute_with_tools(
            "Search with reasoning", max_iterations=5, use_react=True
        )

        assert result.success is True
        assert len(result.reasoning_trace) > 0

    def test_reasoning_step_tracking(self, db_manager, mock_llm, mock_skills):
        """Test reasoning step tracking."""
        session_id = db_manager.create_session(
            session_id="test-reasoning",
            session_type="react_test",
            model="gpt-4",
            orchestrator_type="ToolOrchestrator",
        )

        step_id = db_manager.add_reasoning_step(
            session_id=session_id,
            iteration=1,
            step_type="thought",
            content="I need to search for information",
            metadata={"confidence": 0.8},
        )

        assert step_id > 0

    def test_reflection_on_errors(self, db_manager, mock_llm, mock_skills):
        """Test reflection on errors."""
        corrector = SelfCorrector(
            llm_skill=mock_llm, db_manager=db_manager, config=SelfCorrectorConfig()
        )

        reflection = corrector.reflect_on_failure(
            "Network timeout error",
            {
                "tool_name": "search",
                "iteration": 2,
                "parameters": {"query": "test"},
            },
        )

        assert reflection is not None
        assert "error" in reflection
        assert "error_pattern" in reflection
        assert "suggested_corrections" in reflection

    def test_plan_revision(self, db_manager, mock_llm, mock_skills):
        """Test plan revision during ReAct."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=True,
            react_config={"plan_revision_threshold": 2},
        )

        assert orchestrator._should_revise_plan(3, SkillResult(success=False)) is True

    def test_reasoning_trace_visualization(self, db_manager, mock_llm, mock_skills):
        """Test reasoning trace visualization."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=True,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Thought: Analyzing\nFinal Answer: Done",
                    },
                    "reasoning_content": "Thought: Analyzing",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                },
            )
        )

        result = orchestrator.execute_with_tools("Analyze", use_react=True)

        assert len(result.reasoning_trace) > 0


class TestParallelExecutionIntegration:
    """Tests for parallel execution integration."""

    def test_independent_parallel_calls(self, db_manager, mock_skills):
        """Test independent parallel calls."""
        executor = ParallelExecutor(
            skills=mock_skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(
                max_parallel_calls=3,
                enable_parallel=True,
                track_performance=True,
            ),
        )

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "test1"}),
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"value": 1}),
                },
            },
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "validate",
                    "arguments": json.dumps({"data": "test"}),
                },
            },
        ]

        result = executor.execute_parallel(tool_calls)

        assert result.success is True
        assert len(result.results) == 3
        assert result.speedup > 0

    def test_dependent_sequential_calls(self, db_manager, mock_skills):
        """Test dependent sequential calls."""
        executor = ParallelExecutor(
            skills=mock_skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(max_parallel_calls=1),
        )

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "test"}),
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"value": 1}),
                },
            },
        ]

        result = executor.execute_parallel(tool_calls)

        assert result.success is True
        assert len(result.results) == 2

    def test_mixed_parallel_sequential(self, db_manager, mock_skills):
        """Test mixed parallel and sequential execution."""
        executor = ParallelExecutor(
            skills=mock_skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(max_parallel_calls=2, enable_parallel=True),
        )

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "test1"}),
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "test2"}),
                },
            },
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"value": 1}),
                },
            },
        ]

        result = executor.execute_parallel(tool_calls)

        assert result.success is True
        assert len(result.results) == 3

    def test_error_handling_in_parallel(self, db_manager, mock_skills):
        """Test error handling in parallel execution."""
        mock_skills["failing"] = MockSkill(name="failing", should_fail=True)

        executor = ParallelExecutor(
            skills=mock_skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(enable_parallel=True),
        )

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": json.dumps({})},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "failing", "arguments": json.dumps({})},
            },
        ]

        result = executor.execute_parallel(tool_calls)

        assert len(result.results) == 2
        assert result.results[0].get("success") is True
        assert result.results[1].get("success") is False

    def test_performance_measurement(self, db_manager, mock_skills):
        """Test performance measurement in parallel execution."""
        executor = ParallelExecutor(
            skills=mock_skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(track_performance=True),
        )

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": json.dumps({})},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "calculate", "arguments": json.dumps({})},
            },
        ]

        result = executor.execute_parallel(tool_calls)

        assert result.total_time_ms > 0
        assert result.parallel_time_ms > 0
        assert result.sequential_time_estimate_ms > 0

        stats = executor.get_performance_stats()

        assert "avg_speedup" in stats
        assert "avg_execution_time_ms" in stats


class TestSelfCorrectionIntegration:
    """Tests for self-correction integration."""

    def test_error_detection_and_correction(self, db_manager, mock_llm, mock_skills):
        """Test error detection and correction."""
        corrector = SelfCorrector(
            llm_skill=mock_llm,
            db_manager=db_manager,
            config=SelfCorrectorConfig(max_correction_attempts=3),
        )

        session_id = db_manager.create_session(
            session_id="test-correction",
            session_type="correction_test",
            model="gpt-4",
        )

        context = {
            "tool_result": SkillResult(success=False, error="Tool failed"),
            "iteration": 1,
            "consecutive_errors": 2,
            "confidence": 0.3,
            "session_id": session_id,
            "tool_name": "search",
        }

        should_correct, trigger = corrector.should_self_correct(context)

        assert should_correct is True

    def test_alternative_strategy_generation(self, db_manager, mock_llm, mock_skills):
        """Test alternative strategy generation."""
        corrector = SelfCorrector(
            llm_skill=mock_llm,
            db_manager=db_manager,
            config=SelfCorrectorConfig(),
        )

        context = {
            "tool_result": SkillResult(success=False, error="Network timeout"),
            "tool_name": "search",
            "parameters": {"query": "test"},
            "error": "Network timeout",
        }

        alternatives = corrector.generate_alternatives(context)

        assert len(alternatives) > 0

    def test_correction_execution(self, db_manager, mock_llm, mock_skills):
        """Test correction execution."""
        corrector = SelfCorrector(
            llm_skill=mock_llm,
            db_manager=db_manager,
            config=SelfCorrectorConfig(),
        )

        from core.self_corrector import CorrectionStrategy, CorrectionType

        correction = CorrectionStrategy(
            strategy_type=CorrectionType.RETRY_WITH_DELAY,
            description="Retry with delay",
            delay_ms=100,
            confidence_score=0.8,
        )

        session_id = db_manager.create_session(
            session_id="test-correction-exec",
            session_type="correction_test",
            model="gpt-4",
        )

        context = {
            "tool_result": SkillResult(success=False, error="Test error"),
            "session_id": session_id,
        }

        result = corrector.apply_correction(correction, context)

        assert result is not None

    def test_correction_tracking(self, db_manager, mock_llm, mock_skills):
        """Test correction tracking in database."""
        corrector = SelfCorrector(
            llm_skill=mock_llm,
            db_manager=db_manager,
            config=SelfCorrectorConfig(),
        )

        session_id = db_manager.create_session(
            session_id="test-correction-track",
            session_type="correction_test",
            model="gpt-4",
        )

        context = {
            "tool_result": SkillResult(success=False, error="Test error"),
            "session_id": session_id,
        }

        from core.self_corrector import CorrectionStrategy, CorrectionType

        correction = CorrectionStrategy(
            strategy_type=CorrectionType.FALLBACK_STRATEGY,
            description="Use fallback",
            confidence_score=0.7,
        )

        result = corrector.apply_correction(correction, context)

        assert result is not None


class TestTreeOfThoughtsIntegration:
    """Tests for Tree of Thoughts integration."""

    def test_branch_generation(self, db_manager, mock_llm, mock_skills):
        """Test branch generation in ToT."""
        orchestrator = TreeOfThoughtsOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            tot_config=ToTConfig(branching_factor=3, max_tree_depth=5),
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "content": "Initial thought about the problem",
                    "model": "gpt-4",
                },
            )
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "content": json.dumps(
                        [
                            {
                                "thought": "Approach 1",
                                "action": "search",
                                "confidence": 0.8,
                            },
                            {
                                "thought": "Approach 2",
                                "action": "calculate",
                                "confidence": 0.7,
                            },
                            {
                                "thought": "Approach 3",
                                "action": "validate",
                                "confidence": 0.6,
                            },
                        ]
                    ),
                    "model": "gpt-4",
                },
            )
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={"content": "Score: 0.8", "model": "gpt-4"},
            )
        )

        result = orchestrator.execute_with_tot(
            "Solve complex problem", max_iterations=5
        )

        assert result is not None

    def test_branch_evaluation(self, db_manager, mock_llm, mock_skills):
        """Test branch evaluation in ToT."""
        orchestrator = TreeOfThoughtsOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            tot_config=ToTConfig(),
        )

        from core.tot_orchestrator import ThoughtNode

        parent = ThoughtNode(
            node_id="parent",
            thought="Parent thought",
            depth=0,
            score=0.5,
        )

        child = ThoughtNode(
            node_id="child",
            thought="Child thought",
            depth=1,
            parent_id="parent",
            confidence=0.7,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "content": '{"score": 0.8, "reasoning": "Good approach"}',
                    "model": "gpt-4",
                },
            )
        )

        score = orchestrator._evaluate_branch(child, parent)

        assert 0.0 <= score <= 1.0

    def test_path_selection(self, db_manager, mock_llm, mock_skills):
        """Test path selection in ToT."""
        orchestrator = TreeOfThoughtsOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            tot_config=ToTConfig(),
        )

        from core.tot_orchestrator import ThoughtTree, ThoughtNode

        root = ThoughtNode(node_id="root", thought="Root", depth=0, score=0.5)

        child1 = ThoughtNode(
            node_id="child1",
            thought="Path 1",
            depth=1,
            parent_id="root",
            score=0.8,
        )

        child2 = ThoughtNode(
            node_id="child2",
            thought="Path 2",
            depth=1,
            parent_id="root",
            score=0.6,
        )

        tree = ThoughtTree(root_id="root", nodes={})
        tree.add_node(root)
        tree.add_node(child1)
        tree.add_node(child2)
        root.children = ["child1", "child2"]

        best_path = orchestrator._select_best_path()

        assert best_path is not None

    def test_backtracking(self, db_manager, mock_llm, mock_skills):
        """Test backtracking in ToT."""
        orchestrator = TreeOfThoughtsOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            tot_config=ToTConfig(enable_backtracking=True),
        )

        from core.tot_orchestrator import ThoughtTree, ThoughtNode

        root = ThoughtNode(node_id="root", thought="Root", depth=0, score=0.5)

        failed = ThoughtNode(
            node_id="failed",
            thought="Failed path",
            depth=1,
            parent_id="root",
            score=0.1,
        )

        alternative = ThoughtNode(
            node_id="alternative",
            thought="Alternative path",
            depth=1,
            parent_id="root",
            score=0.7,
        )

        tree = ThoughtTree(root_id="root", nodes={})
        tree.add_node(root)
        tree.add_node(failed)
        tree.add_node(alternative)
        root.children = ["failed", "alternative"]

        backtrack_node = orchestrator._backtrack(failed)

        assert backtrack_node is not None

    def test_tree_visualization(self, db_manager, mock_llm, mock_skills):
        """Test tree visualization."""
        orchestrator = TreeOfThoughtsOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            tot_config=ToTConfig(enable_visualization=True),
        )

        from core.tot_orchestrator import ThoughtTree, ThoughtNode

        root = ThoughtNode(node_id="root", thought="Root thought", depth=0, score=0.5)

        tree = ThoughtTree(root_id="root", nodes={})
        tree.add_node(root)

        visualization = orchestrator.visualize_tree()

        assert "Tree of Thoughts Visualization" in visualization
        assert "Root thought" in visualization


class TestPlanAndExecuteIntegration:
    """Tests for Plan-and-Execute integration."""

    def test_plan_generation(self, db_manager, mock_llm, mock_skills):
        """Test plan generation."""
        orchestrator = PlanExecuteOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            config=PlanExecuteOrchestratorConfig(),
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "content": json.dumps(
                        {
                            "goal": "Execute complex task",
                            "steps": [
                                {
                                    "action": "Search for data",
                                    "tool": "search",
                                    "parameters": {"query": "test"},
                                    "dependencies": [],
                                    "complexity": 1,
                                },
                                {
                                    "action": "Calculate results",
                                    "tool": "calculate",
                                    "parameters": {"value": 42},
                                    "dependencies": ["step_1"],
                                    "complexity": 2,
                                },
                            ],
                        }
                    ),
                    "model": "gpt-4",
                },
            )
        )

        plan = orchestrator.generate_plan(
            "Search and calculate", {"session_id": "test"}
        )

        assert plan is not None
        assert len(plan.steps) > 0

    def test_plan_validation(self, db_manager, mock_llm, mock_skills):
        """Test plan validation."""
        orchestrator = PlanExecuteOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            config=PlanExecuteOrchestratorConfig(),
        )

        from core.plan_execute_orchestrator import Plan, PlanStep, StepType

        plan = Plan(
            plan_id="test-plan",
            goal="Test goal",
            steps=[
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.NORMAL,
                    action="Search",
                    tool="search",
                    parameters={"query": "test"},
                    dependencies=[],
                ),
                PlanStep(
                    step_id="step_2",
                    step_type=StepType.NORMAL,
                    action="Calculate",
                    tool="calculate",
                    parameters={"value": 42},
                    dependencies=["step_1"],
                ),
            ],
        )

        validation = orchestrator.validate_plan(plan)

        assert validation.is_valid is True

    def test_plan_execution(self, db_manager, mock_llm, mock_skills):
        """Test plan execution."""
        orchestrator = PlanExecuteOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            config=PlanExecuteOrchestratorConfig(),
        )

        from core.plan_execute_orchestrator import Plan, PlanStep, StepType

        plan = Plan(
            plan_id="test-plan-exec",
            goal="Execute plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.NORMAL,
                    action="Search",
                    tool="search",
                    parameters={"query": "test"},
                    dependencies=[],
                ),
                PlanStep(
                    step_id="step_2",
                    step_type=StepType.NORMAL,
                    action="Calculate",
                    tool="calculate",
                    parameters={"value": 42},
                    dependencies=["step_1"],
                ),
            ],
        )

        result = orchestrator.execute_plan(plan)

        assert result is not None
        assert result.steps_completed > 0

    def test_plan_adaptation(self, db_manager, mock_llm, mock_skills):
        """Test plan adaptation."""
        orchestrator = PlanExecuteOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            config=PlanExecuteOrchestratorConfig(
                adaptation_sensitivity=0.7, enable_plan_learning=True
            ),
        )

        from core.plan_execute_orchestrator import Plan, PlanStep, StepType

        plan = Plan(
            plan_id="test-plan-adapt",
            goal="Adaptive plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.NORMAL,
                    action="Search",
                    tool="search",
                    parameters={"query": "test"},
                    dependencies=[],
                    status=StepStatus.COMPLETED,
                ),
                PlanStep(
                    step_id="step_2",
                    step_type=StepType.NORMAL,
                    action="Calculate",
                    tool="calculate",
                    parameters={"value": 42},
                    dependencies=["step_1"],
                    status=StepStatus.FAILED,
                    error="Calculation failed",
                ),
            ],
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "content": json.dumps(
                        {
                            "modifications": [
                                {
                                    "action": "modify",
                                    "step_id": "step_2",
                                    "new_step": {
                                        "action": "Calculate with retry",
                                        "tool": "calculate",
                                        "parameters": {"value": 42, "retry": True},
                                        "dependencies": ["step_1"],
                                    },
                                }
                            ]
                        }
                    ),
                    "model": "gpt-4",
                },
            )
        )

        adapted_plan = orchestrator.adapt_plan(plan, {})

        assert adapted_plan is not None

    def test_plan_visualization(self, db_manager, mock_llm, mock_skills):
        """Test plan visualization."""
        orchestrator = PlanExecuteOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
        )

        from core.plan_execute_orchestrator import Plan, PlanStep, StepType

        plan = Plan(
            plan_id="test-plan-vis",
            goal="Visualize this plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.NORMAL,
                    action="First step",
                    tool="search",
                    parameters={},
                    dependencies=[],
                    status=StepStatus.COMPLETED,
                ),
                PlanStep(
                    step_id="step_2",
                    step_type=StepType.NORMAL,
                    action="Second step",
                    tool="calculate",
                    parameters={},
                    dependencies=["step_1"],
                    status=StepStatus.PENDING,
                ),
            ],
        )

        visualization = orchestrator.visualize_plan(plan)

        assert "Plan: Visualize this plan" in visualization
        assert "step_1" in visualization
        assert "step_2" in visualization


class TestContextManagementIntegration:
    """Tests for context management integration."""

    def test_context_windowing(self, context_manager):
        """Test context windowing."""
        messages = [
            {"role": "user", "content": "Message " + str(i) * 100} for i in range(20)
        ]

        managed = context_manager.manage_context(messages, max_tokens=1000)

        assert len(managed) <= len(messages)
        assert context_manager.get_token_usage(managed) <= 1000

    def test_message_summarization(self, context_manager):
        """Test message summarization."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]

        summary = context_manager.summarize_messages(messages)

        assert summary is not None
        assert summary.original_messages == len(messages)
        assert summary.token_reduction > 0

    def test_importance_scoring(self, context_manager):
        """Test importance scoring."""
        message = {"role": "user", "content": "This is important!"}

        score = context_manager.calculate_importance(message)

        assert score.importance_score > 0
        assert score.final_score > 0

    def test_context_retrieval(self, context_manager):
        """Test context retrieval."""
        history = [
            {"role": "user", "content": "Search for Python"},
            {"role": "assistant", "content": "Python is a programming language"},
            {"role": "user", "content": "What about JavaScript?"},
            {"role": "assistant", "content": "JavaScript is also popular"},
        ]

        relevant = context_manager.retrieve_relevant_context("programming", history)

        assert len(relevant) > 0

    def test_token_optimization(self, context_manager):
        """Test token optimization."""
        messages = [
            {"role": "user", "content": "Long message " * 50} for _ in range(10)
        ]

        optimized = context_manager.optimize_for_tokens(messages, max_tokens=500)

        assert context_manager.get_token_usage(optimized) <= 500


class TestAnalyticsIntegration:
    """Tests for analytics integration."""

    def test_performance_analysis(self, db_manager, analytics_engine):
        """Test performance analysis."""
        session_id = db_manager.create_session(
            session_id="test-analytics-1",
            session_type="test",
            model="gpt-4",
        )

        db_manager.add_performance_metric(
            session_id=session_id,
            metric_name="execution_time",
            metric_value=1000,
            metric_unit="ms",
        )

        db_manager.add_performance_metric(
            session_id=session_id,
            metric_name="token_usage",
            metric_value=500,
        )

        metrics = analytics_engine.get_execution_time_distribution()

        assert metrics is not None

    def test_pattern_detection(self, db_manager, analytics_engine):
        """Test pattern detection."""
        session_id = db_manager.create_session(
            session_id="test-analytics-2",
            session_type="test",
            model="gpt-4",
        )

        for i in range(5):
            db_manager.add_tool_call(
                session_id=session_id,
                call_index=i,
                tool_name="search",
                parameters={"query": f"test{i}"},
                execution_time_ms=100,
                success=(i % 2 == 0),
            )

        patterns = analytics_engine.find_successful_patterns()

        assert patterns is not None

    def test_trend_analysis(self, db_manager, analytics_engine):
        """Test trend analysis."""
        session_id = db_manager.create_session(
            session_id="test-analytics-3",
            session_type="test",
            model="gpt-4",
        )

        db_manager.update_session(
            session_id,
            final_status="success",
            total_iterations=3,
            total_tool_calls=2,
            total_duration_ms=1500,
        )

        trends = analytics_engine.get_success_rate_trend(days=1)

        assert trends is not None

    def test_insight_generation(self, db_manager, analytics_engine):
        """Test insight generation."""
        insights = analytics_engine.generate_daily_insights(days=1)

        assert insights is not None
        assert "summary" in insights

    def test_report_generation(self, db_manager, analytics_engine):
        """Test report generation."""
        report = analytics_engine.generate_weekly_report()

        assert report is not None
        assert "overview" in report
        assert "performance" in report


class TestModelConfigIntegration:
    """Tests for model configuration integration."""

    def test_model_specific_configuration(self, mock_llm):
        """Test model-specific configuration."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills={},
            use_react=True,
            react_config={"max_reasoning_steps": 15, "confidence_threshold": 0.8},
        )

        assert orchestrator.react_config is not None
        assert orchestrator.react_config["max_reasoning_steps"] == 15

    def test_capability_detection(self, mock_llm):
        """Test capability detection."""
        mock_llm.model = "gpt-4-turbo"

        orchestrator = ToolOrchestrator(llm_skill=mock_llm, skills={}, use_react=True)

        model_type = orchestrator._get_model_type()

        assert model_type == "gpt4"

    def test_configuration_switching(self, mock_llm):
        """Test configuration switching."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills={},
            use_react=True,
            react_config={"max_reasoning_steps": 10},
        )

        orchestrator.react_config["max_reasoning_steps"] = 20

        assert orchestrator.react_config["max_reasoning_steps"] == 20


class TestErrorHandlingIntegration:
    """Tests for error handling integration."""

    def test_network_error_handling(self, db_manager, mock_llm, mock_skills):
        """Test network error handling."""
        session_id = db_manager.create_session(
            session_id="test-error-net",
            session_type="error_test",
            model="gpt-4",
        )

        tool_call_id = db_manager.add_tool_call(
            session_id=session_id,
            call_index=0,
            tool_name="search",
            parameters={"query": "test"},
            execution_time_ms=100,
            success=False,
            error_message="Connection refused",
            error_type="NetworkError",
        )

        db_manager.add_error_recovery(
            session_id=session_id,
            tool_call_id=tool_call_id,
            error_type="NetworkError",
            recovery_strategy="retry_with_backoff",
            attempt_number=1,
            success=True,
            original_error="Connection refused",
        )

        assert tool_call_id > 0

    def test_timeout_error_handling(self, db_manager, mock_llm, mock_skills):
        """Test timeout error handling."""
        corrector = SelfCorrector(llm_skill=mock_llm, db_manager=db_manager)

        error_pattern = corrector.get_error_pattern("Operation timed out")

        assert error_pattern.value == "timeout_error"

    def test_validation_error_handling(self, db_manager, mock_llm, mock_skills):
        """Test validation error handling."""
        corrector = SelfCorrector(llm_skill=mock_llm, db_manager=db_manager)

        error_pattern = corrector.get_error_pattern("Invalid input parameter")

        assert error_pattern.value == "invalid_input"

    def test_permission_error_handling(self, db_manager, mock_llm, mock_skills):
        """Test permission error handling."""
        corrector = SelfCorrector(llm_skill=mock_llm, db_manager=db_manager)

        error_pattern = corrector.get_error_pattern("Access denied")

        assert error_pattern.value == "permission_denied"

    def test_rate_limit_error_handling(self, db_manager, mock_llm, mock_skills):
        """Test rate limit error handling."""
        corrector = SelfCorrector(llm_skill=mock_llm, db_manager=db_manager)

        error_pattern = corrector.get_error_pattern("Rate limit exceeded")

        assert error_pattern.value == "rate_limit"


class TestPerformanceIntegration:
    """Tests for performance integration."""

    def test_concurrent_sessions(self, db_manager, mock_llm, mock_skills):
        """Test concurrent session handling."""
        session_ids = []

        for i in range(5):
            session_id = db_manager.create_session(
                session_id=f"test-concurrent-{i}",
                session_type="concurrent_test",
                model="gpt-4",
            )
            session_ids.append(session_id)

        assert len(session_ids) == 5

        for session_id in session_ids:
            session = db_manager.get_session(session_id)
            assert session is not None

    def test_large_datasets(self, db_manager, analytics_engine):
        """Test handling large datasets."""
        session_id = db_manager.create_session(
            session_id="test-large",
            session_type="large_test",
            model="gpt-4",
        )

        for i in range(100):
            db_manager.add_tool_call(
                session_id=session_id,
                call_index=i,
                tool_name="search",
                parameters={"query": f"test{i}"},
                execution_time_ms=100,
                success=(i % 3 == 0),
            )

        tool_stats = db_manager.get_tool_success_rate()

        assert len(tool_stats) > 0
        assert tool_stats[0]["total_calls"] == 100

    def test_complex_queries(self, db_manager, analytics_engine):
        """Test complex analytical queries."""
        for i in range(10):
            session_id = db_manager.create_session(
                session_id=f"test-complex-{i}",
                session_type="complex_test",
                model="gpt-4" if i % 2 == 0 else "gpt-3.5-turbo",
            )

            db_manager.update_session(
                session_id,
                final_status="success" if i % 3 == 0 else "failed",
                total_iterations=i + 1,
                total_tool_calls=i,
                total_duration_ms=(i + 1) * 500,
            )

        model_comparison = analytics_engine.get_model_comparison()

        assert len(model_comparison) > 0

    def test_memory_usage(self, db_manager):
        """Test memory efficiency."""
        session_id = db_manager.create_session(
            session_id="test-memory",
            session_type="memory_test",
            model="gpt-4",
        )

        large_data = {"result": "x" * 10000}

        for i in range(10):
            db_manager.add_message(
                session_id=session_id,
                role="user",
                content=f"Message {i}",
                message_index=i,
            )

        messages = db_manager.get_session_messages(session_id)

        assert len(messages) == 10

    def test_response_time(self, db_manager, mock_llm, mock_skills):
        """Test response time measurement."""
        orchestrator = ToolOrchestrator(
            llm_skill=mock_llm,
            skills=mock_skills,
            db_manager=db_manager,
            use_react=False,
        )

        mock_llm.responses.append(
            SkillResult(
                success=True,
                data={
                    "message": {
                        "role": "assistant",
                        "content": "Response",
                    },
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                },
            )
        )

        start_time = time.time()
        result = orchestrator.execute_with_tools("Quick test", max_iterations=1)
        elapsed = (time.time() - start_time) * 1000

        assert result.success is True
        assert elapsed < 5000
