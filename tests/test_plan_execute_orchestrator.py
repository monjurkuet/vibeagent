"""Tests for Plan-and-Execute orchestrator."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from core.plan_execute_orchestrator import (
    Plan,
    PlanExecuteOrchestrator,
    PlanExecuteOrchestratorConfig,
    PlanStep,
    StepStatus,
    StepType,
)
from core.skill import BaseSkill, SkillResult


class MockSkill(BaseSkill):
    """Mock skill for testing."""

    def __init__(self, name: str, return_value=None):
        super().__init__(name)
        self.return_value = return_value or {"result": "success"}

    def execute(self, **kwargs) -> SkillResult:
        return SkillResult(success=True, data=self.return_value)

    def validate(self) -> bool:
        return True

    def get_dependencies(self) -> list:
        return []

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"Mock {self.name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


@pytest.fixture
def mock_llm_skill():
    """Create mock LLM skill."""
    llm = Mock()
    llm.base_url = "http://localhost:11434/v1"
    llm.model = "llama3.2"
    return llm


@pytest.fixture
def mock_skills():
    """Create mock skills."""
    return {
        "search": MockSkill("search", {"results": ["result1", "result2"]}),
        "analyze": MockSkill("analyze", {"analysis": "completed"}),
        "summarize": MockSkill("summarize", {"summary": "done"}),
    }


@pytest.fixture
def orchestrator(mock_llm_skill, mock_skills):
    """Create orchestrator instance."""
    return PlanExecuteOrchestrator(
        llm_skill=mock_llm_skill,
        skills=mock_skills,
        db_manager=None,
        config=PlanExecuteOrchestratorConfig(
            max_plan_steps=10,
            enable_parallel=False,
        ),
    )


class TestPlanStep:
    """Test PlanStep functionality."""

    def test_plan_step_creation(self):
        """Test creating a plan step."""
        step = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Search web",
            tool="search",
            parameters={"query": "test"},
            dependencies=[],
            complexity=1,
        )

        assert step.step_id == "step_1"
        assert step.step_type == StepType.NORMAL
        assert step.action == "Search web"
        assert step.tool == "search"
        assert step.status == StepStatus.PENDING

    def test_plan_step_to_dict(self):
        """Test converting step to dictionary."""
        step = PlanStep(
            step_id="step_1",
            step_type=StepType.CONDITIONAL,
            action="Test action",
            tool="test",
            parameters={"key": "value"},
            dependencies=["step_0"],
            condition="True",
        )

        step_dict = step.to_dict()

        assert step_dict["step_id"] == "step_1"
        assert step_dict["step_type"] == "conditional"
        assert step_dict["condition"] == "True"
        assert step_dict["dependencies"] == ["step_0"]


class TestPlan:
    """Test Plan functionality."""

    def test_plan_creation(self):
        """Test creating a plan."""
        steps = [
            PlanStep(
                step_id="step_1",
                step_type=StepType.NORMAL,
                action="Search",
                tool="search",
                parameters={},
            )
        ]

        plan = Plan(
            plan_id="plan_1",
            goal="Test goal",
            steps=steps,
        )

        assert plan.plan_id == "plan_1"
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 1
        assert plan.status == "created"

    def test_add_step(self):
        """Test adding step to plan."""
        plan = Plan(plan_id="plan_1", goal="Test", steps=[])

        step = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Test",
            tool="test",
            parameters={},
        )

        plan.add_step(step)

        assert len(plan.steps) == 1
        assert plan.steps[0].step_id == "step_1"

    def test_get_ready_steps(self):
        """Test getting ready steps."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="First",
            tool="tool1",
            parameters={},
            dependencies=[],
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Second",
            tool="tool2",
            parameters={},
            dependencies=["step_1"],
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1, step2])

        ready = plan.get_ready_steps()

        assert len(ready) == 1
        assert ready[0].step_id == "step_1"

    def test_is_complete(self):
        """Test checking if plan is complete."""
        step = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Test",
            tool="test",
            parameters={},
            status=StepStatus.COMPLETED,
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step])

        assert plan.is_complete()

    def test_get_completion_percentage(self):
        """Test getting completion percentage."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Test",
            tool="test",
            parameters={},
            status=StepStatus.COMPLETED,
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Test",
            tool="test",
            parameters={},
            status=StepStatus.PENDING,
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1, step2])

        assert plan.get_completion_percentage() == 50.0


class TestPlanExecuteOrchestrator:
    """Test PlanExecuteOrchestrator."""

    def test_orchestrator_creation(self, orchestrator):
        """Test creating orchestrator."""
        assert orchestrator is not None
        assert orchestrator.config.max_plan_steps == 10
        assert len(orchestrator.skills) == 3

    def test_generate_plan(self, orchestrator):
        """Test plan generation."""
        with patch.object(orchestrator, "_call_llm_for_plan") as mock_llm:
            mock_llm.return_value = SkillResult(
                success=True,
                data={
                    "content": json.dumps(
                        {
                            "goal": "Test goal",
                            "steps": [
                                {
                                    "action": "Search",
                                    "tool": "search",
                                    "parameters": {"query": "test"},
                                    "dependencies": [],
                                    "complexity": 1,
                                }
                            ],
                        }
                    )
                },
            )

            plan = orchestrator.generate_plan("Test task", {})

            assert plan is not None
            assert plan.goal == "Test goal"
            assert len(plan.steps) == 1
            assert plan.steps[0].tool == "search"

    def test_validate_plan_valid(self, orchestrator):
        """Test validating a valid plan."""
        steps = [
            PlanStep(
                step_id="step_1",
                step_type=StepType.NORMAL,
                action="Search",
                tool="search",
                parameters={},
                dependencies=[],
            )
        ]

        plan = Plan(plan_id="plan_1", goal="Test", steps=steps)

        result = orchestrator.validate_plan(plan)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_plan_missing_tool(self, orchestrator):
        """Test validating plan with missing tool."""
        steps = [
            PlanStep(
                step_id="step_1",
                step_type=StepType.NORMAL,
                action="Test",
                tool="nonexistent_tool",
                parameters={},
                dependencies=[],
            )
        ]

        plan = Plan(plan_id="plan_1", goal="Test", steps=steps)

        result = orchestrator.validate_plan(plan)

        assert "nonexistent_tool" in result.missing_tools

    def test_detect_circular_dependencies(self, orchestrator):
        """Test detecting circular dependencies."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="First",
            tool="tool1",
            parameters={},
            dependencies=["step_2"],
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Second",
            tool="tool2",
            parameters={},
            dependencies=["step_1"],
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1, step2])

        cycles = orchestrator._detect_circular_dependencies(plan)

        assert len(cycles) > 0

    def test_execute_plan(self, orchestrator):
        """Test executing a plan."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Search",
            tool="search",
            parameters={},
            dependencies=[],
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1])

        result = orchestrator.execute_plan(plan, None)

        assert result.steps_completed == 1
        assert result.steps_failed == 0
        assert plan.is_complete()

    def test_execute_plan_with_dependencies(self, orchestrator):
        """Test executing plan with dependencies."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Search",
            tool="search",
            parameters={},
            dependencies=[],
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Analyze",
            tool="analyze",
            parameters={},
            dependencies=["step_1"],
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1, step2])

        result = orchestrator.execute_plan(plan, None)

        assert result.steps_completed == 2
        assert result.steps_failed == 0
        assert plan.steps[0].status == StepStatus.COMPLETED
        assert plan.steps[1].status == StepStatus.COMPLETED

    def test_execute_conditional_step(self, orchestrator):
        """Test executing conditional step."""
        step = PlanStep(
            step_id="step_1",
            step_type=StepType.CONDITIONAL,
            action="Test",
            tool="search",
            parameters={},
            condition="True",
        )

        orchestrator._execute_step(step, None)

        assert step.status == StepStatus.COMPLETED

    def test_execute_loop_step(self, orchestrator):
        """Test executing loop step."""
        step = PlanStep(
            step_id="step_1",
            step_type=StepType.LOOP,
            action="Test",
            tool="search",
            parameters={},
            loop_count=3,
        )

        orchestrator._execute_step(step, None)

        assert step.status == StepStatus.COMPLETED

    def test_adapt_plan(self, orchestrator):
        """Test adapting a plan."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Search",
            tool="search",
            parameters={},
            dependencies=[],
            status=StepStatus.COMPLETED,
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Analyze",
            tool="analyze",
            parameters={},
            dependencies=["step_1"],
            status=StepStatus.FAILED,
            error="Test error",
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1, step2])

        with (
            patch.object(orchestrator, "_call_llm_for_plan") as mock_llm,
            patch.object(orchestrator, "_parse_plan_response") as mock_parse,
        ):
            mock_parse.return_value = {
                "modifications": [
                    {
                        "action": "modify",
                        "step_id": "step_2",
                        "new_step": {
                            "action": "Retry analyze",
                            "tool": "analyze",
                            "parameters": {"retry": True},
                            "dependencies": ["step_1"],
                        },
                    }
                ]
            }

            adapted = orchestrator.adapt_plan(plan, {})

            assert adapted.plan_id != plan.plan_id
            assert adapted.steps[1].parameters.get("retry") is True

    def test_visualize_plan(self, orchestrator):
        """Test plan visualization."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Search",
            tool="search",
            parameters={},
            status=StepStatus.COMPLETED,
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Analyze",
            tool="analyze",
            parameters={},
            status=StepStatus.PENDING,
        )

        plan = Plan(plan_id="plan_1", goal="Test goal", steps=[step1, step2])

        visualization = orchestrator.visualize_plan(plan)

        assert "Plan: Test goal" in visualization
        assert "✓" in visualization
        assert "○" in visualization
        assert "50.0%" in visualization

    def test_compare_plans(self, orchestrator):
        """Test comparing two plans."""
        plan1 = Plan(
            plan_id="plan_1",
            goal="Test",
            steps=[
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.NORMAL,
                    action="Test",
                    tool="search",
                    parameters={},
                )
            ],
        )

        plan2 = Plan(
            plan_id="plan_2",
            goal="Test",
            steps=[
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.NORMAL,
                    action="Test",
                    tool="search",
                    parameters={},
                ),
                PlanStep(
                    step_id="step_2",
                    step_type=StepType.NORMAL,
                    action="Test2",
                    tool="analyze",
                    parameters={},
                ),
            ],
        )

        comparison = orchestrator.compare_plans(plan1, plan2)

        assert comparison["goal_same"]
        assert comparison["steps_count_diff"] == 1
        assert "step_2" in comparison["added_steps"]

    def test_score_plan(self, orchestrator):
        """Test scoring a plan."""
        step1 = PlanStep(
            step_id="step_1",
            step_type=StepType.NORMAL,
            action="Test",
            tool="search",
            parameters={},
            status=StepStatus.COMPLETED,
        )

        step2 = PlanStep(
            step_id="step_2",
            step_type=StepType.NORMAL,
            action="Test",
            tool="analyze",
            parameters={},
            status=StepStatus.FAILED,
        )

        plan = Plan(plan_id="plan_1", goal="Test", steps=[step1, step2])

        score = orchestrator._score_plan(plan)

        assert score >= 0

    def test_execute_with_tools_fallback(self, orchestrator):
        """Test fallback to sequential execution."""
        with patch.object(orchestrator, "generate_plan") as mock_gen:
            mock_gen.return_value = None

            with patch.object(
                orchestrator.__class__.__bases__[0], "execute_with_tools"
            ) as mock_super:
                mock_super.return_value = MagicMock(success=True, final_response="Fallback")

                result = orchestrator.execute_with_tools("Test task")

                assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
