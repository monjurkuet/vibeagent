"""Plan-and-Execute orchestrator for complex multi-step tasks."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .parallel_executor import ParallelExecutor, ParallelExecutorConfig
from .retry_manager import RetryManager
from .skill import BaseSkill, SkillResult
from .tool_orchestrator import OrchestratorResult, ToolOrchestrator

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a plan step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class StepType(Enum):
    """Type of plan step."""

    NORMAL = "normal"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"
    FALLBACK = "fallback"


@dataclass
class PlanStep:
    """Individual step in a plan."""

    step_id: str
    step_type: StepType
    action: str
    tool: str
    parameters: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    complexity: int = 1
    condition: str | None = None
    loop_count: int | None = None
    fallback_for: str | None = None
    result: SkillResult | None = None
    error: str | None = None
    execution_time_ms: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert step to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "action": self.action,
            "tool": self.tool,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "complexity": self.complexity,
            "condition": self.condition,
            "loop_count": self.loop_count,
            "fallback_for": self.fallback_for,
            "result": {
                "success": self.result.success if self.result else None,
                "data": self.result.data if self.result else None,
                "error": self.result.error if self.result else None,
            }
            if self.result
            else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class Plan:
    """Execution plan with steps and dependencies."""

    plan_id: str
    goal: str
    steps: list[PlanStep]
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "created"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> PlanStep | None:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def add_step(self, step: PlanStep):
        """Add step to plan."""
        self.steps.append(step)
        self.modified_at = datetime.now().isoformat()

    def remove_step(self, step_id: str) -> bool:
        """Remove step from plan."""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                self.steps.pop(i)
                self.modified_at = datetime.now().isoformat()
                return True
        return False

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute."""
        ready_steps = []
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}

        for step in self.steps:
            if step.status == StepStatus.READY:
                ready_steps.append(step)
            elif step.status == StepStatus.PENDING:
                deps_satisfied = all(dep_id in completed_ids for dep_id in step.dependencies)
                if deps_satisfied:
                    step.status = StepStatus.READY
                    ready_steps.append(step)

        return ready_steps

    def get_parallel_ready_steps(self) -> list[list[PlanStep]]:
        """Get groups of steps that can execute in parallel."""
        ready = self.get_ready_steps()
        if not ready:
            return []

        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        parallel_groups = []

        for step in ready:
            if step.step_type == StepType.PARALLEL:
                parallel_groups.append([step])
            else:
                deps = set(step.dependencies)
                can_parallel = True

                for other in ready:
                    if other.step_id != step.step_id:
                        other_deps = set(other.dependencies)
                        if deps & other_deps or other.step_id in deps:
                            can_parallel = False
                            break

                if can_parallel:
                    parallel_groups.append([step])

        return parallel_groups

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] for s in self.steps)

    def is_failed(self) -> bool:
        """Check if plan has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_completion_percentage(self) -> float:
        """Get plan completion percentage."""
        if not self.steps:
            return 0.0

        completed = sum(
            1 for s in self.steps if s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
        )
        return (completed / len(self.steps)) * 100

    def to_dict(self) -> dict:
        """Convert plan to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class PlanValidationResult:
    """Result of plan validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    circular_dependencies: list[list[str]] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    missing_parameters: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PlanExecutionResult:
    """Result of plan execution."""

    success: bool
    plan: Plan
    total_time_ms: float
    steps_completed: int
    steps_failed: int
    steps_skipped: int
    errors: list[str] = field(default_factory=list)
    final_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PlanExecuteOrchestratorConfig:
    """Configuration for Plan-and-Execute orchestrator."""

    def __init__(
        self,
        max_plan_steps: int = 20,
        max_plan_depth: int = 5,
        plan_validation_strictness: str = "moderate",
        adaptation_sensitivity: float = 0.7,
        enable_parallel: bool = True,
        max_parallel_steps: int = 3,
        enable_plan_learning: bool = True,
        fallback_to_sequential: bool = True,
        plan_timeout_ms: int = 300000,
        step_timeout_ms: int = 30000,
    ):
        self.max_plan_steps = max_plan_steps
        self.max_plan_depth = max_plan_depth
        self.plan_validation_strictness = plan_validation_strictness
        self.adaptation_sensitivity = adaptation_sensitivity
        self.enable_parallel = enable_parallel
        self.max_parallel_steps = max_parallel_steps
        self.enable_plan_learning = enable_plan_learning
        self.fallback_to_sequential = fallback_to_sequential
        self.plan_timeout_ms = plan_timeout_ms
        self.step_timeout_ms = step_timeout_ms


class PlanExecuteOrchestrator(ToolOrchestrator):
    """Orchestrator for Plan-and-Execute pattern."""

    def __init__(
        self,
        llm_skill,
        skills: dict[str, BaseSkill],
        db_manager=None,
        config: PlanExecuteOrchestratorConfig | None = None,
    ):
        """Initialize the Plan-and-Execute orchestrator.

        Args:
            llm_skill: LLMSkill instance for interacting with LLM
            skills: Dictionary of skill_name -> BaseSkill for available tools
            db_manager: Optional DatabaseManager for tracking operations
            config: Optional configuration for Plan-and-Execute behavior
        """
        super().__init__(llm_skill, skills, db_manager, use_react=False)

        self.config = config or PlanExecuteOrchestratorConfig()
        self._plan_history: list[Plan] = []
        self._plan_templates: dict[str, Plan] = {}

        self.parallel_executor = ParallelExecutor(
            skills=skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(
                max_parallel_calls=self.config.max_parallel_steps,
                enable_parallel=self.config.enable_parallel,
                track_performance=True,
            ),
        )

        self.retry_manager = RetryManager(db_manager=db_manager)

    def execute_with_tools(
        self, user_message: str, max_iterations: int = 10, use_react=None
    ) -> OrchestratorResult:
        """Execute user message using Plan-and-Execute pattern.

        Args:
            user_message: The user's message to process
            max_iterations: Maximum number of plan adaptation iterations
            use_react: Ignored (Plan-and-Execute always uses planning)

        Returns:
            OrchestratorResult with final response and metrics
        """
        session_id_str = str(uuid.uuid4())
        session_db_id = None
        start_time = time.time()

        context = {
            "user_message": user_message,
            "session_id": session_id_str,
            "available_tools": list(self.skills.keys()),
        }

        if self.db_manager:
            try:
                session_db_id = self.db_manager.create_session(
                    session_id=session_id_str,
                    session_type="plan_execute",
                    model=self.llm_skill.model,
                    orchestrator_type="PlanExecuteOrchestrator",
                    metadata={
                        "max_iterations": max_iterations,
                        "config": self.config.__dict__,
                    },
                )
                self.db_manager.add_message(
                    session_id=session_db_id,
                    role="user",
                    content=user_message,
                    message_index=0,
                    model=self.llm_skill.model,
                )
            except Exception as e:
                logger.error(f"Failed to create session in database: {e}")

        try:
            plan = self.generate_plan(user_message, context)

            if not plan:
                return OrchestratorResult(
                    success=False,
                    final_response="",
                    iterations=0,
                    tool_calls_made=0,
                    tool_results=[],
                    error="Failed to generate plan",
                )

            if self.db_manager and session_db_id:
                self._store_plan(session_db_id, plan)

            validation = self.validate_plan(plan)

            if not validation.is_valid:
                if self.config.fallback_to_sequential:
                    logger.warning(
                        f"Plan validation failed, falling back to sequential: {validation.errors}"
                    )
                    return super().execute_with_tools(user_message, max_iterations)
                return OrchestratorResult(
                    success=False,
                    final_response="",
                    iterations=0,
                    tool_calls_made=0,
                    tool_results=[],
                    error=f"Plan validation failed: {validation.errors}",
                )

            execution_result = self.execute_plan(plan, session_db_id)

            iterations = 0
            while (
                not execution_result.success
                and iterations < max_iterations
                and not plan.is_complete()
            ):
                iterations += 1

                adapted_plan = self.adapt_plan(plan, execution_result.metadata)

                if adapted_plan.plan_id != plan.plan_id:
                    plan = adapted_plan
                    if self.db_manager and session_db_id:
                        self._store_plan(session_db_id, plan)

                    validation = self.validate_plan(plan)
                    if not validation.is_valid:
                        break

                    execution_result = self.execute_plan(plan, session_db_id)
                else:
                    break

            final_response = self._generate_final_response(plan, execution_result)

            if self.db_manager and session_db_id:
                try:
                    self.db_manager.update_session(
                        session_db_id,
                        final_status="completed" if execution_result.success else "failed",
                        total_duration_ms=int((time.time() - start_time) * 1000),
                        total_iterations=iterations + 1,
                        total_tool_calls=len(plan.steps),
                    )
                except Exception as e:
                    logger.error(f"Failed to update session: {e}")

            return OrchestratorResult(
                success=execution_result.success,
                final_response=final_response,
                iterations=iterations + 1,
                tool_calls_made=len(plan.steps),
                tool_results=[
                    {
                        "tool_call": {"function": {"name": s.tool, "arguments": s.parameters}},
                        "result": s.result,
                    }
                    for s in plan.steps
                    if s.result
                ],
                metadata={
                    "plan_id": plan.plan_id,
                    "plan_steps": len(plan.steps),
                    "steps_completed": execution_result.steps_completed,
                    "steps_failed": execution_result.steps_failed,
                    "execution_time_ms": execution_result.total_time_ms,
                },
            )

        except Exception as e:
            logger.error(f"Plan-and-Execute orchestration failed: {e}")

            if self.db_manager and session_db_id:
                try:
                    self.db_manager.update_session(
                        session_db_id,
                        final_status="error",
                        total_duration_ms=int((time.time() - start_time) * 1000),
                        total_iterations=0,
                        total_tool_calls=0,
                    )
                except Exception as db_error:
                    logger.error(f"Failed to update session on error: {db_error}")

            if self.config.fallback_to_sequential:
                logger.info("Falling back to sequential execution")
                return super().execute_with_tools(user_message, max_iterations)

            return OrchestratorResult(
                success=False,
                final_response="",
                iterations=0,
                tool_calls_made=0,
                tool_results=[],
                error=f"Plan-and-Execute failed: {str(e)}",
            )

    def generate_plan(self, user_message: str, context: dict[str, Any]) -> Plan | None:
        """Generate execution plan from user message.

        Args:
            user_message: User's task description
            context: Execution context

        Returns:
            Generated Plan or None if generation fails
        """
        try:
            available_tools = list(self.skills.keys())
            tool_descriptions = self._get_tool_descriptions()

            prompt = self._build_planning_prompt(user_message, tool_descriptions)

            llm_result = self._call_llm_for_plan(prompt)

            if not llm_result.success:
                logger.error(f"LLM plan generation failed: {llm_result.error}")
                return None

            plan_data = self._parse_plan_response(llm_result.data.get("content", ""))

            if not plan_data:
                logger.error("Failed to parse plan response")
                return None

            plan = self._create_plan_from_data(plan_data, user_message, context)

            if len(plan.steps) > self.config.max_plan_steps:
                logger.warning(
                    f"Plan has {len(plan.steps)} steps, exceeding max of {self.config.max_plan_steps}"
                )
                plan.steps = plan.steps[: self.config.max_plan_steps]

            return plan

        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return None

    def _build_planning_prompt(self, user_message: str, tool_descriptions: dict[str, str]) -> str:
        """Build planning prompt for LLM.

        Args:
            user_message: User's task
            tool_descriptions: Available tool descriptions

        Returns:
            Planning prompt string
        """
        prompt = f"""You are an expert task planner. Your job is to break down complex tasks into clear, executable steps.

Available tools:
{json.dumps(tool_descriptions, indent=2)}

Task: {user_message}

Create a detailed execution plan by breaking down the task into steps. For each step, specify:
- action: What the step does
- tool: Which tool to use
- parameters: Required parameters for the tool
- dependencies: Which other steps must complete first (by step ID)
- complexity: Estimated complexity (1-5)

Format your response as a JSON object with this structure:
{{
  "goal": "Brief summary of the overall goal",
  "steps": [
    {{
      "action": "Description of what this step does",
      "tool": "tool_name",
      "parameters": {{"param1": "value1"}},
      "dependencies": [],
      "complexity": 1
    }}
  ]
}}

Guidelines:
1. Start with information gathering steps
2. Break complex tasks into smaller steps
3. Identify dependencies between steps
4. Use appropriate tools for each step
5. Keep steps focused and achievable
6. Number steps sequentially (step_1, step_2, etc.)
"""

        return prompt

    def _call_llm_for_plan(self, prompt: str) -> SkillResult:
        """Call LLM to generate plan.

        Args:
            prompt: Planning prompt

        Returns:
            SkillResult with LLM response
        """
        try:
            import requests

            chat_url = f"{self.llm_skill.base_url}/chat/completions"

            payload = {
                "model": self.llm_skill.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            response = requests.post(chat_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            return SkillResult(
                success=True,
                data={
                    "content": content,
                    "model": self.llm_skill.model,
                    "usage": result.get("usage", {}),
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")

    def _parse_plan_response(self, response: str) -> dict | None:
        """Parse plan response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed plan data or None
        """
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                return None

            json_str = response[json_start:json_end]
            plan_data = json.loads(json_str)

            if "steps" not in plan_data:
                return None

            return plan_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            return None

    def _create_plan_from_data(
        self, plan_data: dict, user_message: str, context: dict[str, Any]
    ) -> Plan:
        """Create Plan object from parsed data.

        Args:
            plan_data: Parsed plan data
            user_message: Original user message
            context: Execution context

        Returns:
            Plan object
        """
        plan_id = str(uuid.uuid4())
        steps = []

        for i, step_data in enumerate(plan_data.get("steps", [])):
            step_id = f"step_{i + 1}"
            step = PlanStep(
                step_id=step_id,
                step_type=StepType.NORMAL,
                action=step_data.get("action", ""),
                tool=step_data.get("tool", ""),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                complexity=step_data.get("complexity", 1),
            )
            steps.append(step)

        plan = Plan(
            plan_id=plan_id,
            goal=plan_data.get("goal", user_message),
            steps=steps,
            context=context,
        )

        return plan

    def _get_tool_descriptions(self) -> dict[str, str]:
        """Get descriptions of available tools.

        Returns:
            Dictionary mapping tool names to descriptions
        """
        descriptions = {}
        for skill_name, skill in self.skills.items():
            try:
                schema = skill.get_tool_schema()
                if schema:
                    descriptions[skill_name] = schema.get("function", {}).get(
                        "description", skill_name
                    )
                else:
                    descriptions[skill_name] = skill_name
            except NotImplementedError:
                descriptions[skill_name] = skill_name

        return descriptions

    def validate_plan(self, plan: Plan) -> PlanValidationResult:
        """Validate plan structure and dependencies.

        Args:
            plan: Plan to validate

        Returns:
            PlanValidationResult with validation details
        """
        result = PlanValidationResult(is_valid=True)

        if not plan.steps:
            result.is_valid = False
            result.errors.append("Plan has no steps")
            return result

        step_ids = {s.step_id for s in plan.steps}

        for step in plan.steps:
            if step.tool not in self.skills:
                result.missing_tools.append(step.tool)

            for dep in step.dependencies:
                if dep not in step_ids:
                    result.missing_parameters.append((step.step_id, dep))

            required_params = self._get_required_params(step.tool)
            for param in required_params:
                if param not in step.parameters:
                    result.missing_parameters.append((step.step_id, param))

        if result.missing_tools:
            result.errors.append(f"Missing tools: {result.missing_tools}")

        if result.missing_parameters:
            result.errors.append(f"Missing parameters: {result.missing_parameters}")

        circular_deps = self._detect_circular_dependencies(plan)
        if circular_deps:
            result.circular_dependencies = circular_deps
            result.errors.append(f"Circular dependencies detected: {circular_deps}")

        result.is_valid = (
            len(result.errors) == 0 or self.config.plan_validation_strictness == "lenient"
        )

        if result.errors and self.config.plan_validation_strictness == "moderate":
            result.warnings.extend(result.errors)
            result.errors = []

        return result

    def _detect_circular_dependencies(self, plan: Plan) -> list[list[str]]:
        """Detect circular dependencies in plan.

        Args:
            plan: Plan to check

        Returns:
            List of circular dependency chains
        """
        graph = {s.step_id: set(s.dependencies) for s in plan.steps}
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: list[str]):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [node])

            rec_stack.remove(node)

        for step in plan.steps:
            dfs(step.step_id, [])

        return cycles

    def _get_required_params(self, tool_name: str) -> list[str]:
        """Get required parameters for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of required parameter names
        """
        skill = self.skills.get(tool_name)
        if not skill:
            return []

        try:
            schema = skill.get_tool_schema()
            if not schema:
                return []

            params_schema = schema.get("function", {}).get("parameters", {})
            return params_schema.get("required", [])
        except (NotImplementedError, AttributeError):
            return []

    def execute_plan(self, plan: Plan, session_id: int | None = None) -> PlanExecutionResult:
        """Execute validated plan.

        Args:
            plan: Plan to execute
            session_id: Optional database session ID

        Returns:
            PlanExecutionResult with execution details
        """
        start_time = time.time()
        errors = []

        try:
            plan.status = "running"

            while not plan.is_complete() and not plan.is_failed():
                ready_steps = plan.get_ready_steps()

                if not ready_steps:
                    if plan.is_failed():
                        break
                    errors.append("No ready steps but plan not complete")
                    break

                if self.config.enable_parallel and len(ready_steps) > 1:
                    parallel_groups = plan.get_parallel_ready_steps()
                    for group in parallel_groups:
                        for step in group:
                            self._execute_step(step, session_id)
                else:
                    for step in ready_steps:
                        self._execute_step(step, session_id)

            plan.status = "completed" if plan.is_complete() else "failed"

            total_time_ms = (time.time() - start_time) * 1000

            steps_completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
            steps_failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)
            steps_skipped = sum(1 for s in plan.steps if s.status == StepStatus.SKIPPED)

            final_response = self._generate_final_response(plan, {})

            return PlanExecutionResult(
                success=plan.is_complete(),
                plan=plan,
                total_time_ms=total_time_ms,
                steps_completed=steps_completed,
                steps_failed=steps_failed,
                steps_skipped=steps_skipped,
                errors=errors,
                final_response=final_response,
            )

        except Exception as e:
            errors.append(f"Plan execution failed: {str(e)}")
            plan.status = "failed"

            return PlanExecutionResult(
                success=False,
                plan=plan,
                total_time_ms=(time.time() - start_time) * 1000,
                steps_completed=0,
                steps_failed=len(plan.steps),
                steps_skipped=0,
                errors=errors,
            )

    def _execute_step(self, step: PlanStep, session_id: int | None = None):
        """Execute a single plan step.

        Args:
            step: Step to execute
            session_id: Optional database session ID
        """
        step.status = StepStatus.RUNNING
        start_time = time.time()

        try:
            if step.step_type == StepType.CONDITIONAL:
                if self._evaluate_condition(step):
                    result = self._execute_tool_step(step)
                else:
                    step.status = StepStatus.SKIPPED
                    return
            elif step.step_type == StepType.LOOP:
                result = self._execute_loop_step(step)
            else:
                result = self._execute_tool_step(step)

            step.result = result
            step.execution_time_ms = (time.time() - start_time) * 1000

            if result.success:
                step.status = StepStatus.COMPLETED
            else:
                step.status = StepStatus.FAILED
                step.error = result.error

                if step.fallback_for:
                    fallback_step = self._find_fallback_step(step)
                    if fallback_step:
                        self._execute_step(fallback_step, session_id)

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.execution_time_ms = (time.time() - start_time) * 1000

    def _execute_tool_step(self, step: PlanStep) -> SkillResult:
        """Execute a tool step.

        Args:
            step: Step to execute

        Returns:
            SkillResult from execution
        """
        skill = self.skills.get(step.tool)
        if not skill:
            return SkillResult(success=False, error=f"Tool '{step.tool}' not found")

        def execute():
            return skill.execute(**step.parameters)

        return self.retry_manager.execute_with_retry(execute, tool_name=step.tool)

    def _execute_loop_step(self, step: PlanStep) -> SkillResult:
        """Execute a loop step.

        Args:
            step: Loop step to execute

        Returns:
            SkillResult from last iteration
        """
        if not step.loop_count or step.loop_count <= 0:
            return SkillResult(success=False, error="Invalid loop count")

        last_result = None
        for i in range(step.loop_count):
            last_result = self._execute_tool_step(step)
            if not last_result.success:
                break

        return (
            last_result
            if last_result
            else SkillResult(success=False, error="Loop execution failed")
        )

    def _evaluate_condition(self, step: PlanStep) -> bool:
        """Evaluate conditional step condition.

        Args:
            step: Conditional step

        Returns:
            True if condition is met
        """
        if not step.condition:
            return True

        try:
            return bool(eval(step.condition, {"__builtins__": {}}, {}))
        except Exception:
            return True

    def _find_fallback_step(self, failed_step: PlanStep) -> PlanStep | None:
        """Find fallback step for a failed step.

        Args:
            failed_step: Step that failed

        Returns:
            Fallback step or None
        """
        return None

    def adapt_plan(self, plan: Plan, execution_context: dict[str, Any]) -> Plan:
        """Adapt plan based on execution results.

        Args:
            plan: Current plan
            execution_context: Context from previous execution

        Returns:
            Adapted plan (may be same if no changes needed)
        """
        if not plan.is_failed():
            return plan

        failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]

        if not failed_steps:
            return plan

        adaptation_prompt = self._build_adaptation_prompt(plan, failed_steps)

        llm_result = self._call_llm_for_plan(adaptation_prompt)

        if not llm_result.success:
            return plan

        adaptation_data = self._parse_plan_response(llm_result.data.get("content", ""))

        if not adaptation_data:
            return plan

        adapted_plan = self._create_adapted_plan(plan, adaptation_data)

        if adapted_plan.plan_id != plan.plan_id:
            if self.config.enable_plan_learning:
                self._learn_from_plan(plan, adapted_plan)

        return adapted_plan

    def _build_adaptation_prompt(self, plan: Plan, failed_steps: list[PlanStep]) -> str:
        """Build adaptation prompt for LLM.

        Args:
            plan: Current plan
            failed_steps: Steps that failed

        Returns:
            Adaptation prompt string
        """
        failed_info = [
            {
                "step_id": s.step_id,
                "tool": s.tool,
                "error": s.error,
                "parameters": s.parameters,
            }
            for s in failed_steps
        ]

        prompt = f"""The following plan has failed. Please suggest modifications to fix the issues.

Original goal: {plan.goal}

Failed steps:
{json.dumps(failed_info, indent=2)}

Completed steps:
{json.dumps([s.step_id for s in plan.steps if s.status == StepStatus.COMPLETED], indent=2)}

Suggest modifications as a JSON object:
{{
  "modifications": [
    {{
      "action": "add|remove|modify",
      "step_id": "step_id (for modify/remove)",
      "new_step": {{"action": "...", "tool": "...", "parameters": {{}}, "dependencies": []}} (for add/modify)
    }}
  ]
}}

Be conservative - only suggest essential changes.
"""

        return prompt

    def _create_adapted_plan(self, original_plan: Plan, adaptation_data: dict) -> Plan:
        """Create adapted plan from modifications.

        Args:
            original_plan: Original plan
            adaptation_data: Adaptation modifications

        Returns:
            New adapted plan
        """
        import copy

        adapted_plan = copy.deepcopy(original_plan)
        adapted_plan.plan_id = str(uuid.uuid4())
        adapted_plan.modified_at = datetime.now().isoformat()

        modifications = adaptation_data.get("modifications", [])

        for mod in modifications:
            action = mod.get("action")

            if action == "remove":
                step_id = mod.get("step_id")
                adapted_plan.remove_step(step_id)

            elif action == "add":
                new_step_data = mod.get("new_step", {})
                step_id = f"step_{len(adapted_plan.steps) + 1}"
                new_step = PlanStep(
                    step_id=step_id,
                    step_type=StepType.NORMAL,
                    action=new_step_data.get("action", ""),
                    tool=new_step_data.get("tool", ""),
                    parameters=new_step_data.get("parameters", {}),
                    dependencies=new_step_data.get("dependencies", []),
                    complexity=1,
                )
                adapted_plan.add_step(new_step)

            elif action == "modify":
                step_id = mod.get("step_id")
                step = adapted_plan.get_step(step_id)
                if step:
                    new_step_data = mod.get("new_step", {})
                    step.action = new_step_data.get("action", step.action)
                    step.tool = new_step_data.get("tool", step.tool)
                    step.parameters.update(new_step_data.get("parameters", {}))
                    step.dependencies = new_step_data.get("dependencies", step.dependencies)

        return adapted_plan

    def _learn_from_plan(self, original_plan: Plan, adapted_plan: Plan):
        """Learn from plan adaptation for future use.

        Args:
            original_plan: Original plan that failed
            adapted_plan: Adapted plan that succeeded
        """
        pattern_key = self._extract_plan_pattern(original_plan)

        if pattern_key and adapted_plan.is_complete():
            self._plan_templates[pattern_key] = adapted_plan

            if len(self._plan_templates) > 100:
                self._plan_templates.popitem()

    def _extract_plan_pattern(self, plan: Plan) -> str | None:
        """Extract pattern key from plan.

        Args:
            plan: Plan to extract pattern from

        Returns:
            Pattern key string
        """
        tool_sequence = "-".join([s.tool for s in plan.steps])
        return f"{tool_sequence}"

    def _generate_final_response(self, plan: Plan, execution_result: dict[str, Any]) -> str:
        """Generate final response from plan execution.

        Args:
            plan: Executed plan
            execution_result: Execution result metadata

        Returns:
            Final response string
        """
        if plan.is_complete():
            response = f"Task completed successfully: {plan.goal}\n\n"

            completed_steps = [s for s in plan.steps if s.status == StepStatus.COMPLETED]
            if completed_steps:
                response += f"Executed {len(completed_steps)} steps:\n"
                for step in completed_steps:
                    response += f"  - {step.action}\n"

            return response
        failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
        error_msg = f"Task failed: {plan.goal}\n\n"

        if failed_steps:
            error_msg += "Failed steps:\n"
            for step in failed_steps:
                error_msg += f"  - {step.action}: {step.error}\n"

        return error_msg

    def _store_plan(self, session_id: int, plan: Plan):
        """Store plan in database.

        Args:
            session_id: Database session ID
            plan: Plan to store
        """
        if not self.db_manager:
            return

        try:
            plan_json = json.dumps(plan.to_dict())
            self.db_manager.add_message(
                session_id=session_id,
                role="system",
                content=f"Plan: {plan_json}",
                message_index=-1,
                metadata={"plan_id": plan.plan_id, "plan_goal": plan.goal},
            )
        except Exception as e:
            logger.error(f"Failed to store plan: {e}")

    def visualize_plan(self, plan: Plan) -> str:
        """Generate plan visualization.

        Args:
            plan: Plan to visualize

        Returns:
            Visualization string
        """
        lines = [f"Plan: {plan.goal}", f"Status: {plan.status}", "Steps:"]

        for step in plan.steps:
            status_symbol = {
                StepStatus.PENDING: "○",
                StepStatus.READY: "→",
                StepStatus.RUNNING: "▶",
                StepStatus.COMPLETED: "✓",
                StepStatus.FAILED: "✗",
                StepStatus.SKIPPED: "⊘",
                StepStatus.BLOCKED: "⊘",
            }.get(step.status, "?")

            deps_str = f" (after: {', '.join(step.dependencies)})" if step.dependencies else ""
            lines.append(
                f"  {status_symbol} [{step.step_id}] {step.action} ({step.tool}){deps_str}"
            )

        completion = plan.get_completion_percentage()
        lines.append(f"\nProgress: {completion:.1f}%")

        return "\n".join(lines)

    def compare_plans(self, plan1: Plan, plan2: Plan) -> dict[str, Any]:
        """Compare two plans.

        Args:
            plan1: First plan
            plan2: Second plan

        Returns:
            Comparison results
        """
        diff = {
            "steps_count_diff": len(plan2.steps) - len(plan1.steps),
            "goal_same": plan1.goal == plan2.goal,
            "added_steps": [],
            "removed_steps": [],
            "modified_steps": [],
        }

        plan1_steps = {s.step_id: s for s in plan1.steps}
        plan2_steps = {s.step_id: s for s in plan2.steps}

        for step_id, step in plan2_steps.items():
            if step_id not in plan1_steps:
                diff["added_steps"].append(step_id)
            elif (
                step.tool != plan1_steps[step_id].tool
                or step.parameters != plan1_steps[step_id].parameters
            ):
                diff["modified_steps"].append(step_id)

        for step_id in plan1_steps:
            if step_id not in plan2_steps:
                diff["removed_steps"].append(step_id)

        diff["score1"] = self._score_plan(plan1)
        diff["score2"] = self._score_plan(plan2)
        diff["better_plan"] = "plan1" if diff["score1"] > diff["score2"] else "plan2"

        return diff

    def _score_plan(self, plan: Plan) -> float:
        """Score plan quality.

        Args:
            plan: Plan to score

        Returns:
            Quality score
        """
        score = 0.0

        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                score += 1.0
            elif step.status == StepStatus.FAILED:
                score -= 0.5

        score -= len(plan.steps) * 0.1

        cycles = self._detect_circular_dependencies(plan)
        score -= len(cycles) * 2.0

        return max(score, 0.0)
