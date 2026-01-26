from .skill import BaseSkill, SkillResult
from .retry_manager import RetryManager
from .parallel_executor import ParallelExecutor, ParallelExecutorConfig, ParallelExecutionResult
from .context_manager import ContextManager

"""Tool orchestrator for managing LLM tool calling loop."""

import json
import uuid
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .skill import BaseSkill, SkillResult
from .retry_manager import RetryManager
from .parallel_executor import (
    ParallelExecutor,
    ParallelExecutorConfig,
    ParallelExecutionResult,
)

logger = logging.getLogger(__name__)


REACT_CONFIG = {
    "max_reasoning_steps": 20,
    "reflection_frequency": 3,
    "plan_revision_threshold": 2,
    "error_retry_threshold": 3,
    "confidence_threshold": 0.7,
}


@dataclass
class OrchestratorResult:
    """Result from tool orchestration."""

    success: bool
    final_response: str
    iterations: int
    tool_calls_made: int
    tool_results: List[Dict]
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    reasoning_trace: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.reasoning_trace is None:
            self.reasoning_trace = []


class ToolOrchestrator:
    """Orchestrator for managing LLM tool calling execution loop."""

    def __init__(
        self,
        llm_skill,
        skills: Dict[str, BaseSkill],
        db_manager=None,
        use_react=False,
        react_config=None,
    ):
        """Initialize the tool orchestrator.

        Args:
            llm_skill: LLMSkill instance for interacting with LLM
            skills: Dictionary of skill_name -> BaseSkill for available tools
            db_manager: Optional DatabaseManager for tracking operations
            use_react: Whether to use ReAct (Reasoning + Acting) pattern
            react_config: Optional configuration for ReAct behavior
        """
        self.llm_skill = llm_skill
        self.skills = skills
        self.db_manager = db_manager
        self.use_react = use_react
        self.react_config = {**REACT_CONFIG, **(react_config or {})}
        self._tool_schemas = self._build_tool_schemas()
        self._error_count = {}
        self._consecutive_errors = 0
        self.retry_manager = RetryManager(db_manager=db_manager)
        self.parallel_executor = ParallelExecutor(
            skills=skills,
            db_manager=db_manager,
            config=ParallelExecutorConfig(
                max_parallel_calls=5,
                enable_parallel=True,
                track_performance=True,
            ),
        )

    def _build_tool_schemas(self) -> List[Dict]:
        """Build tool schemas from registered skills.

        Returns:
            List of tool schemas in OpenAI function format
        """
        tool_schemas = []
        for skill_name, skill in self.skills.items():
            try:
                schema = skill.get_tool_schema()
                if schema:
                    tool_schemas.append(schema)
            except NotImplementedError:
                continue
        return tool_schemas

    def _get_model_type(self) -> str:
        """Determine model type for prompt selection.

        Returns:
            Model type string (gpt4, claude, local_llm, default)
        """
        model_name = self.llm_skill.model.lower()
        if "gpt-4" in model_name:
            return "gpt4"
        elif "claude" in model_name:
            return "claude"
        elif any(x in model_name for x in ["llama", "mistral", "phi", "gemma"]):
            return "local_llm"
        return "default"

    def _build_react_messages(self, messages: List[Dict]) -> List[Dict]:
        """Build complete ReAct prompt with system message and examples.

        Args:
            messages: Original conversation messages

        Returns:
            Complete message list with ReAct system prompt
        """
        from prompts.react_prompt import build_react_prompt

        task_type = self._determine_task_type(messages)
        example_categories = self._get_example_categories(task_type)

        return build_react_prompt(
            messages=messages,
            tools=self._tool_schemas,
            model_type=self._get_model_type(),
            include_examples=True,
            example_categories=example_categories,
        )

    def _determine_task_type(self, messages: List[Dict]) -> str:
        """Determine task type from user message.

        Args:
            messages: Conversation messages

        Returns:
            Task type string (simple, chaining, error_recovery, parallel, complex)
        """
        if not messages:
            return "simple"

        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "").lower()
                break

        if any(
            word in user_content
            for word in ["and", "then", "after", "first", "next", "finally"]
        ):
            if any(
                word in user_content
                for word in ["simultaneously", "parallel", "together"]
            ):
                return "parallel"
            return "chaining"

        if any(word in user_content for word in ["error", "fail", "fix", "correct"]):
            return "error_recovery"

        if len(user_content.split()) > 20:
            return "complex"

        return "simple"

    def _get_example_categories(self, task_type: str) -> Optional[List[str]]:
        """Get example categories based on task type.

        Args:
            task_type: Type of task

        Returns:
            List of example categories or None for all
        """
        return [task_type]

    def execute_with_tools(
        self, user_message: str, max_iterations: int = 10, use_react=None
    ) -> OrchestratorResult:
        """Execute user message with tool calling loop.

        Args:
            user_message: The user's message to process
            max_iterations: Maximum number of tool calling iterations
            use_react: Override ReAct mode (None=use instance setting)

        Returns:
            OrchestratorResult with final response and metrics
        """
        session_id_str = str(uuid.uuid4())
        session_db_id = None
        start_time = time.time()

        enable_react = use_react if use_react is not None else self.use_react
        self._error_count = {}
        self._consecutive_errors = 0

        messages = []
        messages.append({"role": "user", "content": user_message})

        iterations = 0
        tool_calls_made = 0
        tool_results = []
        reasoning_trace = []

        if self.db_manager:
            try:
                session_db_id = self.db_manager.create_session(
                    session_id=session_id_str,
                    session_type="tool_orchestration",
                    model=self.llm_skill.model,
                    orchestrator_type="ToolOrchestrator",
                    metadata={
                        "max_iterations": max_iterations,
                        "use_react": enable_react,
                        "react_config": self.react_config if enable_react else None,
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
            while iterations < max_iterations:
                iterations += 1

                llm_start = time.time()

                if enable_react:
                    llm_result = self._call_llm_with_react(messages)
                else:
                    llm_result = self._call_llm_with_tools(messages)

                llm_time = (time.time() - llm_start) * 1000

                if not llm_result.success:
                    if self.db_manager and session_db_id:
                        try:
                            self._track_llm_response(
                                session_db_id,
                                llm_result,
                                llm_time,
                                success=False,
                            )
                            self.db_manager.update_session(
                                session_db_id,
                                final_status="failed",
                                total_duration_ms=int(
                                    (time.time() - start_time) * 1000
                                ),
                                total_iterations=iterations,
                                total_tool_calls=tool_calls_made,
                            )
                        except Exception as e:
                            logger.error(f"Failed to update session: {e}")

                assistant_message = llm_result.data.get("message", {})
                messages.append(assistant_message)

                if enable_react:
                    reasoning = self._extract_reasoning(assistant_message)
                    if reasoning:
                        reasoning_trace.append(
                            {
                                "iteration": iterations,
                                "reasoning": reasoning,
                            }
                        )
                        if session_db_id:
                            self._track_reasoning_steps(
                                session_db_id,
                                iterations,
                                reasoning,
                            )

                if self.db_manager and session_db_id:
                    try:
                        self._track_llm_response(
                            session_db_id,
                            llm_result,
                            llm_time,
                            success=True,
                        )
                    except Exception as e:
                        logger.error(f"Failed to track LLM response: {e}")

                tool_calls = self.parse_tool_calls(assistant_message)

                if not tool_calls:
                    final_content = assistant_message.get("content", "")

                    if enable_react and "Final Answer:" in final_content:
                        final_content = final_content.split("Final Answer:")[-1].strip()

                    if self.db_manager and session_db_id:
                        try:
                            self.db_manager.update_session(
                                session_db_id,
                                final_status="completed",
                                total_duration_ms=int(
                                    (time.time() - start_time) * 1000
                                ),
                                total_iterations=iterations,
                                total_tool_calls=tool_calls_made,
                            )
                        except Exception as e:
                            logger.error(f"Failed to update session: {e}")

                    return OrchestratorResult(
                        success=True,
                        final_response=final_content,
                        iterations=iterations,
                        tool_calls_made=tool_calls_made,
                        tool_results=tool_results,
                        reasoning_trace=reasoning_trace,
                    )

                if len(tool_calls) > 1:
                    parallel_results = self._execute_tools_parallel(
                        tool_calls, session_db_id
                    )

                    for i, (tool_call, result) in enumerate(
                        zip(tool_calls, parallel_results)
                    ):
                        tool_calls_made += 1
                        tool_result = SkillResult(
                            success=result.get("success", False),
                            data=result.get("data"),
                            error=result.get("error"),
                        )

                        if self.db_manager and session_db_id:
                            try:
                                self._track_tool_call(
                                    session_db_id,
                                    tool_calls_made,
                                    tool_call,
                                    tool_result,
                                )
                            except Exception as e:
                                logger.error(f"Failed to track tool call: {e}")

                        tool_results.append(
                            {
                                "tool_call": tool_call,
                                "result": tool_result,
                            }
                        )

                        observation_text = self._format_observation(tool_result)
                        messages.append(
                            {
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "content": json.dumps(
                                    {
                                        "success": tool_result.success,
                                        "data": tool_result.data,
                                        "error": tool_result.error,
                                    }
                                ),
                            }
                        )

                        if enable_react:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": f"Observation: {observation_text}",
                                }
                            )

                            if not tool_result.success:
                                self._consecutive_errors += 1
                                tool_name = tool_call.get("function", {}).get(
                                    "name", "unknown"
                                )
                                self._error_count[tool_name] = (
                                    self._error_count.get(tool_name, 0) + 1
                                )

                                if self._should_revise_plan(iterations, tool_result):
                                    reflection = self._reflect_on_observation(
                                        tool_result, messages
                                    )
                                    if reflection:
                                        messages.append(
                                            {
                                                "role": "system",
                                                "content": f"Reflection: {reflection}",
                                            }
                                        )
                                        reasoning_trace.append(
                                            {
                                                "iteration": iterations,
                                                "reflection": reflection,
                                            }
                                        )
                            else:
                                self._consecutive_errors = 0
                else:
                    for tool_call in tool_calls:
                        tool_calls_made += 1
                        tool_result = self._execute_tool(tool_call)

                        if self.db_manager and session_db_id:
                            try:
                                self._track_tool_call(
                                    session_db_id,
                                    tool_calls_made,
                                    tool_call,
                                    tool_result,
                                )
                            except Exception as e:
                                logger.error(f"Failed to track tool call: {e}")

                        tool_results.append(
                            {
                                "tool_call": tool_call,
                                "result": tool_result,
                            }
                        )

                        observation_text = self._format_observation(tool_result)
                        messages.append(
                            {
                                "tool_call_id": tool_call.get("id"),
                                "role": "tool",
                                "content": json.dumps(
                                    {
                                        "success": tool_result.success,
                                        "data": tool_result.data,
                                        "error": tool_result.error,
                                    }
                                ),
                            }
                        )

                        if enable_react:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": f"Observation: {observation_text}",
                                }
                            )

                            if not tool_result.success:
                                self._consecutive_errors += 1
                                tool_name = tool_call.get("function", {}).get(
                                    "name", "unknown"
                                )
                                self._error_count[tool_name] = (
                                    self._error_count.get(tool_name, 0) + 1
                                )

                                if self._should_revise_plan(iterations, tool_result):
                                    reflection = self._reflect_on_observation(
                                        tool_result, messages
                                    )
                                    if reflection:
                                        messages.append(
                                            {
                                                "role": "system",
                                                "content": f"Reflection: {reflection}",
                                            }
                                        )
                                        reasoning_trace.append(
                                            {
                                                "iteration": iterations,
                                                "reflection": reflection,
                                            }
                                        )
                            else:
                                self._consecutive_errors = 0

            if self.db_manager and session_db_id:
                try:
                    self.db_manager.update_session(
                        session_db_id,
                        final_status="max_iterations",
                        total_duration_ms=int((time.time() - start_time) * 1000),
                        total_iterations=iterations,
                        total_tool_calls=tool_calls_made,
                    )
                except Exception as e:
                    logger.error(f"Failed to update session: {e}")

            return OrchestratorResult(
                success=False,
                final_response="",
                iterations=iterations,
                tool_calls_made=tool_calls_made,
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                error=f"Max iterations ({max_iterations}) reached without completion",
            )

        except Exception as e:
            if self.db_manager and session_db_id:
                try:
                    self.db_manager.update_session(
                        session_db_id,
                        final_status="error",
                        total_duration_ms=int((time.time() - start_time) * 1000),
                        total_iterations=iterations,
                        total_tool_calls=tool_calls_made,
                    )
                except Exception as db_error:
                    logger.error(f"Failed to update session on error: {db_error}")

            return OrchestratorResult(
                success=False,
                final_response="",
                iterations=iterations,
                tool_calls_made=tool_calls_made,
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                error=f"Orchestration failed: {str(e)}",
            )

    def _call_llm_with_tools(self, messages: List[Dict]) -> SkillResult:
        """Call LLM with tool definitions.

        Args:
            messages: Conversation messages so far

        Returns:
            SkillResult with LLM response
        """
        try:
            import requests

            chat_url = f"{self.llm_skill.base_url}/chat/completions"

            payload = {
                "model": self.llm_skill.model,
                "messages": messages,
                "tools": self._tool_schemas,
                "tool_choice": "auto",
                "temperature": 0.7,
                "max_tokens": 2000,
            }

            response = requests.post(chat_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]

            return SkillResult(
                success=True,
                data={
                    "message": message,
                    "model": self.llm_skill.model,
                    "usage": result.get("usage", {}),
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")

    def parse_tool_calls(self, response: Dict) -> List[Dict]:
        """Extract tool calls from LLM response.

        Args:
            response: LLM response message

        Returns:
            List of tool call dictionaries
        """
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            return []

        parsed_calls = []
        for tool_call in tool_calls:
            parsed = {
                "id": tool_call.get("id"),
                "type": tool_call.get("type", "function"),
                "function": tool_call.get("function", {}),
            }
            parsed_calls.append(parsed)

        return parsed_calls

    def _execute_tool(self, tool_call: Dict) -> SkillResult:
        """Execute a single tool call with retry logic.

        Args:
            tool_call: Tool call dictionary with function info

        Returns:
            SkillResult from tool execution
        """
        try:
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name")
            arguments_str = function_info.get("arguments", "{}")

            if not function_name:
                return SkillResult(
                    success=False, error="Tool call missing function name"
                )

            skill = self.skills.get(function_name)
            if not skill:
                return SkillResult(
                    success=False, error=f"Tool '{function_name}' not found"
                )

            arguments = json.loads(arguments_str)

            def execute_tool():
                return skill.execute(**arguments)

            result = self.retry_manager.execute_with_retry(
                execute_tool,
                tool_name=function_name,
            )

            return result

        except json.JSONDecodeError as e:
            return SkillResult(
                success=False, error=f"Failed to parse tool arguments: {str(e)}"
            )
        except Exception as e:
            return SkillResult(success=False, error=f"Tool execution failed: {str(e)}")

    def _track_llm_response(
        self,
        session_id: int,
        llm_result: SkillResult,
        response_time_ms: float,
        success: bool,
    ):
        """Track LLM response in database.

        Args:
            session_id: Database session ID
            llm_result: LLM result from API call
            response_time_ms: Response time in milliseconds
            success: Whether the LLM call was successful
        """
        if not self.db_manager:
            return

        try:
            if success and llm_result.data:
                usage = llm_result.data.get("usage", {})
                self.db_manager.add_llm_response(
                    session_id=session_id,
                    message_id=None,
                    model=self.llm_skill.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    response_time_ms=int(response_time_ms),
                    finish_reason=None,
                    raw_response=llm_result.data.get("message"),
                    reasoning_content=None,
                    tool_calls_count=0,
                )
        except Exception as e:
            logger.error(f"Failed to track LLM response: {e}")

    def _track_tool_call(
        self,
        session_id: int,
        call_index: int,
        tool_call: Dict,
        tool_result: SkillResult,
    ):
        """Track tool call in database.

        Args:
            session_id: Database session ID
            call_index: Index of the tool call in session
            tool_call: Tool call dictionary
            tool_result: Result from tool execution
        """
        if not self.db_manager:
            return

        try:
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name")
            arguments_str = function_info.get("arguments", "{}")
            arguments = json.loads(arguments_str)

            tool_call_id = self.db_manager.add_tool_call(
                session_id=session_id,
                call_index=call_index,
                tool_name=function_name,
                parameters=arguments,
                execution_time_ms=0,
                success=tool_result.success,
                error_message=tool_result.error if not tool_result.success else None,
                error_type=None,
                retry_count=0,
                is_parallel=False,
                parallel_batch_id=None,
                metadata={"tool_call_id": tool_call.get("id")},
            )

            self.db_manager.add_tool_result(
                tool_call_id=tool_call_id,
                success=tool_result.success,
                data=tool_result.data if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
                result_size_bytes=(
                    len(json.dumps(tool_result.data)) if tool_result.data else 0
                ),
                metadata=None,
            )
        except Exception as e:
            logger.error(f"Failed to track tool call: {e}")

    def track_reasoning_step(
        self,
        session_id: int,
        iteration: int,
        step_type: str,
        content: str,
        tool_call_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[int]:
        """Track a reasoning step in database.

        Args:
            session_id: Database session ID
            iteration: Iteration number
            step_type: Type of step (thought, action, observation)
            content: Content of the reasoning step
            tool_call_id: Optional linked tool call ID
            metadata: Additional metadata

        Returns:
            Database ID of the reasoning step or None if failed
        """
        if not self.db_manager:
            return None

        try:
            return self.db_manager.add_reasoning_step(
                session_id=session_id,
                iteration=iteration,
                step_type=step_type,
                content=content,
                tool_call_id=tool_call_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to track reasoning step: {e}")
            return None

    def _call_llm_with_react(self, messages: List[Dict]) -> SkillResult:
        """Call LLM with ReAct prompt.

        Args:
            messages: Conversation messages so far

        Returns:
            SkillResult with LLM response
        """
        try:
            import requests

            chat_url = f"{self.llm_skill.base_url}/chat/completions"
            react_messages = self._build_react_messages(messages)

            payload = {
                "model": self.llm_skill.model,
                "messages": react_messages,
                "tools": self._tool_schemas,
                "tool_choice": "auto",
                "temperature": 0.7,
                "max_tokens": 3000,
            }

            response = requests.post(chat_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]

            reasoning_content = self._extract_reasoning_content(message)

            return SkillResult(
                success=True,
                data={
                    "message": message,
                    "model": self.llm_skill.model,
                    "usage": result.get("usage", {}),
                    "reasoning_content": reasoning_content,
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")

    def _extract_reasoning(self, message: Dict) -> Optional[str]:
        """Extract reasoning from LLM message.

        Args:
            message: LLM message

        Returns:
            Extracted reasoning or None
        """
        content = message.get("content", "")
        if "Thought:" in content:
            match = re.search(
                r"Thought: (.*?)(?=Action:|Observation:|Final Answer:|$)",
                content,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()
        return None

    def _extract_reasoning_content(self, message: Dict) -> Optional[str]:
        """Extract reasoning content from LLM message.

        Args:
            message: LLM response message

        Returns:
            Reasoning content string or None
        """
        content = message.get("content", "")

        if not content:
            return None

        thought_pattern = r"Thought:\s*(.*?)(?=\n(?:Action|Observation|Final Answer)|$)"
        thoughts = re.findall(thought_pattern, content, re.DOTALL | re.IGNORECASE)

        if thoughts:
            return "\n".join(thought.strip() for thought in thoughts)

        return content[:1000] if len(content) > 1000 else content

    def _track_reasoning_steps(self, session_id: int, iteration: int, reasoning: str):
        """Track reasoning steps in database.

        Args:
            session_id: Database session ID
            iteration: Iteration number
            reasoning: Reasoning content
        """
        if not self.db_manager:
            return

        try:
            self.track_reasoning_step(
                session_id=session_id,
                iteration=iteration,
                step_type="thought",
                content=reasoning,
            )
        except Exception as e:
            logger.error(f"Failed to track reasoning steps: {e}")

    def _format_observation(self, tool_result: SkillResult) -> str:
        """Format tool result as observation.

        Args:
            tool_result: Result from tool execution

        Returns:
            Formatted observation string
        """
        if tool_result.success:
            return f"Success: {json.dumps(tool_result.data, default=str)[:200]}"
        else:
            return f"Error: {tool_result.error}"

    def _should_revise_plan(self, iteration: int, tool_result: SkillResult) -> bool:
        """Determine if plan should be revised based on errors.

        Args:
            iteration: Current iteration number
            tool_result: Result from tool execution

        Returns:
            True if plan should be revised
        """
        if not tool_result.success:
            error_threshold = self.react_config.get("error_retry_threshold", 3)
            if self._consecutive_errors >= error_threshold:
                return True

            reflection_frequency = self.react_config.get("reflection_frequency", 3)
            if iteration % reflection_frequency == 0:
                return True

        return False

    def _reflect_on_observation(
        self, tool_result: SkillResult, messages: List[Dict]
    ) -> Optional[str]:
        """Generate reflection on observation.

        Args:
            tool_result: Result from tool execution
            messages: Conversation messages

        Returns:
            Reflection string or None
        """
        if not tool_result.success:
            return f"Encountered error: {tool_result.error}. Reconsidering approach."
        return None

    def _execute_tools_parallel(
        self, tool_calls: List[Dict], session_id: Optional[int] = None
    ) -> List[Dict]:
        """Execute tools in parallel where safe.

        Args:
            tool_calls: List of tool calls
            session_id: Optional database session ID

        Returns:
            List of results
        """
        try:
            parallel_result = self.parallel_executor.execute_parallel(
                tool_calls, session_id
            )
            return parallel_result.results
        except Exception as e:
            logger.warning(f"Parallel execution failed, falling back: {e}")
            results = []
            for tc in tool_calls:
                skill_result = self._execute_tool(tc)
                results.append(
                    {
                        "success": skill_result.success,
                        "data": skill_result.data,
                        "error": skill_result.error,
                        "tool_call": tc,
                    }
                )
            return results
