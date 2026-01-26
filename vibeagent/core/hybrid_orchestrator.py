"""Hybrid orchestrator combining tool calling and prompt-based approaches."""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .skill import BaseSkill, SkillResult
from .tool_orchestrator import ToolOrchestrator, OrchestratorResult

logger = logging.getLogger(__name__)


@dataclass
class HybridOrchestratorResult:
    """Result from hybrid orchestration."""

    success: bool
    final_response: str
    method_used: str
    iterations: int
    tool_calls_made: int
    tool_results: List[Dict]
    error: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridOrchestrator:
    """Orchestrator that tries tool calling first, falls back to prompt-based approach."""

    def __init__(self, llm_skill, skills: Dict[str, BaseSkill]):
        """Initialize the hybrid orchestrator.

        Args:
            llm_skill: LLMSkill instance for interacting with LLM
            skills: Dictionary of skill_name -> BaseSkill for available tools
        """
        self.llm_skill = llm_skill
        self.skills = skills
        self.tool_orchestrator = ToolOrchestrator(llm_skill, skills)
        logger.info(f"HybridOrchestrator initialized with {len(skills)} skills")

    def execute(
        self, user_message: str, max_iterations: int = 10
    ) -> HybridOrchestratorResult:
        """Execute user message with hybrid approach.

        Args:
            user_message: The user's message to process
            max_iterations: Maximum number of iterations for tool calling

        Returns:
            HybridOrchestratorResult with final response and metrics
        """
        logger.info(
            f"Executing user message with hybrid approach: {user_message[:100]}..."
        )

        try:
            tool_result = self.tool_orchestrator.execute_with_tools(
                user_message, max_iterations
            )

            if tool_result.success and tool_result.final_response:
                logger.info("Tool calling succeeded, returning result")
                return HybridOrchestratorResult(
                    success=tool_result.success,
                    final_response=tool_result.final_response,
                    method_used="tool_calling",
                    iterations=tool_result.iterations,
                    tool_calls_made=tool_result.tool_calls_made,
                    tool_results=tool_result.tool_results,
                    error=tool_result.error,
                    metadata=tool_result.metadata,
                )

            logger.info(
                "Tool calling failed or returned no results, falling back to prompt-based"
            )
            return self._execute_prompt_based(user_message)

        except Exception as e:
            logger.error(f"Error in hybrid execution: {str(e)}", exc_info=True)
            try:
                logger.info("Attempting prompt-based fallback after error")
                return self._execute_prompt_based(user_message)
            except Exception as fallback_error:
                return HybridOrchestratorResult(
                    success=False,
                    final_response="",
                    method_used="failed",
                    iterations=0,
                    tool_calls_made=0,
                    tool_results=[],
                    error=f"Both tool calling and prompt-based failed: {str(e)} | Fallback error: {str(fallback_error)}",
                )

    def _execute_prompt_based(self, user_message: str) -> HybridOrchestratorResult:
        """Execute user message using prompt-based approach.

        Args:
            user_message: The user's message to process

        Returns:
            HybridOrchestratorResult with final response and metrics
        """
        logger.info("Executing prompt-based approach")

        try:
            task_plan = self._generate_task_plan(user_message)

            if not task_plan:
                return HybridOrchestratorResult(
                    success=False,
                    final_response="",
                    method_used="prompt_based",
                    iterations=0,
                    tool_calls_made=0,
                    tool_results=[],
                    error="Failed to generate task plan from LLM",
                )

            execution_results = []
            tool_calls_made = 0

            for task in task_plan:
                skill_name = task.get("skill")
                arguments = task.get("arguments", {})

                if not skill_name:
                    logger.warning(f"Task missing skill name: {task}")
                    continue

                skill = self.skills.get(skill_name)
                if not skill:
                    logger.warning(f"Skill not found: {skill_name}")
                    execution_results.append(
                        {"task": task, "result": f"Skill '{skill_name}' not found"}
                    )
                    continue

                try:
                    result = skill.execute(**arguments)
                    tool_calls_made += 1
                    execution_results.append(
                        {
                            "task": task,
                            "result": result.data if result.success else result.error,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error executing task {skill_name}: {str(e)}")
                    execution_results.append(
                        {"task": task, "result": f"Error: {str(e)}"}
                    )

            final_response = self._generate_final_response(
                user_message, execution_results
            )

            return HybridOrchestratorResult(
                success=True,
                final_response=final_response,
                method_used="prompt_based",
                iterations=1,
                tool_calls_made=tool_calls_made,
                tool_results=execution_results,
                metadata={"task_plan": task_plan},
            )

        except Exception as e:
            logger.error(f"Error in prompt-based execution: {str(e)}", exc_info=True)
            return HybridOrchestratorResult(
                success=False,
                final_response="",
                method_used="prompt_based",
                iterations=0,
                tool_calls_made=0,
                tool_results=[],
                error=f"Prompt-based execution failed: {str(e)}",
            )

    def _generate_task_plan(self, user_message: str) -> Optional[List[Dict]]:
        """Generate a task plan from the user message using LLM.

        Args:
            user_message: The user's message to process

        Returns:
            List of task dictionaries or None if generation fails
        """
        available_skills = list(self.skills.keys())
        skill_descriptions = self._get_skill_descriptions()

        prompt = f"""Analyze the following user request and create a JSON task plan.

Available skills: {", ".join(available_skills)}

Skill descriptions:
{skill_descriptions}

User request: {user_message}

Generate a task plan as a JSON array. Each task should have:
- "skill": the name of the skill to use
- "arguments": a dictionary of arguments for the skill
- "description": a brief description of what the task does

Return ONLY the JSON array, no other text.

Example format:
[
  {{
    "skill": "search",
    "arguments": {{"query": "example query"}},
    "description": "Search for information"
  }}
]
"""

        try:
            response = self._call_llm(prompt)

            if not response.success:
                logger.error(f"LLM call failed: {response.error}")
                return None

            content = response.data.get("content", "")
            task_plan = json.loads(content)

            if not isinstance(task_plan, list):
                logger.error(f"Task plan is not a list: {type(task_plan)}")
                return None

            logger.info(f"Generated task plan with {len(task_plan)} tasks")
            return task_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse task plan JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error generating task plan: {str(e)}")
            return None

    def _generate_final_response(
        self, user_message: str, execution_results: List[Dict]
    ) -> str:
        """Generate final response from execution results.

        Args:
            user_message: The original user message
            execution_results: Results from executing tasks

        Returns:
            Final response string
        """
        results_summary = json.dumps(execution_results, indent=2)

        prompt = f"""Based on the following execution results, provide a helpful response to the user's request.

User request: {user_message}

Execution results:
{results_summary}

Provide a clear, concise response that directly addresses the user's request based on the execution results.
"""

        try:
            response = self._call_llm(prompt)

            if response.success:
                return response.data.get("content", "Unable to generate response")
            else:
                return f"Error generating final response: {response.error}"

        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            return f"Error generating final response: {str(e)}"

    def _get_skill_descriptions(self) -> str:
        """Get descriptions of all available skills.

        Returns:
            String containing skill descriptions
        """
        descriptions = []
        for skill_name, skill in self.skills.items():
            info = skill.get_info()
            descriptions.append(f"- {skill_name}: {info.get('version', '1.0.0')}")
        return "\n".join(descriptions)

    def _call_llm(self, prompt: str) -> SkillResult:
        """Call LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            SkillResult with LLM response
        """
        try:
            import requests

            chat_url = f"{self.llm_skill.base_url}/chat/completions"

            payload = {
                "model": self.llm_skill.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
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
            logger.error(f"LLM request failed: {str(e)}")
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")
