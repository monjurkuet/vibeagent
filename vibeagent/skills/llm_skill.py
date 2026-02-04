"""LLM interaction skill for the agent framework with round-robin model selection."""

import logging
import threading
from typing import Any

import requests

from ..core.skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

# List of available models (excluding gemini)
AVAILABLE_MODELS = [
    "deepseek-v3.2",
    "qwen3-235b-a22b-instruct",
    "iflow-rome-30ba3b",
    "qwen3-max",
    "glm-4.7",
    "deepseek-v3",
    "qwen3-235b-a22b-thinking-2507",
    "minimax-m2",
    "minimax-m2.1",
    "qwen3-coder-flash_QWEN",
    "tstars2.0",
    "deepseek-v3.2-reasoner",
    "deepseek-v3.1",
    "qwen3-235b",
    "qwen3-coder-plus_QWEN",
    "kimi-k2-thinking",
    "qwen3-32b",
    "vision-model_QWEN",
    "glm-4.6",
    "kimi-k2",
    "deepseek-v3.2-chat",
    "deepseek-r1",
    "qwen3-coder-plus",
    "qwen3-max-preview",
    "kimi-k2-0905",
]

# Models to exclude (gemini models)
EXCLUDED_MODELS = ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash"]


class ModelRoundRobin:
    """Round-robin model selector with retry support."""

    _lock = threading.Lock()
    _current_index = 0

    @classmethod
    def get_next_model(cls) -> str:
        """Get next model in round-robin sequence."""
        with cls._lock:
            model = AVAILABLE_MODELS[cls._current_index]
            cls._current_index = (cls._current_index + 1) % len(AVAILABLE_MODELS)
            return model

    @classmethod
    def get_models(cls, exclude: list[str] | None = None) -> list[str]:
        """Get list of available models, excluding specified ones."""
        exclude = exclude or []
        return [m for m in AVAILABLE_MODELS if m not in exclude]

    @classmethod
    def reset_index(cls):
        """Reset round-robin index to start."""
        with cls._lock:
            cls._current_index = 0


class LLMSkill(BaseSkill):
    """Skill for interacting with LLM APIs with round-robin model selection."""

    def __init__(
        self,
        base_url: str = "http://localhost:8087/v1",
        model: str | None = None,
        enable_round_robin: bool = True,
        max_retries: int = 3,
    ):
        """Initialize LLM skill.

        Args:
            base_url: Base URL for OpenAI-compatible API
            model: Specific model to use (if None, uses round-robin)
            enable_round_robin: Enable round-robin model selection
            max_retries: Maximum retry attempts with different models
        """
        super().__init__("llm", "1.0.0")
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.model = model
        self.enable_round_robin = enable_round_robin
        self.max_retries = max_retries
        self.activate()

    def _get_model_for_request(self, exclude_models: list[str] | None = None) -> str:
        """Get model for current request.

        Args:
            exclude_models: Models to exclude from selection

        Returns:
            Model name
        """
        if self.model and not self.enable_round_robin:
            return self.model

        exclude = list(EXCLUDED_MODELS)
        if exclude_models:
            exclude.extend(exclude_models)

        return ModelRoundRobin.get_next_model()

    @property
    def parameters_schema(self) -> dict:
        """JSON Schema for the skill's parameters."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The user prompt to send to the LLM",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt to set context",
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 1000,
                    "description": "Maximum number of tokens to generate",
                },
                "temperature": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Sampling temperature",
                },
            },
            "required": ["prompt"],
        }

    def get_tool_schema(self) -> dict:
        """Get the tool schema for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Generate text responses using an LLM with round-robin model selection",
                "parameters": self.parameters_schema,
            },
        }

    def validate(self) -> bool:
        """Validate the skill configuration."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.Timeout:
            logger.error(f"LLM validation timeout: {self.base_url}")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"LLM connection error: {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return False

    def get_dependencies(self) -> list[str]:
        """Return list of dependencies."""
        return ["requests"]

    def _execute_with_retry(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        retry_count: int = 0,
        exclude_models: list[str] | None = None,
    ) -> SkillResult:
        """Execute LLM request with retry logic.

        Args:
            model: Model to use
            messages: Messages to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Tool definitions
            tool_choice: Tool choice setting
            retry_count: Current retry count
            exclude_models: Models to exclude from retry

        Returns:
            SkillResult
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if tools:
                payload["tools"] = tools

            if tool_choice:
                payload["tool_choice"] = tool_choice

            logger.info(f"Executing LLM request with model: {model}")

            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]
            content = message.get("reasoning_content") or message.get("content", "")
            tool_calls = message.get("tool_calls", None)

            logger.info(f"LLM request succeeded with model: {model}")

            return SkillResult(
                success=True,
                data={
                    "content": content,
                    "tool_calls": tool_calls,
                    "model": model,
                    "usage": result.get("usage", {}),
                },
            )

        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json() if e.response else str(e)
            logger.warning(f"LLM HTTP error with model {model}: {error_detail}")

            # Retry with different model if within retry limit
            if retry_count < self.max_retries:
                new_exclude = (exclude_models or []) + [model]
                new_model = self._get_model_for_request(exclude_models=new_exclude)
                logger.info(f"Retrying with model: {new_model} (attempt {retry_count + 1}/{self.max_retries})")
                return self._execute_with_retry(
                    new_model,
                    messages,
                    temperature,
                    max_tokens,
                    tools,
                    tool_choice,
                    retry_count + 1,
                    new_exclude,
                )
            else:
                return SkillResult(success=False, error=f"LLM HTTP error: {error_detail}")

        except requests.exceptions.Timeout:
            logger.warning(f"LLM timeout with model {model}")

            # Retry with different model if within retry limit
            if retry_count < self.max_retries:
                new_exclude = (exclude_models or []) + [model]
                new_model = self._get_model_for_request(exclude_models=new_exclude)
                logger.info(f"Retrying with model: {new_model} (attempt {retry_count + 1}/{self.max_retries})")
                return self._execute_with_retry(
                    new_model,
                    messages,
                    temperature,
                    max_tokens,
                    tools,
                    tool_choice,
                    retry_count + 1,
                    new_exclude,
                )
            else:
                return SkillResult(success=False, error="LLM request timed out")

        except requests.exceptions.RequestException as e:
            logger.warning(f"LLM request error with model {model}: {str(e)}")

            # Retry with different model if within retry limit
            if retry_count < self.max_retries:
                new_exclude = (exclude_models or []) + [model]
                new_model = self._get_model_for_request(exclude_models=new_exclude)
                logger.info(f"Retrying with model: {new_model} (attempt {retry_count + 1}/{self.max_retries})")
                return self._execute_with_retry(
                    new_model,
                    messages,
                    temperature,
                    max_tokens,
                    tools,
                    tool_choice,
                    retry_count + 1,
                    new_exclude,
                )
            else:
                return SkillResult(success=False, error=f"LLM request failed: {str(e)}")

        except Exception as e:
            logger.error(f"LLM execution error with model {model}: {str(e)}")
            return SkillResult(success=False, error=f"LLM execution error: {str(e)}")

    def execute(self, **kwargs) -> SkillResult:
        """Execute a prompt through the LLM with round-robin model selection."""
        prompt = kwargs.get("prompt")
        system_prompt = kwargs.get("system_prompt")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)

        if not prompt:
            return SkillResult(success=False, error="Prompt is required")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Get model and execute with retry
        model = self._get_model_for_request()
        return self._execute_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def execute_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> SkillResult:
        """Execute LLM request with tool calling support and round-robin model selection."""
        # Get model and execute with retry
        model = self._get_model_for_request()
        return self._execute_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )

    def parse_tool_calls(self, response_data: dict) -> list[dict]:
        """Parse tool calls from LLM response data."""
        tool_calls = response_data.get("tool_calls", [])

        if not tool_calls:
            return []

        parsed_calls = []
        for tool_call in tool_calls:
            call_dict = {
                "id": tool_call.get("id"),
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": tool_call.get("function", {}).get("name"),
                    "arguments": tool_call.get("function", {}).get("arguments"),
                },
            }
            parsed_calls.append(call_dict)

        return parsed_calls