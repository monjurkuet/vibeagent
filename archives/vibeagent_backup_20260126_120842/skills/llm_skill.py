"""LLM interaction skill for the agent framework."""

import requests
from typing import Dict, List, Optional

from core.skill import BaseSkill, SkillResult


class LLMSkill(BaseSkill):
    """Skill for interacting with LLM APIs."""

    def __init__(
        self, base_url: str = "http://localhost:8087/v1", model: str = "glm-4.7"
    ):
        super().__init__("llm", "1.0.0")
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.model = model

    @property
    def parameters_schema(self) -> Dict:
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
            },
            "required": ["prompt"],
        }

    def get_tool_schema(self) -> Dict:
        """Get the tool schema for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Generate text responses using an LLM",
                "parameters": self.parameters_schema,
            },
        }

    def validate(self) -> bool:
        """Validate the skill configuration."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"LLM validation failed: {e}")
            return False

    def get_dependencies(self) -> List[str]:
        """Return list of dependencies."""
        return ["requests"]

    def execute(self, **kwargs) -> SkillResult:
        """Execute a prompt through the LLM."""
        prompt = kwargs.get("prompt")
        system_prompt = kwargs.get("system_prompt")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)

        if not prompt:
            return SkillResult(success=False, error="Prompt is required")

        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                self.chat_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]
            content = message.get("reasoning_content") or message.get("content", "")

            return SkillResult(
                success=True,
                data={
                    "content": content,
                    "model": self.model,
                    "usage": result.get("usage", {}),
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")

    def execute_with_tools(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> SkillResult:
        """Execute LLM request with tool calling support."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if tools:
                payload["tools"] = tools

            if tool_choice:
                payload["tool_choice"] = tool_choice

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

            return SkillResult(
                success=True,
                data={
                    "content": content,
                    "tool_calls": tool_calls,
                    "model": self.model,
                    "usage": result.get("usage", {}),
                },
            )
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json() if e.response else str(e)
            return SkillResult(success=False, error=f"LLM HTTP error: {error_detail}")
        except requests.exceptions.Timeout:
            return SkillResult(success=False, error="LLM request timed out")
        except requests.exceptions.RequestException as e:
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")
        except Exception as e:
            return SkillResult(success=False, error=f"LLM execution error: {str(e)}")

    def parse_tool_calls(self, response_data: Dict) -> List[Dict]:
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
