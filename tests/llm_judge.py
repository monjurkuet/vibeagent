"""
LLM Judge for semantic verification of tool calls.
Uses gemini-2.5-flash to evaluate if tool calls are semantically correct.
"""

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMJudge:
    """Uses an LLM to semantically verify tool calls."""

    def __init__(
        self,
        base_url: str = "http://localhost:8087/v1",
        judge_model: str = "gemini-2.5-flash",
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.judge_model = judge_model

    def verify_tool_call(
        self, test_case: dict[str, Any], actual_tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Use LLM to verify if tool calls are semantically correct.

        Returns:
            {
                "passed": bool,
                "reasoning": str,
                "confidence": float,
                "details": Dict
            }
        """
        if not actual_tool_calls:
            return {
                "passed": not bool(test_case.get("expected_tools")),
                "reasoning": "No tool calls made"
                if not test_case.get("expected_tools")
                else "Expected tool calls but none made",
                "confidence": 1.0,
                "details": {},
            }

        # Build verification prompt
        prompt = self._build_verification_prompt(test_case, actual_tool_calls)

        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": self.judge_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an impartial judge evaluating if LLM tool calls are semantically correct. Respond only in valid JSON format.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,  # Low temperature for consistent judgments
                    "max_tokens": 1000,
                },
                timeout=60,  # Increased timeout
            )
            response.raise_for_status()
            data = response.json()

            # Extract and parse judgment
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            try:
                # Try to extract JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                judgment = json.loads(content)
                return {
                    "passed": judgment.get("passed", False),
                    "reasoning": judgment.get("reasoning", ""),
                    "confidence": judgment.get("confidence", 0.5),
                    "details": judgment.get("details", {}),
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM judgment: {e}")
                logger.error(f"Raw content: {content}")
                # Fallback to basic verification
                return self._fallback_verification(test_case, actual_tool_calls)

        except Exception as e:
            logger.error(f"LLM judge request failed: {e}")
            # Fallback to basic verification
            return self._fallback_verification(test_case, actual_tool_calls)

    def _build_verification_prompt(
        self, test_case: dict[str, Any], actual_tool_calls: list[dict[str, Any]]
    ) -> str:
        """Build the prompt for the LLM judge."""

        prompt = f"""Evaluate if the following tool calls are semantically correct for the given test case.

## Test Case
**Name**: {test_case.get("name", "Unknown")}
**User Request**: {test_case.get("messages", [{}])[0].get("content", "No content")}

## Available Tools
"""
        # Add tools info
        for i, tool in enumerate(test_case.get("tools", []), 1):
            func = tool.get("function", {})
            prompt += f"\n{i}. **{func.get('name', 'Unknown')}**: {func.get('description', 'No description')}\n"
            params = func.get("parameters", {}).get("properties", {})
            if params:
                prompt += "   Parameters:\n"
                for param_name, param_info in params.items():
                    desc = param_info.get("description", "")
                    param_type = param_info.get("type", "unknown")
                    required = param_name in func.get("parameters", {}).get("required", [])
                    prompt += f"   - {param_name} ({param_type}){' [REQUIRED]' if required else ' [OPTIONAL]'}: {desc}\n"

        prompt += """

## Expected Behavior
"""
        if test_case.get("expected_tools"):
            prompt += "Expected to call these tools (in order):\n"
            for i, expected_tool in enumerate(test_case.get("expected_tools", []), 1):
                prompt += f"{i}. {expected_tool.get('name', 'Unknown')}\n"
                if expected_tool.get("parameters"):
                    prompt += f"   Expected parameters: {json.dumps(expected_tool.get('parameters'), indent=2)}\n"
        elif test_case.get("expect_no_tools"):
            prompt += "Expected NO tools to be called (should answer directly).\n"
        else:
            prompt += "Expected to call appropriate tools based on the request.\n"

        prompt += """

## Actual Tool Calls Made by Model
"""
        for i, tool_call in enumerate(actual_tool_calls, 1):
            func = tool_call.get("function", {})
            func_name = func.get("name", "Unknown")
            args = func.get("arguments", "{}")

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    pass

            prompt += f"\n{i}. **{func_name}**\n"
            prompt += (
                f"   Arguments: {json.dumps(args, indent=2) if isinstance(args, dict) else args}\n"
            )

        prompt += """

## Evaluation Criteria

Consider the following when evaluating:

1. **Semantic Correctness**: Does the tool call address the user's intent? (Even if exact wording differs)
2. **Parameter Appropriateness**: Are the parameters reasonable for the request?
3. **Tool Selection**: Is the right tool selected for the task?
4. **Completeness**: Are all required parameters provided?
5. **Reasonableness**: Are the parameter values sensible (e.g., not requesting 1,000,000 results)?

**Be lenient**: Allow for reasonable interpretations. "AI" could match "machine learning", "neural networks", "deep learning", etc.
**Be strict on**: Required parameters, completely wrong tools, nonsensical values.

## Your Task

Provide a JSON response with this exact structure:

```json
{
  "passed": true/false,
  "reasoning": "Explain why it passed or failed in 1-2 sentences",
  "confidence": 0.0-1.0,
  "details": {
    "correct_tools": ["list of correctly called tools"],
    "incorrect_tools": ["list of incorrectly called tools"],
    "parameter_issues": ["list of any parameter problems"],
    "overall_assessment": "brief summary"
  }
}
```

Examples:
- If model calls arxiv_search with query="AI" when expected was "machine learning" → PASS (semantically equivalent)
- If model calls arxiv_search with query="banana recipes" when expected was "machine learning" → FAIL (completely unrelated)
- If model adds max_results=10 when not expected → PASS (reasonable default)
- If model calls save_paper before searching → FAIL (wrong tool order)

Provide ONLY the JSON response, nothing else.
"""

        return prompt

    def _fallback_verification(
        self, test_case: dict[str, Any], actual_tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Fallback to basic verification if LLM judge fails."""

        logger.warning("Using fallback verification")

        expected_tools = test_case.get("expected_tools", [])

        if not expected_tools:
            passed = len(actual_tool_calls) == 0
            reasoning = (
                "No tools expected and none called"
                if passed
                else "No tools expected but tools were called"
            )
        # Basic check: same number of tools and same names
        elif len(actual_tool_calls) != len(expected_tools):
            passed = False
            reasoning = f"Wrong number of tools: expected {len(expected_tools)}, got {len(actual_tool_calls)}"
        else:
            passed = True
            reasoning = "Basic verification passed (same tool count)"
            for i, (actual, expected) in enumerate(
                zip(actual_tool_calls, expected_tools, strict=False)
            ):
                actual_name = actual.get("function", {}).get("name", "")
                expected_name = expected.get("name", "")
                if actual_name != expected_name:
                    passed = False
                    reasoning = f"Tool {i}: expected {expected_name}, got {actual_name}"
                    break

        return {
            "passed": passed,
            "reasoning": reasoning,
            "confidence": 0.5,
            "details": {"method": "fallback"},
        }

    def verify_multiple_calls(
        self, test_case: dict[str, Any], actual_tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Verify multiple tool calls (for complex scenarios).

        Returns overall judgment with per-call details.
        """
        if not actual_tool_calls:
            return self.verify_tool_call(test_case, actual_tool_calls)

        results = []
        for i, tool_call in enumerate(actual_tool_calls):
            # Create a mini test case for each tool call
            mini_test = {
                "name": f"{test_case.get('name')} - Call {i + 1}",
                "tools": test_case.get("tools", []),
                "expected_tools": [tool_call],  # Self-verify
            }

            # Actually, we want to verify against the overall expected tools
            # So let's just use the main verification

        # For now, use the main verification
        return self.verify_tool_call(test_case, actual_tool_calls)
