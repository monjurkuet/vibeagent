"""Prompt templates for the VibeAgent multi-agent system.

This package provides production-ready prompt templates for:
- ReAct (Reasoning + Acting) pattern
- Tool orchestration
- Multi-step reasoning
- Error recovery
- Parallel tool execution
"""

from .react_prompt import (
    REACT_SYSTEM_PROMPTS,
    FEW_SHOT_EXAMPLES,
    PromptTemplate,
    get_react_system_prompt,
    get_few_shot_examples,
    format_example,
    build_react_prompt,
    get_example_by_name,
    validate_prompt_structure,
    extract_tool_descriptions,
    build_tool_focused_prompt,
)

__all__ = [
    "REACT_SYSTEM_PROMPTS",
    "FEW_SHOT_EXAMPLES",
    "PromptTemplate",
    "get_react_system_prompt",
    "get_few_shot_examples",
    "format_example",
    "build_react_prompt",
    "get_example_by_name",
    "validate_prompt_structure",
    "extract_tool_descriptions",
    "build_tool_focused_prompt",
]
