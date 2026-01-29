"""Prompt templates for the VibeAgent multi-agent system.

This package provides production-ready prompt templates for:
- ReAct (Reasoning + Acting) pattern
- Tool orchestration
- Multi-step reasoning
- Error recovery
- Parallel tool execution
"""

from .react_prompt import (
    FEW_SHOT_EXAMPLES,
    REACT_SYSTEM_PROMPTS,
    PromptTemplate,
    build_react_prompt,
    build_tool_focused_prompt,
    extract_tool_descriptions,
    format_example,
    get_example_by_name,
    get_few_shot_examples,
    get_react_system_prompt,
    validate_prompt_structure,
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
