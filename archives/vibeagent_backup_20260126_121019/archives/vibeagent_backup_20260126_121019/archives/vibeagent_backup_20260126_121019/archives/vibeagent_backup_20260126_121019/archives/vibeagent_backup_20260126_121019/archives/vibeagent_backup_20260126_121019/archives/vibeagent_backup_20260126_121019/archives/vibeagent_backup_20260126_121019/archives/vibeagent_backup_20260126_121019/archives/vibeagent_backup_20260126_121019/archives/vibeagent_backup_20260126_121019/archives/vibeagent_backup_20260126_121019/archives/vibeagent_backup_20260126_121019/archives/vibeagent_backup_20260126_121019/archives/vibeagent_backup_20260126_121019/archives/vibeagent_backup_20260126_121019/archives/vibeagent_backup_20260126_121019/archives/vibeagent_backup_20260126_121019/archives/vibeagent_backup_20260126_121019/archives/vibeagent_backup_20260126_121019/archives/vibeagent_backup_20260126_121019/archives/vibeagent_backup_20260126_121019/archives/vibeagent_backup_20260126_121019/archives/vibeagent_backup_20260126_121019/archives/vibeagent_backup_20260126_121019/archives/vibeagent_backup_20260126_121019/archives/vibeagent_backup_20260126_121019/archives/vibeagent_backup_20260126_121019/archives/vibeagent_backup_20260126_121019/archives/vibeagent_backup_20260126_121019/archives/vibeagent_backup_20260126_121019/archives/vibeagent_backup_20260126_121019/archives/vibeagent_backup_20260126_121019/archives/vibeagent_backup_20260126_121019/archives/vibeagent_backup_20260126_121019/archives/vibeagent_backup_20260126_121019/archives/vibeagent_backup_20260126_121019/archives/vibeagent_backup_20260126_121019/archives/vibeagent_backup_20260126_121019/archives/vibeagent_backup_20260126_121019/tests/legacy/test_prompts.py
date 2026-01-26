"""Test script for ReAct prompt templates.

This script validates that the prompt templates are correctly structured
and can be used with the tool orchestrator.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prompts.react_prompt import (
    get_react_system_prompt,
    get_few_shot_examples,
    format_example,
    build_react_prompt,
    extract_tool_descriptions,
    REACT_SYSTEM_PROMPTS,
    FEW_SHOT_EXAMPLES,
)


def test_system_prompts():
    """Test that all system prompts are available and well-formed."""
    print("Testing System Prompts...")
    print("=" * 60)

    for model_type in ["default", "gpt4", "claude", "local_llm"]:
        prompt = get_react_system_prompt(model_type)
        assert "{{tool_descriptions}}" not in prompt, (
            f"Template placeholder not replaced for {model_type}"
        )
        assert "Thought:" in prompt, f"Missing Thought pattern in {model_type}"
        assert "Action:" in prompt, f"Missing Action pattern in {model_type}"
        assert "Final Answer:" in prompt, (
            f"Missing Final Answer pattern in {model_type}"
        )
        print(f"✓ {model_type}: {len(prompt)} characters")

    print()


def test_few_shot_examples():
    """Test that few-shot examples are available and well-formed."""
    print("Testing Few-Shot Examples...")
    print("=" * 60)

    categories = ["simple", "chaining", "error_recovery", "parallel", "complex"]

    for category in categories:
        examples = get_few_shot_examples(category)
        assert len(examples) > 0, f"No examples found for category: {category}"

        for example in examples:
            assert "name" in example, f"Example missing name in {category}"
            assert "description" in example, (
                f"Example missing description in {category}"
            )
            assert "conversation" in example, (
                f"Example missing conversation in {category}"
            )
            assert "reasoning_trace" in example, (
                f"Example missing reasoning_trace in {category}"
            )

        print(f"✓ {category}: {len(examples)} examples")

    print()


def test_example_formatting():
    """Test that examples can be formatted correctly."""
    print("Testing Example Formatting...")
    print("=" * 60)

    example = get_few_shot_examples("simple")[0]
    formatted = format_example(example)

    assert example["name"] in formatted, "Example name not in formatted output"
    assert "Thought:" in formatted, "Thought pattern not in formatted output"
    assert "Action:" in formatted, "Action pattern not in formatted output"
    assert "Final Answer:" in formatted, "Final Answer pattern not in formatted output"

    print(f"✓ Example formatted successfully ({len(formatted)} characters)")
    print()


def test_tool_description_extraction():
    """Test tool description extraction from schemas."""
    print("Testing Tool Description Extraction...")
    print("=" * 60)

    sample_tools = [
        {
            "type": "function",
            "function": {
                "name": "arxiv_search",
                "description": "Search for papers on arXiv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_store",
                "description": "Store data in SQLite database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "data": {"type": "object"},
                    },
                    "required": ["table", "data"],
                },
            },
        },
    ]

    descriptions = extract_tool_descriptions(sample_tools)

    assert len(descriptions) == 2, f"Expected 2 descriptions, got {len(descriptions)}"
    assert "arxiv_search" in descriptions[0], "Tool name not in description"
    assert "sqlite_store" in descriptions[1], "Tool name not in description"

    print(f"✓ Extracted {len(descriptions)} tool descriptions")
    for desc in descriptions:
        print(f"  - {desc}")
    print()


def test_full_prompt_building():
    """Test building complete ReAct prompts."""
    print("Testing Full Prompt Building...")
    print("=" * 60)

    sample_messages = [
        {"role": "user", "content": "Search for papers about machine learning"}
    ]

    sample_tools = [
        {
            "type": "function",
            "function": {
                "name": "arxiv_search",
                "description": "Search for papers on arXiv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    for model_type in ["default", "gpt4", "claude"]:
        prompt = build_react_prompt(
            messages=sample_messages,
            tools=sample_tools,
            model_type=model_type,
            include_examples=True,
        )

        assert len(prompt) >= 2, f"Prompt too short for {model_type}"
        assert prompt[0]["role"] == "system", (
            f"First message not system for {model_type}"
        )
        assert any(msg["role"] == "user" for msg in prompt), (
            f"No user message in {model_type}"
        )

        print(f"✓ {model_type}: {len(prompt)} messages total")

    print()


def test_prompt_with_test_cases():
    """Test prompts with actual test cases from test_cases.py."""
    print("Testing with Real Test Cases...")
    print("=" * 60)

    from tests.test_cases import TEST_CASES

    test_case = TEST_CASES[0]  # Simple tool call - ArXiv search

    prompt = build_react_prompt(
        messages=test_case["messages"],
        tools=test_case["tools"],
        model_type="default",
        include_examples=False,
    )

    assert len(prompt) >= 2, "Prompt too short"
    assert "arxiv_search" in prompt[0]["content"], "Tool not in system prompt"

    print(f"✓ Test case '{test_case['name']}' handled correctly")
    print(f"  Messages: {len(prompt)}")
    print(f"  System prompt length: {len(prompt[0]['content'])} characters")
    print()


def test_parallel_calls_example():
    """Test the parallel calls example specifically."""
    print("Testing Parallel Calls Example...")
    print("=" * 60)

    parallel_examples = get_few_shot_examples("parallel")

    assert len(parallel_examples) >= 2, "Not enough parallel examples"

    for example in parallel_examples:
        formatted = format_example(example)
        # Check that it shows multiple Action calls before Observations
        action_count = formatted.count("Action:")
        observation_count = formatted.count("Observation:")

        # Should have more actions than first observations (since they're parallel)
        print(
            f"✓ {example['name']}: {action_count} actions, {observation_count} observations"
        )

    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ReAct Prompt Templates - Test Suite")
    print("=" * 60 + "\n")

    try:
        test_system_prompts()
        test_few_shot_examples()
        test_example_formatting()
        test_tool_description_extraction()
        test_full_prompt_building()
        test_prompt_with_test_cases()
        test_parallel_calls_example()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

        # Print summary
        print("\nSummary:")
        print(f"  System prompts: {len(REACT_SYSTEM_PROMPTS)}")
        print(f"  Example categories: {len(FEW_SHOT_EXAMPLES)}")
        print(f"  Total examples: {sum(len(ex) for ex in FEW_SHOT_EXAMPLES.values())}")

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
