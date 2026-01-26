"""Example integration of ReAct prompts with the tool orchestrator.

This script demonstrates how to use the ReAct prompt templates
to improve tool calling success rates in the VibeAgent system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import (
    build_react_prompt,
    get_react_system_prompt,
    extract_tool_descriptions,
)


def example_basic_usage():
    """Example: Basic usage with enhanced prompts."""
    print("\n" + "=" * 60)
    print("Example 1: Basic ReAct Prompt Usage")
    print("=" * 60 + "\n")

    # Sample tools (in production, these come from your skills)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "arxiv_search",
                "description": "Search for papers on arXiv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 10},
                        "categories": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # User message
    user_message = "Search for papers about machine learning"

    # Build enhanced ReAct prompt
    messages = [{"role": "user", "content": user_message}]
    react_prompt = build_react_prompt(
        messages=messages,
        tools=tools,
        model_type="gpt4",
        include_examples=True,
        example_categories=["simple"],
    )

    print("System Prompt (first 500 chars):")
    print(react_prompt[0]["content"][:500] + "...\n")

    print("User Message:")
    print(react_prompt[-1]["content"] + "\n")

    print(f"Total messages in prompt: {len(react_prompt)}")


def example_multi_step_workflow():
    """Example: Multi-step workflow with chaining."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Step Workflow")
    print("=" * 60 + "\n")

    # Sample tools for chaining
    tools = [
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
                "name": "llm_extract",
                "description": "Extract information using LLM",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "extract_field": {"type": "string"},
                    },
                    "required": ["text", "extract_field"],
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

    # Complex task requiring multiple tools
    user_message = "Find papers about neural networks, extract their abstracts, and save them to the database"

    # Build prompt with chaining examples
    messages = [{"role": "user", "content": user_message}]
    react_prompt = build_react_prompt(
        messages=messages,
        tools=tools,
        model_type="gpt4",
        include_examples=True,
        example_categories=["chaining"],
    )

    print("Task:", user_message)
    print("\nRecommended Tools:")
    descriptions = extract_tool_descriptions(tools)
    for desc in descriptions:
        print(f"  - {desc}")

    print(f"\nPrompt Structure:")
    print(f"  - System prompt with ReAct instructions")
    print(f"  - Few-shot examples for chaining workflows")
    print(f"  - User task")
    print(f"Total messages: {len(react_prompt)}")


def example_error_recovery():
    """Example: Handling errors gracefully."""
    print("\n" + "=" * 60)
    print("Example 3: Error Recovery")
    print("=" * 60 + "\n")

    tools = [
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

    # User message with incomplete information
    user_message = "Search for papers"  # Missing query

    # Build prompt with error recovery examples
    messages = [{"role": "user", "content": user_message}]
    react_prompt = build_react_prompt(
        messages=messages,
        tools=tools,
        model_type="claude",  # Claude is good at careful error handling
        include_examples=True,
        example_categories=["error_recovery"],
    )

    print("Scenario: User provides incomplete information")
    print("Message:", user_message)
    print("\nExpected Behavior:")
    print("  1. Agent attempts to call arxiv_search")
    print("  2. Tool returns error: 'query' parameter required")
    print("  3. Agent analyzes error and asks user for clarification")
    print("  4. Agent waits for user response before proceeding")
    print(
        f"\nPrompt includes {len(react_prompt)} messages with error recovery examples"
    )


def example_parallel_execution():
    """Example: Parallel tool execution."""
    print("\n" + "=" * 60)
    print("Example 4: Parallel Tool Execution")
    print("=" * 60 + "\n")

    tools = [
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

    # Task requiring independent searches
    user_message = "Search for papers about reinforcement learning and computer vision simultaneously"

    # Build prompt with parallel examples
    messages = [{"role": "user", "content": user_message}]
    react_prompt = build_react_prompt(
        messages=messages,
        tools=tools,
        model_type="gpt4",
        include_examples=True,
        example_categories=["parallel"],
    )

    print("Task:", user_message)
    print("\nParallel Execution Strategy:")
    print("  1. Thought: Identify two independent searches")
    print("  2. Action: arxiv_search with 'reinforcement learning'")
    print("  3. Thought: Start second search while first runs")
    print("  4. Action: arxiv_search with 'computer vision'")
    print("  5. Observation: Receive results from both searches")
    print("  6. Final Answer: Combine and present results")
    print(
        f"\nPrompt includes {len(react_prompt)} messages with parallel execution examples"
    )


def example_model_comparison():
    """Example: Compare prompts for different models."""
    print("\n" + "=" * 60)
    print("Example 5: Model-Specific Prompt Comparison")
    print("=" * 60 + "\n")

    tools = [
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

    user_message = "Search for papers about AI"
    messages = [{"role": "user", "content": user_message}]

    models = ["default", "gpt4", "claude", "local_llm"]

    print("Comparing system prompts for different models:\n")

    for model_type in models:
        prompt = get_react_system_prompt(
            model_type=model_type, tool_descriptions=extract_tool_descriptions(tools)
        )

        print(f"{model_type.upper()}:")
        print(f"  Length: {len(prompt)} characters")
        print(f"  Key features:")

        if "step-by-step" in prompt:
            print("    - Step-by-step reasoning emphasis")
        if "parallel" in prompt.lower():
            print("    - Parallel execution guidance")
        if "error" in prompt.lower():
            print("    - Error handling instructions")
        if "simple" in prompt.lower():
            print("    - Simplified instructions")

        print()


def example_custom_integration():
    """Example: Custom integration with existing orchestrator."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Integration Pattern")
    print("=" * 60 + "\n")

    # This shows how to modify the orchestrator to use ReAct prompts

    code_example = '''
    class ReActToolOrchestrator(ToolOrchestrator):
        """Enhanced orchestrator with ReAct prompts."""
        
        def __init__(self, llm_skill, skills, model_type="gpt4"):
            super().__init__(llm_skill, skills)
            self.model_type = model_type
        
        def _build_enhanced_messages(self, user_message: str) -> List[Dict]:
            """Build messages with ReAct prompt."""
            base_messages = [{"role": "user", "content": user_message}]
            
            return build_react_prompt(
                messages=base_messages,
                tools=self._tool_schemas,
                model_type=self.model_type,
                include_examples=True,
                example_categories=["simple", "chaining", "error_recovery"]
            )
        
        def execute_with_tools(self, user_message: str, max_iterations: int = 10):
            """Execute with ReAct-enhanced prompts."""
            # Build enhanced prompt
            messages = self._build_enhanced_messages(user_message)
            
            # Continue with standard orchestration
            iterations = 0
            tool_calls_made = 0
            
            while iterations < max_iterations:
                iterations += 1
                
                # Call LLM with enhanced prompt
                llm_result = self._call_llm_with_tools(messages)
                
                if not llm_result.success:
                    return self._create_error_result(llm_result.error)
                
                assistant_message = llm_result.data.get("message", {})
                messages.append(assistant_message)
                
                # Process tool calls
                tool_calls = self.parse_tool_calls(assistant_message)
                
                if not tool_calls:
                    # Task complete
                    return self._create_success_result(
                        assistant_message.get("content", ""),
                        iterations,
                        tool_calls_made
                    )
                
                # Execute tools
                for tool_call in tool_calls:
                    tool_calls_made += 1
                    tool_result = self._execute_tool(tool_call)
                    
                    messages.append({
                        "tool_call_id": tool_call.get("id"),
                        "role": "tool",
                        "content": json.dumps({
                            "success": tool_result.success,
                            "data": tool_result.data,
                            "error": tool_result.error,
                        })
                    })
            
            return self._create_error_result("Max iterations reached")
    '''

    print("Integration Code:")
    print(code_example)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ReAct Prompt Templates - Integration Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_multi_step_workflow()
        example_error_recovery()
        example_parallel_execution()
        example_model_comparison()
        example_custom_integration()

        print("\n" + "=" * 60)
        print("All Examples Completed")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  1. Use build_react_prompt() for enhanced tool orchestration")
        print("  2. Choose model_type based on your LLM")
        print("  3. Include relevant example categories for your tasks")
        print("  4. Model-specific prompts optimize for different capabilities")
        print("  5. Error recovery examples improve resilience")
        print("  6. Parallel examples guide independent execution")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
