#!/usr/bin/env python3
"""
Example usage of LLMToolCallingTester with DatabaseManager integration.

This script demonstrates how to use the enhanced test tracking capabilities.
"""

from core.database_manager import DatabaseManager

from tests.llm_tool_calling_tester import LLMToolCallingTester

# Initialize database manager (optional)
db_manager = DatabaseManager(db_path="data/vibeagent.db")

# Initialize tester with database tracking
tester = LLMToolCallingTester(
    base_url="http://localhost:8087/v1",
    use_llm_judge=True,
    judge_model="gemini-2.5-flash",
    db_manager=db_manager,  # Optional: pass None to disable DB tracking
)

# Define test cases
test_cases = [
    {
        "name": "simple_weather_query",
        "category": "tool_calling",
        "description": "Test basic weather tool calling",
        "messages": [{"role": "user", "content": "What's the weather in New York?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        "expected_tools": [{"name": "get_weather", "parameters": {"location": "New York"}}],
    }
]

# Run tests
results = tester.test_model_tool_calling("gpt-4", test_cases)

# Get test history
history = tester.get_test_history("simple_weather_query", limit=5)
print(f"\nHistory for 'simple_weather_query': {len(history)} runs")

# Get performance trends
trends = tester.get_performance_trends(days=30)
print(f"\nPerformance trends: {trends.get('total_test_cases', 0)} test cases tracked")
