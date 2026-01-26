"""Example demonstrating parallel tool execution in VibeAgent."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parallel_executor import (
    ParallelExecutor,
    ParallelExecutorConfig,
)
from core.skill import BaseSkill, SkillResult
from core.database_manager import DatabaseManager


class WebSearchSkill(BaseSkill):
    """Mock web search skill."""

    def __init__(self):
        super().__init__("web_search")

    def execute(self, query: str, **kwargs) -> SkillResult:
        return SkillResult(
            success=True,
            data={
                "results": [
                    f"Result 1 for '{query}'",
                    f"Result 2 for '{query}'",
                ],
                "count": 2,
            },
        )

    def validate(self) -> bool:
        return True

    def get_dependencies(self) -> list:
        return []

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }


class FileReadSkill(BaseSkill):
    """Mock file read skill."""

    def __init__(self):
        super().__init__("file_read")

    def execute(self, path: str, **kwargs) -> SkillResult:
        return SkillResult(
            success=True,
            data={"content": f"Content of {path}", "lines": 42},
        )

    def validate(self) -> bool:
        return True

    def get_dependencies(self) -> list:
        return []

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path",
                        }
                    },
                    "required": ["path"],
                },
            },
        }


class DataAnalysisSkill(BaseSkill):
    """Mock data analysis skill."""

    def __init__(self):
        super().__init__("data_analysis")

    def execute(self, data: str, **kwargs) -> SkillResult:
        return SkillResult(
            success=True,
            data={"analysis": f"Analyzed {len(data)} characters", "insights": 5},
        )

    def validate(self) -> bool:
        return True

    def get_dependencies(self) -> list:
        return []

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "data_analysis",
                "description": "Analyze data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Data to analyze",
                        }
                    },
                    "required": ["data"],
                },
            },
        }


def main():
    """Demonstrate parallel tool execution."""
    print("\n" + "=" * 60)
    print("Parallel Tool Execution Example")
    print("=" * 60)

    skills = {
        "web_search": WebSearchSkill(),
        "file_read": FileReadSkill(),
        "data_analysis": DataAnalysisSkill(),
    }

    db_manager = DatabaseManager("data/example_parallel.db")

    config = ParallelExecutorConfig(
        max_parallel_calls=5,
        enable_parallel=True,
        track_performance=True,
        default_timeout_ms=30000,
    )

    executor = ParallelExecutor(
        skills=skills,
        db_manager=db_manager,
        config=config,
    )

    print("\n--- Scenario 1: Multiple Independent Searches ---")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "Python programming"}),
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "AsyncIO tutorial"}),
            },
        },
        {
            "id": "call_3",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "Machine learning"}),
            },
        },
    ]

    result = executor.execute_parallel(tool_calls)

    print(f"Success: {result.success}")
    print(f"Parallel time: {result.parallel_time_ms:.0f}ms")
    print(f"Speedup: {result.speedup:.2f}x")
    print(f"Results: {len(result.results)}")

    for i, r in enumerate(result.results):
        if r.get("success"):
            print(f"  [{i}] {r['data']['results'][0]}")

    print("\n--- Scenario 2: Mixed Operations ---")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "file_read",
                "arguments": json.dumps({"path": "/data/document.txt"}),
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "API documentation"}),
            },
        },
        {
            "id": "call_3",
            "type": "function",
            "function": {
                "name": "data_analysis",
                "arguments": json.dumps({"data": "sample data for analysis"}),
            },
        },
    ]

    result = executor.execute_parallel(tool_calls)

    print(f"Success: {result.success}")
    print(f"Parallel time: {result.parallel_time_ms:.0f}ms")
    print(f"Speedup: {result.speedup:.2f}x")

    for i, r in enumerate(result.results):
        tool_name = tool_calls[i]["function"]["name"]
        if r.get("success"):
            print(f"  [{tool_name}] {r['data']}")

    print("\n--- Performance Statistics ---")
    stats = executor.get_performance_stats()
    print(f"Total executions: {stats.get('total_executions', 0)}")
    print(f"Average speedup: {stats.get('avg_speedup', 0):.2f}x")
    print(f"Average execution time: {stats.get('avg_execution_time_ms', 0):.0f}ms")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
