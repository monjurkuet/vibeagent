"""Example usage and tests for ParallelExecutor."""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parallel_executor import (
    ParallelExecutor,
    ParallelExecutorConfig,
    ToolCallInfo,
    ParallelBatch,
    ParallelExecutionStatus,
    ParallelExecutionResult,
)
from core.skill import BaseSkill, SkillResult
from core.database_manager import DatabaseManager


class MockSkill(BaseSkill):
    """Mock skill for testing."""

    def __init__(self, name: str, delay: float = 0.1):
        super().__init__(name)
        self.delay = delay

    def execute(self, **kwargs) -> SkillResult:
        time.sleep(self.delay)
        return SkillResult(
            success=True,
            data={"result": f"Executed {self.name}", "params": kwargs},
        )

    def validate(self) -> bool:
        return True

    def get_dependencies(self) -> list:
        return []

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"Mock skill {self.name}",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": [],
                },
            },
        }


def test_basic_parallel_execution():
    """Test basic parallel execution of independent calls."""
    print("\n=== Test: Basic Parallel Execution ===")

    skills = {
        "search_web": MockSkill("search_web", delay=0.1),
        "search_docs": MockSkill("search_docs", delay=0.1),
        "search_code": MockSkill("search_code", delay=0.1),
        "scrape_url": MockSkill("scrape_url", delay=0.1),
        "analyze_data": MockSkill("analyze_data", delay=0.1),
    }

    db_manager = DatabaseManager("data/test_parallel.db")
    executor = ParallelExecutor(
        skills=skills,
        db_manager=db_manager,
        config=ParallelExecutorConfig(
            max_parallel_calls=5,
            enable_parallel=True,
            track_performance=True,
        ),
    )

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "test"}),
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "search_docs",
                "arguments": json.dumps({"query": "test"}),
            },
        },
        {
            "id": "call_3",
            "type": "function",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "test"}),
            },
        },
        {
            "id": "call_4",
            "type": "function",
            "function": {
                "name": "scrape_url",
                "arguments": json.dumps({"query": "test"}),
            },
        },
        {
            "id": "call_5",
            "type": "function",
            "function": {
                "name": "analyze_data",
                "arguments": json.dumps({"query": "test"}),
            },
        },
    ]

    start = time.time()
    result = executor.execute_parallel(tool_calls)
    parallel_time = time.time() - start

    start_seq = time.time()
    for tc in tool_calls:
        skills[tc["function"]["name"]].execute(
            **json.loads(tc["function"]["arguments"])
        )
    sequential_time = time.time() - start_seq

    actual_speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Actual speedup: {actual_speedup:.2f}x")
    print(f"Success: {result.success}")
    print(f"Results: {len(result.results)}")
    print(f"Batches: {len(result.batches)}")

    for i, r in enumerate(result.results):
        print(f"  Result {i}: success={r.get('success')}, data={r.get('data')}")

    assert result.success
    assert len(result.results) == 5
    assert actual_speedup > 1.5, f"Expected speedup > 1.5, got {actual_speedup}"

    print("✓ Basic parallel execution test passed")


def test_dependency_analysis():
    """Test dependency detection and analysis."""
    print("\n=== Test: Dependency Analysis ===")

    skills = {
        "search": MockSkill("search", delay=0.05),
        "scrape": MockSkill("scrape", delay=0.05),
        "analyze": MockSkill("analyze", delay=0.05),
    }

    executor = ParallelExecutor(skills=skills)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "scrape", "arguments": "{}"},
        },
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "analyze", "arguments": "{}"},
        },
    ]

    independent = executor.identify_independent_calls(tool_calls)
    dependent = executor.identify_dependent_calls(tool_calls)

    print(f"Independent calls: {independent}")
    print(f"Dependent calls: {dependent}")

    dependency_graph = executor.build_dependency_graph(tool_calls)
    print(f"Dependency graph nodes: {len(dependency_graph)}")

    batches = executor.topological_sort(dependency_graph)
    print(f"Execution batches: {len(batches)}")
    for i, batch in enumerate(batches):
        print(f"  Batch {i}: {batch}")

    assert len(independent) == 3
    assert len(dependent) == 0
    assert len(batches) == 1
    assert len(batches[0]) == 3

    print("✓ Dependency analysis test passed")


def test_error_handling():
    """Test error handling in parallel execution."""
    print("\n=== Test: Error Handling ===")

    class FailingSkill(BaseSkill):
        def __init__(self, name: str, should_fail: bool = False):
            super().__init__(name)
            self.should_fail = should_fail

        def execute(self, **kwargs) -> SkillResult:
            if self.should_fail:
                return SkillResult(success=False, error="Intentional failure")
            return SkillResult(success=True, data={"result": "success"})

        def validate(self) -> bool:
            return True

        def get_dependencies(self) -> list:
            return []

        def get_tool_schema(self) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": f"Skill {self.name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }

    skills = {
        "good_skill": FailingSkill("good_skill", should_fail=False),
        "bad_skill": FailingSkill("bad_skill", should_fail=True),
        "another_good": FailingSkill("another_good", should_fail=False),
    }

    executor = ParallelExecutor(skills=skills)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "good_skill", "arguments": "{}"},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bad_skill", "arguments": "{}"},
        },
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "another_good", "arguments": "{}"},
        },
    ]

    result = executor.execute_parallel(tool_calls)

    print(f"Overall success: {result.success}")
    print(f"Errors: {len(result.errors)}")
    print(f"Results: {len(result.results)}")

    success_count = sum(1 for r in result.results if r.get("success"))
    print(f"Successful calls: {success_count}/{len(result.results)}")

    assert not result.success
    assert success_count == 2
    assert len(result.results) == 3

    print("✓ Error handling test passed")


def test_sequential_fallback():
    """Test fallback to sequential execution."""
    print("\n=== Test: Sequential Fallback ===")

    skills = {"test": MockSkill("test", delay=0.05)}

    config = ParallelExecutorConfig(max_parallel_calls=1, enable_parallel=False)
    executor = ParallelExecutor(skills=skills, config=config)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        },
    ]

    result = executor.execute_parallel(tool_calls)

    print(f"Execution mode: {result.metadata.get('execution_mode')}")
    print(f"Speedup: {result.speedup}")

    assert result.metadata.get("execution_mode") == "sequential"
    assert result.speedup == 1.0

    print("✓ Sequential fallback test passed")


def test_performance_tracking():
    """Test performance metrics tracking."""
    print("\n=== Test: Performance Tracking ===")

    skills = {
        "skill1": MockSkill("skill1", delay=0.1),
        "skill2": MockSkill("skill2", delay=0.1),
    }

    executor = ParallelExecutor(skills=skills)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "skill1", "arguments": "{}"},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "skill2", "arguments": "{}"},
        },
    ]

    executor.execute_parallel(tool_calls)
    executor.execute_parallel(tool_calls)
    executor.execute_parallel(tool_calls)

    stats = executor.get_performance_stats()

    print(f"Total executions: {stats.get('total_executions')}")
    print(f"Average speedup: {stats.get('avg_speedup'):.2f}x")
    print(f"Average execution time: {stats.get('avg_execution_time_ms'):.0f}ms")
    print(f"Average success rate: {stats.get('avg_success_rate'):.2%}")

    assert stats.get("total_executions") == 3
    assert stats.get("avg_speedup") > 1.0

    print("✓ Performance tracking test passed")


def test_configuration():
    """Test executor configuration."""
    print("\n=== Test: Configuration ===")

    skills = {"test": MockSkill("test")}

    config = ParallelExecutorConfig(
        max_parallel_calls=10,
        per_tool_parallel_limits={"test": 5},
        default_timeout_ms=60000,
        enable_parallel=True,
        track_performance=True,
        validate_thread_safety=True,
    )

    executor = ParallelExecutor(skills=skills, config=config)

    assert executor.config.max_parallel_calls == 10
    assert executor.config.per_tool_parallel_limits["test"] == 5
    assert executor.config.default_timeout_ms == 60000

    executor.update_config(max_parallel_calls=20, enable_parallel=False)

    assert executor.config.max_parallel_calls == 20
    assert executor.config.enable_parallel is False

    print("✓ Configuration test passed")


def run_all_tests():
    """Run all parallel executor tests."""
    print("\n" + "=" * 60)
    print("Running Parallel Executor Tests")
    print("=" * 60)

    try:
        test_dependency_analysis()
        test_configuration()
        test_basic_parallel_execution()
        test_error_handling()
        test_sequential_fallback()
        test_performance_tracking()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
