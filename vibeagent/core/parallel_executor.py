"""Parallel tool execution system for concurrent tool calls."""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from .skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class ParallelExecutionStatus(Enum):
    """Status of parallel execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass
class ToolCallInfo:
    """Information about a tool call."""

    tool_call: Dict
    index: int
    dependencies: Set[int] = field(default_factory=set)
    dependents: Set[int] = field(default_factory=set)
    is_parallel_safe: bool = True
    execution_time_ms: float = 0
    result: Optional[SkillResult] = None
    error: Optional[str] = None


@dataclass
class ParallelBatch:
    """A batch of tool calls that can be executed in parallel."""

    batch_id: str
    call_indices: List[int]
    start_time: float = 0
    end_time: float = 0
    execution_time_ms: float = 0
    status: ParallelExecutionStatus = ParallelExecutionStatus.PENDING


@dataclass
class ParallelExecutionResult:
    """Result from parallel execution."""

    success: bool
    results: List[Dict]
    total_time_ms: float
    parallel_time_ms: float
    sequential_time_estimate_ms: float
    speedup: float
    batches: List[ParallelBatch]
    errors: List[str]
    metadata: Dict = field(default_factory=dict)


class ParallelExecutorConfig:
    """Configuration for parallel executor."""

    def __init__(
        self,
        max_parallel_calls: int = 5,
        per_tool_parallel_limits: Optional[Dict[str, int]] = None,
        default_timeout_ms: int = 30000,
        enable_parallel: bool = True,
        track_performance: bool = True,
        validate_thread_safety: bool = True,
        resource_limit_cpu: Optional[int] = None,
        resource_limit_memory_mb: Optional[int] = None,
    ):
        self.max_parallel_calls = max_parallel_calls
        self.per_tool_parallel_limits = per_tool_parallel_limits or {}
        self.default_timeout_ms = default_timeout_ms
        self.enable_parallel = enable_parallel
        self.track_performance = track_performance
        self.validate_thread_safety = validate_thread_safety
        self.resource_limit_cpu = resource_limit_cpu
        self.resource_limit_memory_mb = resource_limit_memory_mb


class ParallelExecutor:
    """Executor for parallel tool calls with dependency management."""

    def __init__(
        self,
        skills: Dict[str, BaseSkill],
        db_manager=None,
        config: Optional[ParallelExecutorConfig] = None,
    ):
        self.skills = skills
        self.db_manager = db_manager
        self.config = config or ParallelExecutorConfig()
        self._performance_history: List[Dict] = []

    def identify_independent_calls(self, tool_calls: List[Dict]) -> List[int]:
        """Find calls that can run in parallel (no dependencies).

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of indices of independent calls
        """
        if not tool_calls:
            return []

        call_info = self._analyze_tool_calls(tool_calls)
        independent_indices = [
            i for i, info in enumerate(call_info) if info.is_parallel_safe
        ]

        return independent_indices

    def identify_dependent_calls(self, tool_calls: List[Dict]) -> List[int]:
        """Find calls with dependencies that must run sequentially.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of indices of dependent calls
        """
        if not tool_calls:
            return []

        call_info = self._analyze_tool_calls(tool_calls)
        dependent_indices = [
            i for i, info in enumerate(call_info) if not info.is_parallel_safe
        ]

        return dependent_indices

    def build_dependency_graph(self, tool_calls: List[Dict]) -> List[ToolCallInfo]:
        """Build execution graph showing dependencies between calls.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of ToolCallInfo with dependency information
        """
        call_info = self._analyze_tool_calls(tool_calls)

        for i, info in enumerate(call_info):
            for j, other_info in enumerate(call_info):
                if i != j:
                    if self._has_dependency(info, other_info):
                        info.dependencies.add(j)
                        other_info.dependents.add(i)

        return call_info

    def topological_sort(self, calls: List[ToolCallInfo]) -> List[List[int]]:
        """Determine execution order using topological sort.

        Args:
            calls: List of ToolCallInfo with dependencies

        Returns:
            List of batches, where each batch contains indices that can run in parallel
        """
        in_degree = {i: len(c.dependencies) for i, c in enumerate(calls)}
        queue = [i for i, degree in in_degree.items() if degree == 0]
        batches = []
        current_batch = []

        while queue:
            current_batch = queue.copy()
            batches.append(current_batch)
            queue = []

            for node in current_batch:
                for dependent in calls[node].dependents:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if sum(len(batch) for batch in batches) != len(calls):
            logger.warning("Cycle detected in dependency graph, using sequential order")
            return [[i] for i in range(len(calls))]

        return batches

    def execute_parallel(
        self, tool_calls: List[Dict], session_id: Optional[int] = None
    ) -> ParallelExecutionResult:
        """Execute tool calls in parallel where possible.

        Args:
            tool_calls: List of tool call dictionaries
            session_id: Optional database session ID for tracking

        Returns:
            ParallelExecutionResult with aggregated results
        """
        if not tool_calls:
            return ParallelExecutionResult(
                success=True,
                results=[],
                total_time_ms=0,
                parallel_time_ms=0,
                sequential_time_estimate_ms=0,
                speedup=0,
                batches=[],
                errors=[],
            )

        if not self.config.enable_parallel:
            return self._execute_sequential(tool_calls, session_id)

        start_time = time.time()

        try:
            call_info = self.build_dependency_graph(tool_calls)
            batches = self.topological_sort(call_info)

            parallel_batches = []
            results = []
            all_errors = []

            for batch_idx, batch_indices in enumerate(batches):
                batch_id = str(uuid.uuid4())
                batch = ParallelBatch(batch_id=batch_id, call_indices=batch_indices)

                if len(batch_indices) == 1:
                    batch_result = self._execute_single(
                        tool_calls[batch_indices[0]], session_id
                    )
                    results.append(batch_result)
                else:
                    batch_results = self._execute_batch(
                        [tool_calls[i] for i in batch_indices],
                        batch_id,
                        session_id,
                    )
                    results.extend(batch_results)

                batch.end_time = time.time()
                batch.execution_time_ms = (batch.end_time - batch.start_time) * 1000
                batch.status = ParallelExecutionStatus.COMPLETED
                parallel_batches.append(batch)

            total_time_ms = (time.time() - start_time) * 1000

            sequential_estimate = self._estimate_sequential_time(tool_calls)
            speedup = (sequential_estimate / total_time_ms) if total_time_ms > 0 else 0

            result = ParallelExecutionResult(
                success=all(r.get("success", False) for r in results),
                results=results,
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_estimate_ms=sequential_estimate,
                speedup=speedup,
                batches=parallel_batches,
                errors=all_errors,
                metadata={
                    "total_calls": len(tool_calls),
                    "parallel_batches": len(batches),
                    "max_parallel": max(len(b) for b in batches),
                },
            )

            if self.config.track_performance:
                self._track_performance(result)

            if self.db_manager and session_id:
                self._store_parallel_metrics(session_id, result, parallel_batches)

            return result

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            return self._execute_sequential(tool_calls, session_id)

    def _execute_sequential(
        self, tool_calls: List[Dict], session_id: Optional[int] = None
    ) -> ParallelExecutionResult:
        """Execute tool calls sequentially as fallback.

        Args:
            tool_calls: List of tool call dictionaries
            session_id: Optional database session ID

        Returns:
            ParallelExecutionResult with sequential execution results
        """
        start_time = time.time()
        results = []
        errors = []

        for i, tool_call in enumerate(tool_calls):
            try:
                result = self._execute_single(tool_call, session_id)
                results.append(result)
            except Exception as e:
                error_msg = f"Tool call {i} failed: {str(e)}"
                errors.append(error_msg)
                results.append({"success": False, "error": error_msg})

        total_time_ms = (time.time() - start_time) * 1000

        return ParallelExecutionResult(
            success=all(r.get("success", False) for r in results),
            results=results,
            total_time_ms=total_time_ms,
            parallel_time_ms=total_time_ms,
            sequential_time_estimate_ms=total_time_ms,
            speedup=1.0,
            batches=[],
            errors=errors,
            metadata={"execution_mode": "sequential"},
        )

    def _execute_batch(
        self,
        tool_calls: List[Dict],
        batch_id: str,
        session_id: Optional[int] = None,
    ) -> List[Dict]:
        """Execute a batch of tool calls in parallel.

        Args:
            tool_calls: List of tool call dictionaries to execute
            batch_id: ID for tracking this batch
            session_id: Optional database session ID

        Returns:
            List of results in original order
        """
        if not tool_calls:
            return []

        start_time = time.time()

        async def run_all():
            tasks = []
            for tool_call in tool_calls:
                task = self._execute_single_async(tool_call, session_id)
                tasks.append(task)

            timeout = self.config.default_timeout_ms / 1000
            try:
                return await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Batch {batch_id} timed out")
                return tasks

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_all())
        finally:
            loop.close()

        ordered_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                ordered_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "index": i,
                        "batch_id": batch_id,
                    }
                )
            else:
                ordered_results.append(result)

        execution_time_ms = (time.time() - start_time) * 1000

        if self.db_manager and session_id:
            self._track_batch_execution(
                session_id, batch_id, tool_calls, ordered_results, execution_time_ms
            )

        return ordered_results

    async def _execute_single_async(
        self, tool_call: Dict, session_id: Optional[int] = None
    ) -> Dict:
        """Execute a single tool call asynchronously.

        Args:
            tool_call: Tool call dictionary
            session_id: Optional database session ID

        Returns:
            Result dictionary
        """
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name")
        arguments_str = function_info.get("arguments", "{}")

        if not function_name:
            return {"success": False, "error": "Missing function name"}

        skill = self.skills.get(function_name)
        if not skill:
            return {"success": False, "error": f"Tool '{function_name}' not found"}

        try:
            arguments = json.loads(arguments_str)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: skill.execute(**arguments)
            )

            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "tool_call": tool_call,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "tool_call": tool_call}

    def _execute_single(
        self, tool_call: Dict, session_id: Optional[int] = None
    ) -> Dict:
        """Execute a single tool call synchronously.

        Args:
            tool_call: Tool call dictionary
            session_id: Optional database session ID

        Returns:
            Result dictionary
        """
        start_time = time.time()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._execute_single_async(tool_call, session_id)
            )
        finally:
            loop.close()

        execution_time_ms = (time.time() - start_time) * 1000
        result["execution_time_ms"] = execution_time_ms

        return result

    def _analyze_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCallInfo]:
        """Analyze tool calls for parallel safety.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of ToolCallInfo with safety information
        """
        call_info = []

        for i, tool_call in enumerate(tool_calls):
            info = ToolCallInfo(tool_call=tool_call, index=i)

            function_info = tool_call.get("function", {})
            function_name = function_info.get("name")

            skill = self.skills.get(function_name)
            if skill:
                info.is_parallel_safe = self._is_parallel_safe(skill, tool_call)
            else:
                info.is_parallel_safe = True

            call_info.append(info)

        return call_info

    def _is_parallel_safe(self, skill: BaseSkill, tool_call: Dict) -> bool:
        """Check if a tool call is safe for parallel execution.

        Args:
            skill: Skill to execute
            tool_call: Tool call dictionary

        Returns:
            True if safe for parallel execution
        """
        if not self.config.validate_thread_safety:
            return True

        tool_limit = self.config.per_tool_parallel_limits.get(
            skill.name, self.config.max_parallel_calls
        )

        if tool_limit <= 1:
            return False

        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        arguments = json.loads(arguments_str)

        shared_resources = self._detect_shared_resources(skill, arguments)
        if shared_resources:
            return False

        return True

    def _detect_shared_resources(self, skill: BaseSkill, arguments: Dict) -> Set[str]:
        """Detect if tool call uses shared resources that could conflict.

        Args:
            skill: Skill being executed
            arguments: Tool call arguments

        Returns:
            Set of shared resource identifiers
        """
        shared_resources = set()

        common_shared_params = ["path", "file", "database", "collection", "table"]
        for param in common_shared_params:
            if param in arguments:
                shared_resources.add(f"{skill.name}:{param}:{arguments[param]}")

        return shared_resources

    def _has_dependency(self, call_a: ToolCallInfo, call_b: ToolCallInfo) -> bool:
        """Check if call_a depends on call_b.

        Args:
            call_a: First tool call info
            call_b: Second tool call info

        Returns:
            True if call_a depends on call_b
        """
        if not call_a.is_parallel_safe or not call_b.is_parallel_safe:
            return True

        function_a = call_a.tool_call.get("function", {}).get("name", "")
        function_b = call_b.tool_call.get("function", {}).get("name", "")

        if function_a == function_b:
            limit = self.config.per_tool_parallel_limits.get(
                function_a, self.config.max_parallel_calls
            )
            if limit <= 1:
                return True

        return False

    def _estimate_sequential_time(self, tool_calls: List[Dict]) -> float:
        """Estimate time for sequential execution.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Estimated time in milliseconds
        """
        base_time_per_call = 100
        return len(tool_calls) * base_time_per_call

    def _track_performance(self, result: ParallelExecutionResult):
        """Track performance metrics for analysis.

        Args:
            result: Parallel execution result
        """
        metrics = {
            "timestamp": time.time(),
            "total_calls": len(result.results),
            "total_time_ms": result.total_time_ms,
            "speedup": result.speedup,
            "batches": len(result.batches),
            "success_rate": sum(1 for r in result.results if r.get("success", False))
            / len(result.results)
            if result.results
            else 0,
        }

        self._performance_history.append(metrics)

        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

    def _track_batch_execution(
        self,
        session_id: int,
        batch_id: str,
        tool_calls: List[Dict],
        results: List[Dict],
        execution_time_ms: float,
    ):
        """Track batch execution in database.

        Args:
            session_id: Database session ID
            batch_id: Batch identifier
            tool_calls: Tool calls in batch
            results: Results from batch
            execution_time_ms: Execution time in milliseconds
        """
        if not self.db_manager:
            return

        try:
            for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
                function_info = tool_call.get("function", {})
                function_name = function_info.get("name")
                arguments_str = function_info.get("arguments", "{}")
                arguments = json.loads(arguments_str)

                tool_call_id = self.db_manager.add_tool_call(
                    session_id=session_id,
                    call_index=i,
                    tool_name=function_name,
                    parameters=arguments,
                    execution_time_ms=int(execution_time_ms / len(tool_calls)),
                    success=result.get("success", False),
                    error_message=result.get("error"),
                    error_type=None,
                    retry_count=0,
                    is_parallel=True,
                    parallel_batch_id=batch_id,
                    metadata={"batch_id": batch_id},
                )

                self.db_manager.add_tool_result(
                    tool_call_id=tool_call_id,
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error"),
                    result_size_bytes=(
                        len(json.dumps(result.get("data"))) if result.get("data") else 0
                    ),
                    metadata={"batch_id": batch_id},
                )
        except Exception as e:
            logger.error(f"Failed to track batch execution: {e}")

    def _store_parallel_metrics(
        self,
        session_id: int,
        result: ParallelExecutionResult,
        batches: List[ParallelBatch],
    ):
        """Store parallel execution metrics in database.

        Args:
            session_id: Database session ID
            result: Parallel execution result
            batches: Parallel execution batches
        """
        if not self.db_manager:
            return

        try:
            self.db_manager.add_performance_metric(
                session_id=session_id,
                metric_name="parallel_execution_time_ms",
                metric_value=result.parallel_time_ms,
                metric_unit="milliseconds",
                metadata={
                    "speedup": result.speedup,
                    "batches": len(batches),
                },
            )

            self.db_manager.add_performance_metric(
                session_id=session_id,
                metric_name="parallel_speedup",
                metric_value=result.speedup,
                metric_unit="x",
                metadata={
                    "sequential_estimate": result.sequential_time_estimate_ms,
                    "total_calls": len(result.results),
                },
            )
        except Exception as e:
            logger.error(f"Failed to store parallel metrics: {e}")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self._performance_history:
            return {}

        total_speedup = sum(m["speedup"] for m in self._performance_history)
        avg_speedup = total_speedup / len(self._performance_history)

        avg_time = sum(m["total_time_ms"] for m in self._performance_history) / len(
            self._performance_history
        )

        avg_success_rate = sum(
            m["success_rate"] for m in self._performance_history
        ) / len(self._performance_history)

        return {
            "avg_speedup": avg_speedup,
            "avg_execution_time_ms": avg_time,
            "avg_success_rate": avg_success_rate,
            "total_executions": len(self._performance_history),
            "config": {
                "max_parallel_calls": self.config.max_parallel_calls,
                "enable_parallel": self.config.enable_parallel,
            },
        }

    def update_config(self, **kwargs):
        """Update executor configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} to {value}")
