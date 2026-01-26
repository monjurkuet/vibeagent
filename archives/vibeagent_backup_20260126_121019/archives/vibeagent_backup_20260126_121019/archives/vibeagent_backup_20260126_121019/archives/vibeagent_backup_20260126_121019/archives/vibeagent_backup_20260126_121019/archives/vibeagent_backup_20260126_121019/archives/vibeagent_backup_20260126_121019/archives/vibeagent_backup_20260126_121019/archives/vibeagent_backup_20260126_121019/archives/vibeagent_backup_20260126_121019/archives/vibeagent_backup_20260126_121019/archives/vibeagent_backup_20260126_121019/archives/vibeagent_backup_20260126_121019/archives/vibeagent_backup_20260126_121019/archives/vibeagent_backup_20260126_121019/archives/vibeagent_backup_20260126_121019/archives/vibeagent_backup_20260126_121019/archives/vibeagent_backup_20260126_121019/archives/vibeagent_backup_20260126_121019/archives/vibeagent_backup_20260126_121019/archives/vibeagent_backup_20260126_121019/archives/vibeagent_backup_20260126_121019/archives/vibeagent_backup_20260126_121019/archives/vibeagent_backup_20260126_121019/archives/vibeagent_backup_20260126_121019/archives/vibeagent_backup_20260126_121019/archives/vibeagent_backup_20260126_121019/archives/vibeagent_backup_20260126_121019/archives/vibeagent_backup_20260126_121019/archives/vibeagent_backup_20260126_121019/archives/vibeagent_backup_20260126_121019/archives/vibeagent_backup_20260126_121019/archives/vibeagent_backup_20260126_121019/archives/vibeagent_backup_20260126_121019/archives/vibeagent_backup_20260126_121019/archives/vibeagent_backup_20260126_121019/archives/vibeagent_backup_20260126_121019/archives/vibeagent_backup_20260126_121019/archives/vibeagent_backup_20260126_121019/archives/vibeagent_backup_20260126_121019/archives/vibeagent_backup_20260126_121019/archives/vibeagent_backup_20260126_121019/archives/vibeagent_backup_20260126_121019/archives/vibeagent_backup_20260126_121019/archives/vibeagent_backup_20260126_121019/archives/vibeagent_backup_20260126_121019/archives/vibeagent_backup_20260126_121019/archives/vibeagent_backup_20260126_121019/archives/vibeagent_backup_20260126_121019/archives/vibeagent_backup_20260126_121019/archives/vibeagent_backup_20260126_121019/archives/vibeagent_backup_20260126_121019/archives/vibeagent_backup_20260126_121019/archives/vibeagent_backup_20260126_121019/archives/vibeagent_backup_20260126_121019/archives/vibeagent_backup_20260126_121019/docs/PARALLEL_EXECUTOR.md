# Parallel Tool Execution System

The Parallel Executor enables concurrent execution of independent tool calls, significantly reducing execution time for multi-tool tasks.

## Features

- **Automatic Dependency Detection**: Identifies which tool calls can run in parallel
- **Async Execution**: Uses asyncio for true concurrent execution
- **Safety Checks**: Validates thread-safety and prevents race conditions
- **Error Handling**: Continues execution on partial failures
- **Performance Tracking**: Monitors speedup and execution metrics
- **Database Integration**: Tracks parallel batches and performance metrics

## Architecture

### Core Components

#### ParallelExecutor
Main executor class that manages parallel tool execution.

```python
from core.parallel_executor import ParallelExecutor, ParallelExecutorConfig

executor = ParallelExecutor(
    skills=skills,
    db_manager=db_manager,
    config=ParallelExecutorConfig(
        max_parallel_calls=5,
        enable_parallel=True,
        track_performance=True,
    )
)
```

#### Dependency Analysis
- `identify_independent_calls()` - Finds parallel-safe calls
- `identify_dependent_calls()` - Finds calls with dependencies
- `build_dependency_graph()` - Builds execution graph
- `topological_sort()` - Determines execution order

#### Parallel Execution
- `execute_parallel()` - Main execution method
- `execute_batch()` - Executes a batch in parallel
- `execute_sequential()` - Fallback for sequential execution

## Configuration

```python
config = ParallelExecutorConfig(
    max_parallel_calls=5,              # Maximum concurrent calls
    per_tool_parallel_limits={},        # Per-tool limits
    default_timeout_ms=30000,           # Timeout per call
    enable_parallel=True,               # Enable/disable parallel
    track_performance=True,             # Track metrics
    validate_thread_safety=True,        # Safety checks
    resource_limit_cpu=None,            # CPU limit
    resource_limit_memory_mb=None,      # Memory limit
)
```

## Usage Examples

### Basic Parallel Execution

```python
tool_calls = [
    {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"query": "Python"}),
        },
    },
    {
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"query": "AsyncIO"}),
        },
    },
]

result = executor.execute_parallel(tool_calls)

print(f"Success: {result.success}")
print(f"Speedup: {result.speedup:.2f}x")
print(f"Results: {len(result.results)}")
```

### Integration with ToolOrchestrator

The ToolOrchestrator automatically uses parallel execution for multiple tool calls:

```python
orchestrator = ToolOrchestrator(
    llm_skill=llm,
    skills=skills,
    db_manager=db_manager,
)

result = orchestrator.execute_with_tools(
    "Search for Python and AsyncIO tutorials"
)
```

When the LLM requests multiple tool calls, they execute in parallel when safe.

## Performance

Expected speedup: **40-60%** reduction in execution time for multi-tool tasks.

### Example Performance

```
Sequential execution: 500ms (5 calls × 100ms each)
Parallel execution: 105ms (max of 5 parallel calls)
Speedup: 4.76x
```

## Safety Features

### Thread-Safety Validation

The executor validates that tools are safe for parallel execution:

- Checks for shared resource conflicts
- Validates per-tool parallel limits
- Prevents race conditions

### Error Handling

- Individual call failures don't stop execution
- Partial results are returned
- Detailed error reporting

```python
result = executor.execute_parallel(tool_calls)

if not result.success:
    print(f"Errors: {result.errors}")

for i, r in enumerate(result.results):
    if not r.get("success"):
        print(f"Call {i} failed: {r.get('error')}")
```

## Database Integration

Parallel execution tracks:

- **Parallel batch IDs** - Groups of parallel calls
- **Execution time** - Per-call timing
- **Performance metrics** - Speedup, efficiency
- **Error tracking** - Failed calls and reasons

```python
# Performance metrics stored in database
db_manager.add_performance_metric(
    session_id=session_id,
    metric_name="parallel_speedup",
    metric_value=4.76,
    metric_unit="x",
)
```

## Monitoring and Statistics

### Get Performance Stats

```python
stats = executor.get_performance_stats()

print(f"Total executions: {stats['total_executions']}")
print(f"Average speedup: {stats['avg_speedup']:.2f}x")
print(f"Average execution time: {stats['avg_execution_time_ms']:.0f}ms")
```

### Performance Metrics

- `total_executions` - Number of parallel executions
- `avg_speedup` - Average speedup vs sequential
- `avg_execution_time_ms` - Average execution time
- `avg_success_rate` - Average success rate

## Advanced Usage

### Custom Per-Tool Limits

```python
config = ParallelExecutorConfig(
    max_parallel_calls=10,
    per_tool_parallel_limits={
        "web_search": 5,      # Max 5 concurrent searches
        "file_read": 1,       # Sequential only
        "data_analysis": 3,   # Max 3 concurrent analyses
    },
)
```

### Dynamic Configuration

```python
executor.update_config(
    max_parallel_calls=10,
    enable_parallel=True,
)
```

### Dependency Management

The executor automatically detects dependencies:

```python
# Analyze dependencies
independent = executor.identify_independent_calls(tool_calls)
dependent = executor.identify_dependent_calls(tool_calls)

# Build dependency graph
graph = executor.build_dependency_graph(tool_calls)

# Get execution batches
batches = executor.topological_sort(graph)
# batches = [[0, 1, 2], [3, 4], [5]]
```

## Testing

Run the test suite:

```bash
python tests/test_parallel_executor.py
```

Tests cover:
- Basic parallel execution
- Dependency analysis
- Error handling
- Sequential fallback
- Performance tracking
- Configuration

## Best Practices

1. **Enable Parallel for I/O-Bound Tools**: Best for web searches, file operations, API calls
2. **Limit CPU-Bound Tools**: Use per-tool limits for heavy computations
3. **Monitor Performance**: Track speedup metrics to optimize configuration
4. **Handle Errors Gracefully**: Check individual call results
5. **Use Appropriate Timeouts**: Set timeouts based on expected tool duration

## Limitations

- Tools must be thread-safe for parallel execution
- Shared resources require sequential execution
- Maximum parallel calls limited by system resources
- Overhead of async execution for single calls

## Future Enhancements

- Adaptive parallelism based on tool performance
- Resource-aware scheduling
- Priority-based execution
- Distributed execution across workers
- Caching of repeated tool calls

## API Reference

### ParallelExecutor

```python
class ParallelExecutor:
    def __init__(
        self,
        skills: Dict[str, BaseSkill],
        db_manager=None,
        config: Optional[ParallelExecutorConfig] = None,
    )

    def execute_parallel(
        self,
        tool_calls: List[Dict],
        session_id: Optional[int] = None,
    ) -> ParallelExecutionResult

    def identify_independent_calls(
        self, tool_calls: List[Dict]
    ) -> List[int]

    def identify_dependent_calls(
        self, tool_calls: List[Dict]
    ) -> List[int]

    def build_dependency_graph(
        self, tool_calls: List[Dict]
    ) -> List[ToolCallInfo]

    def topological_sort(
        self, calls: List[ToolCallInfo]
    ) -> List[List[int]]

    def get_performance_stats(self) -> Dict

    def update_config(self, **kwargs)
```

### ParallelExecutionResult

```python
@dataclass
class ParallelExecutionResult:
    success: bool
    results: List[Dict]
    total_time_ms: float
    parallel_time_ms: float
    sequential_time_estimate_ms: float
    speedup: float
    batches: List[ParallelBatch]
    errors: List[str]
    metadata: Dict
```

## License

Part of VibeAgent project.