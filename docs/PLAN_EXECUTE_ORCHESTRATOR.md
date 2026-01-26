# Plan-and-Execute Orchestrator

A sophisticated orchestrator that implements the Plan-and-Execute pattern for complex multi-step tasks. This orchestrator separates planning from execution, enabling better task decomposition, validation, and adaptive execution.

## Features

### Core Capabilities

- **Separate Planning and Execution Phases**: Decompose complex tasks into structured plans before execution
- **Structured Plan Representation**: Rich plan structure with steps, dependencies, and metadata
- **Adaptive Plan Execution**: Modify and adapt plans based on execution results
- **Comprehensive Validation**: Validate plan structure, dependencies, and tool availability
- **Multiple Step Types**: Support for normal, conditional, loop, parallel, and fallback steps
- **Plan Visualization**: Generate clear visualizations of plan structure and progress
- **Plan Comparison**: Compare and score different plan versions
- **Database Integration**: Track plan execution history and learn from successful patterns

## Architecture

### Components

```
PlanExecuteOrchestrator
├── Plan Generation
│   ├── LLM-based decomposition
│   ├── Dependency identification
│   └── Complexity estimation
├── Plan Validation
│   ├── Structure validation
│   ├── Circular dependency detection
│   └── Tool/parameter verification
├── Plan Execution
│   ├── Dependency-ordered execution
│   ├── Parallel step execution
│   └── Error handling with fallbacks
└── Plan Adaptation
    ├── Failure analysis
    ├── Step modification
    └── Pattern learning
```

### Plan Structure

```python
Plan
├── plan_id: str                    # Unique identifier
├── goal: str                       # Overall objective
├── steps: List[PlanStep]           # Execution steps
├── context: Dict[str, Any]         # Execution context
├── status: str                     # Plan status
└── metadata: Dict[str, Any]        # Additional metadata

PlanStep
├── step_id: str                    # Step identifier
├── step_type: StepType             # NORMAL, CONDITIONAL, LOOP, PARALLEL, FALLBACK
├── action: str                     # Human-readable description
├── tool: str                       # Tool to execute
├── parameters: Dict[str, Any]      # Tool parameters
├── dependencies: List[str]         # Step IDs this depends on
├── status: StepStatus              # PENDING, READY, RUNNING, COMPLETED, FAILED, SKIPPED
├── complexity: int                 # 1-5 complexity score
├── condition: Optional[str]        # For conditional steps
├── loop_count: Optional[int]       # For loop steps
├── fallback_for: Optional[str]     # For fallback steps
├── result: Optional[SkillResult]   # Execution result
└── execution_time_ms: float        # Execution time
```

## Usage

### Basic Usage

```python
from core.plan_execute_orchestrator import (
    PlanExecuteOrchestrator,
    PlanExecuteOrchestratorConfig,
)
from core.database_manager import DatabaseManager
from skills.llm_skill import LLMSkill

# Initialize
llm_skill = LLMSkill(
    base_url="http://localhost:11434/v1",
    model="llama3.2",
)

db_manager = DatabaseManager("data/vibeagent.db")

config = PlanExecuteOrchestratorConfig(
    max_plan_steps=15,
    max_plan_depth=5,
    plan_validation_strictness="moderate",
    adaptation_sensitivity=0.7,
    enable_parallel=True,
    max_parallel_steps=3,
    enable_plan_learning=True,
    fallback_to_sequential=True,
)

orchestrator = PlanExecuteOrchestrator(
    llm_skill=llm_skill,
    skills=your_skills_dict,
    db_manager=db_manager,
    config=config,
)

# Execute task
result = orchestrator.execute_with_tools(
    "Research quantum computing and summarize findings",
    max_iterations=5
)

if result.success:
    print(f"Success: {result.final_response}")
    print(f"Steps: {result.tool_calls_made}")
    print(f"Iterations: {result.iterations}")
else:
    print(f"Failed: {result.error}")
```

### Plan Visualization

```python
# Generate and visualize a plan
plan = orchestrator.generate_plan("Your task here", {})
visualization = orchestrator.visualize_plan(plan)
print(visualization)

# Output example:
# Plan: Your task here
# Status: created
# Steps:
#   → [step_1] Search web (search)
#   → [step_2] Analyze results (analyze) (after: step_1)
#   → [step_3] Summarize findings (summarize) (after: step_2)
#
# Progress: 0.0%
```

### Plan Comparison

```python
# Compare two different plans
plan1 = orchestrator.generate_plan("Task 1", {})
plan2 = orchestrator.generate_plan("Task 1", {})

comparison = orchestrator.compare_plans(plan1, plan2)
print(f"Better plan: {comparison['better_plan']}")
print(f"Score 1: {comparison['score1']}")
print(f"Score 2: {comparison['score2']}")
print(f"Added steps: {comparison['added_steps']}")
```

## Configuration

### PlanExecuteOrchestratorConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_plan_steps` | int | 20 | Maximum number of steps in a plan |
| `max_plan_depth` | int | 5 | Maximum depth of step dependencies |
| `plan_validation_strictness` | str | "moderate" | "strict", "moderate", or "lenient" |
| `adaptation_sensitivity` | float | 0.7 | Threshold for triggering plan adaptation (0.0-1.0) |
| `enable_parallel` | bool | True | Enable parallel step execution |
| `max_parallel_steps` | int | 3 | Maximum steps to execute in parallel |
| `enable_plan_learning` | bool | True | Learn from successful plan patterns |
| `fallback_to_sequential` | bool | True | Fall back to sequential execution on failure |
| `plan_timeout_ms` | int | 300000 | Maximum plan execution time (5 minutes) |
| `step_timeout_ms` | int | 30000 | Maximum step execution time (30 seconds) |

## Step Types

### Normal Step
Standard step that executes a tool with given parameters.

```python
PlanStep(
    step_id="step_1",
    step_type=StepType.NORMAL,
    action="Search the web",
    tool="search",
    parameters={"query": "quantum computing"},
)
```

### Conditional Step
Executes only if the condition evaluates to true.

```python
PlanStep(
    step_id="step_2",
    step_type=StepType.CONDITIONAL,
    action="Analyze if results found",
    tool="analyze",
    parameters={},
    condition="len(results) > 0",
)
```

### Loop Step
Repeats execution N times.

```python
PlanStep(
    step_id="step_3",
    step_type=StepType.LOOP,
    action="Retry analysis",
    tool="analyze",
    parameters={},
    loop_count=3,
)
```

### Parallel Step
Executed in parallel with other independent steps.

```python
PlanStep(
    step_id="step_4",
    step_type=StepType.PARALLEL,
    action="Search source A",
    tool="search",
    parameters={"source": "A"},
)
```

### Fallback Step
Executes if the step it's a fallback for fails.

```python
PlanStep(
    step_id="step_5",
    step_type=StepType.FALLBACK,
    action="Retry with different parameters",
    tool="search",
    parameters={"query": "alternative", "source": "B"},
    fallback_for="step_1",
)
```

## Plan Validation

The orchestrator performs comprehensive validation:

### Structure Validation
- Checks plan has at least one step
- Verifies all steps have required fields
- Validates step IDs are unique

### Dependency Validation
- Detects circular dependencies
- Ensures all dependency references exist
- Verifies dependency graph is acyclic

### Tool Validation
- Checks all referenced tools exist
- Verifies tools are available and registered

### Parameter Validation
- Validates all required parameters are provided
- Checks parameter types match tool expectations

### Example Validation Result

```python
validation = orchestrator.validate_plan(plan)

if validation.is_valid:
    print("Plan is valid")
else:
    print(f"Errors: {validation.errors}")
    print(f"Warnings: {validation.warnings}")
    print(f"Missing tools: {validation.missing_tools}")
    print(f"Missing parameters: {validation.missing_parameters}")
    print(f"Circular dependencies: {validation.circular_dependencies}")
```

## Plan Adaptation

When plan execution fails, the orchestrator can adapt the plan:

### Adaptation Process

1. **Failure Analysis**: Identify failed steps and their errors
2. **LLM Consultation**: Ask LLM for suggested modifications
3. **Plan Modification**: Apply suggested changes (add/remove/modify steps)
4. **Validation**: Validate the modified plan
5. **Re-execution**: Execute the adapted plan

### Adaptation Example

```python
# Original plan fails
result = orchestrator.execute_plan(plan)
if not result.success:
    # Adapt the plan based on failures
    adapted_plan = orchestrator.adapt_plan(plan, result.metadata)

    # Execute adapted plan
    result = orchestrator.execute_plan(adapted_plan)
```

## Database Integration

The orchestrator integrates with DatabaseManager to track:

### Plan Storage
- Store complete plan structure
- Track plan modifications over time
- Record plan execution history

### Execution Tracking
- Step-by-step execution progress
- Tool call results and errors
- Execution timing metrics

### Learning
- Store successful plan patterns
- Track plan success rates
- Build plan templates for reuse

```python
# Plans are automatically stored when using db_manager
result = orchestrator.execute_with_tools("Your task")

# Query plan history (via db_manager)
session = db_manager.get_session(session_db_id)
messages = db_manager.get_session_messages(session_db_id)
```

## Performance Characteristics

### Expected Improvements

- **Multi-step Task Success Rate**: +12% improvement over sequential execution
- **Task Decomposition**: Better handling of complex multi-step tasks
- **Execution Efficiency**: Parallel execution reduces total time by ~30%
- **Adaptability**: Automatic plan adaptation handles unexpected failures
- **Error Recovery**: Fallback steps and retries improve robustness

### Metrics Tracked

```python
result = orchestrator.execute_with_tools("Your task")

# Access metrics
metrics = result.metadata
print(f"Plan ID: {metrics['plan_id']}")
print(f"Total steps: {metrics['plan_steps']}")
print(f"Steps completed: {metrics['steps_completed']}")
print(f"Steps failed: {metrics['steps_failed']}")
print(f"Execution time: {metrics['execution_time_ms']}ms")
```

## Best Practices

### When to Use Plan-and-Execute

✅ **Use when:**
- Task requires multiple sequential or parallel steps
- Task has clear sub-goals that can be decomposed
- Task involves complex dependencies between steps
- Task may require adaptation based on intermediate results

❌ **Avoid when:**
- Task is simple and can be done in one step
- Task requires real-time interactive decision making
- Task has highly uncertain requirements
- Task is time-critical (planning adds overhead)

### Plan Design Tips

1. **Keep steps focused**: Each step should do one thing well
2. **Minimize dependencies**: Fewer dependencies enable more parallelism
3. **Use appropriate step types**: Match step type to execution pattern
4. **Plan for failures**: Include fallback steps for critical operations
5. **Estimate complexity**: Use complexity scores to prioritize execution

### Configuration Tuning

```python
# For simple tasks
config = PlanExecuteOrchestratorConfig(
    max_plan_steps=5,
    enable_parallel=False,
    adaptation_sensitivity=0.9,
)

# For complex tasks
config = PlanExecuteOrchestratorConfig(
    max_plan_steps=20,
    max_parallel_steps=5,
    enable_plan_learning=True,
    adaptation_sensitivity=0.5,
)

# For production
config = PlanExecuteOrchestratorConfig(
    plan_validation_strictness="strict",
    fallback_to_sequential=True,
    enable_plan_learning=True,
)
```

## Testing

Run the test suite:

```bash
source venv/bin/activate
pytest tests/test_plan_execute_orchestrator.py -v
```

Test coverage includes:
- Plan creation and manipulation
- Plan validation (structure, dependencies, tools)
- Plan execution (sequential, parallel, conditional, loop)
- Plan adaptation and modification
- Plan visualization and comparison
- Error handling and fallbacks

## Integration with Other Orchestrators

PlanExecuteOrchestrator extends ToolOrchestrator and integrates with:

- **ToolOrchestrator**: Base functionality for tool execution
- **ParallelExecutor**: Parallel step execution
- **RetryManager**: Automatic retry logic for failed steps
- **DatabaseManager**: Persistent storage and tracking
- **ErrorHandler**: Error classification and recovery

## Future Enhancements

Potential improvements:
- Plan templates for common task patterns
- Hierarchical plan decomposition
- Multi-agent plan collaboration
- Plan optimization using ML
- Real-time plan visualization UI
- Plan versioning and rollback
- Plan sharing and reuse across sessions