# Self-Correction System

A comprehensive self-correction system for intelligent error recovery and strategy adaptation in AI agent workflows.

## Overview

The SelfCorrector provides automated error detection, reflection, alternative strategy generation, and correction execution to improve agent reliability and success rates.

## Features

- **Smart Trigger Detection**: Automatically detects when self-correction is needed based on:
  - Repeated failures
  - Unexpected results
  - Confidence drops
  - Stagnation (no progress after N iterations)
  - Error pattern recognition

- **Reflection Engine**: Analyzes failures and results to:
  - Identify root causes
  - Generate reflection summaries
  - Evaluate result quality
  - Track confidence metrics

- **Alternative Strategy Generation**: Generates multiple correction strategies:
  - Different tool selection
  - Modified parameters
  - Different execution order
  - Retry with delay
  - Fallback strategies
  - Ask user for guidance

- **Pattern Learning**: Learns from past corrections to improve future decisions

- **Confidence Scoring**: Calculates and tracks confidence in current approach

- **Database Integration**: Tracks all correction attempts for analysis and learning

## Installation

The self-corrector is part of the core framework:

```python
from core.self_corrector import (
    SelfCorrector,
    SelfCorrectorConfig,
    CorrectionTrigger,
    CorrectionType,
)
```

## Quick Start

### Basic Usage

```python
from core.self_corrector import SelfCorrector, SelfCorrectorConfig
from core.skill import SkillResult

# Initialize with configuration
config = SelfCorrectorConfig(
    max_correction_attempts=3,
    consecutive_error_threshold=2,
    confidence_threshold=0.5,
)

corrector = SelfCorrector(
    llm_skill=llm_skill,  # Optional for LLM-guided corrections
    db_manager=db_manager,  # Optional for tracking
    config=config,
)

# Check if correction is needed
context = {
    "tool_result": SkillResult(success=False, error="Timeout"),
    "consecutive_errors": 2,
    "iteration": 3,
}

should_correct, trigger = corrector.should_self_correct(context)

if should_correct:
    # Reflect on the failure
    reflection = corrector.reflect_on_failure(
        error=context["tool_result"].error,
        context=context
    )

    # Generate alternative strategies
    alternatives = corrector.generate_alternatives(context)

    # Select best strategy
    best_strategy = corrector.select_best_strategy(alternatives)

    # Apply correction
    result = corrector.apply_correction(best_strategy, context)

    print(f"Correction {'succeeded' if result.success else 'failed'}")
```

### Integration with ToolOrchestrator

```python
from core.tool_orchestrator import ToolOrchestrator
from core.self_corrector import SelfCorrector

# Initialize orchestrator with self-corrector
orchestrator = ToolOrchestrator(
    llm_skill=llm_skill,
    skills=skills,
    db_manager=db_manager,
)

# Initialize self-corrector
self_corrector = SelfCorrector(
    llm_skill=llm_skill,
    db_manager=db_manager,
)

# In your execution loop:
while iterations < max_iterations:
    result = execute_tool(tool_call, parameters)

    context = {
        "tool_result": result,
        "tool_name": tool_name,
        "iteration": iterations,
        "consecutive_errors": consecutive_errors,
        "session_id": session_id,
    }

    # Check if self-correction is needed
    should_correct, trigger = self_corrector.should_self_correct(context)

    if should_correct and not result.success:
        # Generate and apply correction
        alternatives = self_corrector.generate_alternatives(context)
        best = self_corrector.select_best_strategy(alternatives)

        if best:
            correction_result = self_corrector.apply_correction(best, context)

            if correction_result.success:
                # Retry with corrected approach
                result = execute_tool(tool_call, corrected_parameters)
```

## Configuration

### SelfCorrectorConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_correction_attempts` | int | 3 | Maximum number of correction attempts |
| `consecutive_error_threshold` | int | 2 | Trigger correction after N consecutive errors |
| `stagnation_threshold` | int | 5 | Trigger correction after N iterations without progress |
| `confidence_threshold` | float | 0.5 | Minimum confidence before triggering correction |
| `confidence_drop_threshold` | float | 0.2 | Trigger correction if confidence drops by this amount |
| `enable_llm_guidance` | bool | True | Enable LLM-guided strategy generation |
| `enable_pattern_learning` | bool | True | Enable learning from past corrections |
| `strategy_diversity_factor` | float | 0.3 | Factor for strategy diversity |
| `min_confidence_for_correction` | float | 0.3 | Minimum confidence to auto-apply correction |
| `timeout_correction_ms` | int | 30000 | Default delay for timeout corrections |
| `reflection_depth` | int | 3 | Depth of reflection analysis |
| `auto_apply_confidence` | float | 0.7 | Confidence threshold for auto-apply |

## Correction Triggers

The system detects multiple trigger types:

- **REPEATED_FAILURES**: Multiple consecutive errors
- **UNEXPECTED_RESULT**: Result doesn't match expectations
- **CONFIDENCE_DROP**: Confidence in approach drops below threshold
- **STAGNATION**: No progress after N iterations
- **ERROR_PATTERN**: Recognized error pattern detected
- **TIMEOUT**: Operation timeout detected
- **VALIDATION_FAILURE**: Result validation failed

## Correction Types

Available correction strategies:

- **DIFFERENT_TOOL**: Switch to alternative tool
- **MODIFIED_PARAMETERS**: Adjust tool parameters
- **DIFFERENT_ORDER**: Change execution order
- **ASK_USER**: Request user guidance
- **RETRY_WITH_DELAY**: Retry with exponential backoff
- **FALLBACK_STRATEGY**: Use previously successful approach
- **SKIP_STEP**: Skip current operation
- **BREAK_DOWN_TASK**: Decompose task into smaller steps

## Error Pattern Recognition

The system automatically recognizes common error patterns:

- `network_error`: Connection issues, DNS failures
- `timeout_error`: Operation timeouts
- `rate_limit`: API rate limiting
- `invalid_input`: Validation failures
- `authentication_error`: Auth failures, invalid tokens
- `not_found`: Resource not found (404)
- `permission_denied`: Access denied (403)
- `internal_error`: Server errors (500)

## Confidence Scoring

The system maintains confidence metrics:

```python
# Calculate current confidence
confidence = corrector.calculate_confidence(context)

# Get confidence metrics
metrics = corrector._confidence_metrics
print(f"Current: {metrics.current_confidence}")
print(f"Success rate: {metrics.success_rate}")
print(f"Trend: {metrics.confidence_trend}")
```

## Tracking and Analytics

### Get Correction Statistics

```python
stats = corrector.get_correction_statistics()
print(f"Total attempts: {stats['total_attempts']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"By trigger: {stats['by_trigger']}")
print(f"By strategy: {stats['by_strategy']}")
```

### Database Tracking

All corrections are tracked in the database:

```python
# Corrections are stored in self_corrections table
# Query recent corrections:
corrections = db_manager.get_session_corrections(session_id)
```

## Pattern Learning

The system learns from successful corrections:

```python
# Enable pattern learning
config = SelfCorrectorConfig(enable_pattern_learning=True)

# System automatically improves strategy selection
# based on historical success rates
```

## API Reference

### SelfCorrector

Main self-correction class.

#### Methods

- `should_self_correct(context)`: Check if correction is needed
- `reflect_on_failure(error, context)`: Analyze failure
- `reflect_on_result(result, context)`: Evaluate result
- `generate_alternatives(context)`: Generate correction strategies
- `select_best_strategy(strategies)`: Select best strategy
- `apply_correction(correction, context)`: Apply correction
- `get_error_pattern(error)`: Identify error pattern
- `get_similar_past_solutions(error)`: Find similar solutions
- `score_strategy(strategy, context)`: Score strategy
- `calculate_confidence(context)`: Calculate confidence
- `get_correction_statistics()`: Get statistics
- `reset()`: Reset state

## Examples

See `examples/self_corrector_example.py` for comprehensive examples:

- Basic correction flow
- Error pattern detection
- Confidence scoring
- Learning from corrections
- Multiple triggers
- Strategy generation
- Integration with ToolOrchestrator

Run examples:

```bash
python3 examples/self_corrector_example.py
```

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_self_corrector.py -v
```

## Performance

The self-correction system is designed to:

- Handle 80% of errors automatically
- Improve success rate by 8-10%
- Learn from past corrections
- Provide transparent correction logs
- Integrate seamlessly with ReAct loop and retry manager

## Best Practices

1. **Start with conservative thresholds**: Begin with higher thresholds and adjust based on your use case
2. **Enable pattern learning**: Allow the system to learn from corrections over time
3. **Use LLM guidance**: Enable LLM-guided strategies for complex scenarios
4. **Monitor statistics**: Regularly review correction statistics to optimize
5. **Track corrections**: Use database tracking for long-term analysis
6. **Reset between sessions**: Call `reset()` between independent sessions

## Troubleshooting

### Too many corrections

Increase thresholds:
```python
config = SelfCorrectorConfig(
    consecutive_error_threshold=3,
    confidence_threshold=0.4,
)
```

### Corrections not triggering

Lower thresholds:
```python
config = SelfCorrectorConfig(
    consecutive_error_threshold=1,
    confidence_threshold=0.6,
)
```

### Poor strategy selection

Enable LLM guidance and pattern learning:
```python
config = SelfCorrectorConfig(
    enable_llm_guidance=True,
    enable_pattern_learning=True,
)
```

## License

Part of the VibeAgent framework.