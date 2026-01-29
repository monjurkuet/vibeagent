# Model Configuration System

A comprehensive model-specific configuration system for optimal LLM interaction across different models and execution phases.

## Features

- **Model-Specific Configurations**: Pre-configured settings for GPT-4, GPT-3.5, Claude, Gemini, and local LLMs
- **Phase-Specific Settings**: Different temperature, token limits, and parameters for planning, execution, reflection, and summarization
- **Capability Detection**: Automatic detection of model capabilities (tool calling, reasoning, parallel execution, etc.)
- **Performance Tracking**: Track and optimize settings based on actual performance data
- **A/B Testing**: Create and test configuration variants
- **Configuration Storage**: Save, version, and load configurations from files
- **Integration**: Easy integration with ToolOrchestrator, PlanExecuteOrchestrator, and other orchestrators

## Quick Start

```python
from config.model_configs import (
    get_model_config,
    get_temperature_for_phase,
    get_max_tokens_for_phase,
    get_llm_params_for_phase,
    ExecutionPhase,
)

# Get configuration for a model
config = get_model_config("gpt-4")

# Get phase-specific settings
temp = get_temperature_for_phase("gpt-4", ExecutionPhase.PLANNING)  # 0.3
tokens = get_max_tokens_for_phase("gpt-4", ExecutionPhase.PLANNING)  # 3000

# Get complete LLM parameters
params = get_llm_params_for_phase("gpt-4", ExecutionPhase.PLANNING)
# Returns: {'model': 'gpt-4', 'temperature': 0.3, 'max_tokens': 3000, 'top_p': 0.9}
```

## Supported Models

### GPT Models
- **gpt-4**: Full capabilities, 128K context, 15 max iterations, 5 parallel calls
- **gpt-4-turbo**: Faster, same capabilities, 20 max iterations, 10 parallel calls
- **gpt-3.5-turbo**: Faster but less capable, 10 max iterations, 3 parallel calls

### Claude Models
- **claude-3-opus**: Highest capability, 200K context, 15 max iterations, 5 parallel calls
- **claude-3-sonnet**: Balanced capability, 200K context, 12 max iterations, 4 parallel calls
- **claude-3-haiku**: Fast for simple tasks, 200K context, 8 max iterations, 2 parallel calls

### Local LLMs
- **glm-4**: Capable but needs careful prompting, 32K context, 8 max iterations, 2 parallel calls
- **llama-2**: Requires explicit ReAct pattern, 4K context, 6 max iterations, 1 parallel call
- **mistral**: Efficient and capable, 8K context, 8 max iterations, 2 parallel calls

### Gemini Models
- **gemini-pro**: Strong multimodal, 32K context, 12 max iterations, 4 parallel calls
- **gemini-ultra**: Most capable, 32K context, 15 max iterations, 5 parallel calls

## Execution Phases

### Planning Phase
- **Temperature**: 0.3 (low for consistent planning)
- **Max Tokens**: 3000 (detailed plans)
- **Best For**: Task decomposition, strategy formation

### Execution Phase
- **Temperature**: 0.7 (balanced for action)
- **Max Tokens**: 2000 (standard responses)
- **Best For**: Tool calling, action execution

### Reflection Phase
- **Temperature**: 0.8 (higher for diverse thinking)
- **Max Tokens**: 2500 (detailed reasoning)
- **Best For**: Self-correction, error analysis

### Summarization Phase
- **Temperature**: 0.2 (very low for consistency)
- **Max Tokens**: 1000 (concise outputs)
- **Best For**: Result synthesis, final answers

### Error Recovery Phase
- **Temperature**: 0.5 (moderate for adaptation)
- **Max Tokens**: 1500 (detailed fixes)
- **Best For**: Error handling, retry strategies

### Validation Phase
- **Temperature**: 0.1 (very low for strict checking)
- **Max Tokens**: 1000 (precise validation)
- **Best For**: Result verification, quality checks

## Advanced Usage

### Custom Configuration

```python
from config.model_configs import (
    ModelConfig,
    ModelCapability,
    ExecutionPhase,
    PhaseSettings,
)

config = ModelConfig(
    model_name="custom-model",
    model_family="custom",
    display_name="Custom Model",
    capabilities={
        ModelCapability.TOOL_CALLING,
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
    },
    max_iterations=20,
    max_parallel_calls=8,
    special_instructions="Optimized for code generation tasks",
)

# Customize phase settings
config.phase_settings[ExecutionPhase.PLANNING] = PhaseSettings(
    temperature=0.2,
    max_tokens=4000,
    enable_reasoning=True,
)
```

### Performance Tracking

```python
from config.model_configs import ModelConfigOptimizer, ExecutionPhase

optimizer = ModelConfigOptimizer()

# Track performance
optimizer.track_performance(
    model_name="gpt-4",
    phase=ExecutionPhase.EXECUTION,
    temperature=0.7,
    max_tokens=2000,
    success=True,
    duration_ms=1500,
    iterations=5,
)

# Get optimal settings
optimal = optimizer.get_optimal_settings("gpt-4", ExecutionPhase.EXECUTION)
print(f"Optimal temperature: {optimal['temperature']}")
```

### A/B Testing

```python
from config.model_configs import create_ab_test_config, ExecutionPhase

base_config = get_model_config("gpt-4")

# Create variant with lower temperature
variant = create_ab_test_config(
    base_config,
    variant_name="conservative",
    changes={
        "phase_settings": {
            ExecutionPhase.PLANNING.value: {"temperature": 0.1},
            ExecutionPhase.EXECUTION.value: {"temperature": 0.5},
        }
    },
)
```

### Configuration Storage

```python
from config.model_configs import ModelConfigStorage

storage = ModelConfigStorage(storage_path="config/model_configs")

# Save configuration
filepath = storage.save_config(config, version="v1.0.0")

# Load configuration
loaded = storage.load_config("gpt-4", version="v1.0.0")

# List versions
versions = storage.list_versions("gpt-4")
```

### Integration with Orchestrators

```python
from config.model_configs import integrate_with_orchestrator

# Automatically integrate with any orchestrator
config = integrate_with_orchestrator(orchestrator, "gpt-4")

# This will:
# - Set model config on llm_skill
# - Update react_config max_reasoning_steps
# - Update parallel_executor max_parallel_calls
```

### Capability Detection

```python
from config.model_configs import detect_model_capabilities, ModelCapability

capabilities = detect_model_capabilities("gpt-4-turbo")

if ModelCapability.TOOL_CALLING in capabilities:
    print("Model supports tool calling")

if ModelCapability.PARALLEL_EXECUTION in capabilities:
    print("Model supports parallel execution")
```

## Environment Configuration

Configure models via environment variables:

```bash
export VIBEAGENT_MODEL_DEFAULT=gpt-4
export VIBEAGENT_MODEL_TEMPERATURE_PLANNING=0.3
export VIBEAGENT_MODEL_TEMPERATURE_EXECUTION=0.7
export VIBEAGENT_MODEL_MAX_TOKENS=3000
```

Load in code:

```python
from config.model_configs import load_config_from_env

env_config = load_config_from_env()
```

## Database Integration

The system includes database migrations for storing:

- Model configurations with versioning
- Performance metrics by configuration
- A/B test configurations and results
- Configuration optimization suggestions

Run migrations:

```bash
sqlite3 data/vibeagent.db < config/migrations/20240124130000_add_model_configs.sql
```

## API Reference

### Core Classes

- **ModelConfig**: Complete configuration for a model
- **ModelConfigRegistry**: Registry of predefined configurations
- **ModelConfigStorage**: File-based configuration storage
- **ModelConfigOptimizer**: Performance tracking and optimization
- **PhaseSettings**: Settings for a specific execution phase
- **RetryPolicy**: Retry policy configuration
- **PromptTemplate**: Reusable prompt templates

### Enums

- **ExecutionPhase**: Planning, Execution, Reflection, Summarization, Error Recovery, Validation
- **ModelCapability**: Tool Calling, Reasoning, Parallel Execution, Reflection, etc.

### Helper Functions

- `get_model_config(model_name)`: Get configuration for a model
- `get_temperature_for_phase(model_name, phase)`: Get phase-specific temperature
- `get_max_tokens_for_phase(model_name, phase)`: Get phase-specific max tokens
- `get_prompt_template(model_name, template_type)`: Get prompt template
- `detect_model_capabilities(model_name)`: Detect model capabilities
- `validate_config(config)`: Validate a configuration
- `create_ab_test_config(base_config, variant_name, changes)`: Create A/B test variant
- `integrate_with_orchestrator(orchestrator, model_name)`: Integrate with orchestrator
- `get_llm_params_for_phase(model_name, phase, additional_params)`: Get complete LLM parameters

## Examples

See `examples/model_config_example.py` for comprehensive examples:

1. Basic configuration usage
2. Phase-specific settings
3. Capability detection
4. LLM parameters
5. Custom configurations
6. Configuration storage
7. Performance tracking
8. A/B testing
9. Registry operations
10. Environment configuration
11. Orchestrator integration
12. Complete workflow

Run examples:

```bash
python examples/model_config_example.py
```

## Testing

Run tests:

```bash
python tests/test_model_configs.py -v
```

## Best Practices

1. **Use Phase-Specific Settings**: Always use phase-specific temperatures and token limits
2. **Track Performance**: Enable performance tracking to optimize over time
3. **Test Variants**: Use A/B testing to find optimal settings
4. **Version Configurations**: Save configuration versions for rollback
5. **Validate Before Use**: Always validate custom configurations
6. **Respect Model Limits**: Configure max_parallel_calls based on model capabilities
7. **Monitor Context Usage**: Adjust max_tokens based on context window size

## Contributing

To add a new model:

1. Create a new configuration method in `ModelConfigRegistry`
2. Add appropriate capabilities
3. Set optimal phase settings
4. Add special instructions and optimization tips
5. Test with the model
6. Add to documentation

## License

This module is part of the VibeAgent project.
