"""Example and test file for model configuration system.

This demonstrates:
1. Getting model configurations
2. Phase-specific settings
3. Capability detection
4. Configuration optimization
5. A/B testing
6. Integration with orchestrators
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_configs import (
    ExecutionPhase,
    ModelCapability,
    ModelConfig,
    ModelConfigRegistry,
    ModelConfigStorage,
    ModelConfigOptimizer,
    get_model_config,
    get_temperature_for_phase,
    get_max_tokens_for_phase,
    get_prompt_template,
    detect_model_capabilities,
    validate_config,
    create_ab_test_config,
    integrate_with_orchestrator,
    get_llm_params_for_phase,
    load_config_from_env,
    registry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_config_usage():
    """Example: Basic model configuration usage."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Model Configuration Usage")
    print("=" * 60)

    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "glm-4", "gemini-pro"]

    for model in models:
        config = get_model_config(model)
        print(f"\n{config.display_name} ({config.model_name}):")
        print(f"  Family: {config.model_family}")
        print(f"  Max Iterations: {config.max_iterations}")
        print(f"  Max Parallel Calls: {config.max_parallel_calls}")
        print(f"  Context Window: {config.context_window}")
        print(f"  Capabilities: {[c.value for c in config.capabilities]}")


def example_phase_specific_settings():
    """Example: Phase-specific temperature and token settings."""
    print("\n" + "=" * 60)
    print("Example 2: Phase-Specific Settings")
    print("=" * 60)

    model = "gpt-4"
    phases = [
        ExecutionPhase.PLANNING,
        ExecutionPhase.EXECUTION,
        ExecutionPhase.REFLECTION,
        ExecutionPhase.SUMMARIZATION,
    ]

    print(f"\n{model} phase settings:")
    for phase in phases:
        temp = get_temperature_for_phase(model, phase)
        tokens = get_max_tokens_for_phase(model, phase)
        print(f"  {phase.value:15s}: temp={temp:.1f}, max_tokens={tokens}")


def example_capability_detection():
    """Example: Model capability detection."""
    print("\n" + "=" * 60)
    print("Example 3: Model Capability Detection")
    print("=" * 60)

    models = ["gpt-4-turbo", "claude-3-haiku", "llama-2", "mistral"]

    for model in models:
        capabilities = detect_model_capabilities(model)
        print(f"\n{model}:")
        for cap in sorted(capabilities, key=lambda x: x.value):
            print(f"  ✓ {cap.value}")


def example_llm_parameters():
    """Example: Getting complete LLM parameters for a phase."""
    print("\n" + "=" * 60)
    print("Example 4: LLM Parameters for Execution")
    print("=" * 60)

    model = "claude-3-opus"
    phase = ExecutionPhase.PLANNING

    params = get_llm_params_for_phase(model, phase)
    print(f"\n{model} parameters for {phase.value} phase:")
    for key, value in params.items():
        print(f"  {key}: {value}")


def example_custom_config():
    """Example: Creating and using custom model configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Model Configuration")
    print("=" * 60)

    custom_config = ModelConfig(
        model_name="custom-model-v1",
        model_family="custom",
        display_name="Custom Model V1",
        capabilities={
            ModelCapability.TOOL_CALLING,
            ModelCapability.REASONING,
            ModelCapability.CODE_GENERATION,
        },
        max_iterations=20,
        max_parallel_calls=8,
        context_window=65536,
        special_instructions="Custom model with optimized settings for code generation tasks.",
        optimization_tips=[
            "Use lower temperatures for factual code generation",
            "Enable reflection for complex algorithms",
            "Leverage parallel execution for independent tasks",
        ],
    )

    print("\nCustom configuration:")
    print(f"  Model: {custom_config.display_name}")
    print(f"  Max Iterations: {custom_config.max_iterations}")
    print(f"  Special Instructions: {custom_config.special_instructions}")

    is_valid, errors = validate_config(custom_config)
    if is_valid:
        print(f"  ✓ Configuration is valid")
    else:
        print(f"  ✗ Configuration errors: {errors}")


def example_config_storage():
    """Example: Saving and loading configurations."""
    print("\n" + "=" * 60)
    print("Example 6: Configuration Storage")
    print("=" * 60)

    storage = ModelConfigStorage(storage_path="config/model_configs")

    config = get_model_config("gpt-4")
    filepath = storage.save_config(config, version="v1.0.0")
    print(f"\nSaved config to: {filepath}")

    loaded_config = storage.load_config("gpt-4", version="v1.0.0")
    if loaded_config:
        print(
            f"Loaded config: {loaded_config.display_name} (v{loaded_config.config_version})"
        )

    versions = storage.list_versions("gpt-4")
    print(f"Available versions: {versions}")


def example_performance_tracking():
    """Example: Performance tracking and optimization."""
    print("\n" + "=" * 60)
    print("Example 7: Performance Tracking and Optimization")
    print("=" * 60)

    optimizer = ModelConfigOptimizer()

    model = "gpt-4"
    phase = ExecutionPhase.EXECUTION

    print(f"\nTracking performance for {model}...")

    for i in range(15):
        optimizer.track_performance(
            model_name=model,
            phase=phase,
            temperature=0.7,
            max_tokens=2000,
            success=(i % 3 != 0),
            duration_ms=1000 + i * 50,
            iterations=3 + (i % 2),
        )

    optimal = optimizer.get_optimal_settings(model, phase, min_samples=10)
    if optimal:
        print(f"\nOptimal settings for {phase.value}:")
        print(f"  Temperature: {optimal['temperature']}")
        print(f"  Max Tokens: {optimal['max_tokens']}")
        print(f"  Avg Duration: {optimal['avg_duration_ms']}ms")
        print(f"  Sample Count: {optimal['sample_count']}")


def example_ab_testing():
    """Example: Creating A/B test configurations."""
    print("\n" + "=" * 60)
    print("Example 8: A/B Testing Setup")
    print("=" * 60)

    base_config = get_model_config("gpt-4")

    print("\nBase configuration:")
    print(
        f"  Planning temp: {base_config.get_temperature(ExecutionPhase.PLANNING):.2f}"
    )
    print(
        f"  Execution temp: {base_config.get_temperature(ExecutionPhase.EXECUTION):.2f}"
    )

    variant_a = create_ab_test_config(
        base_config,
        variant_name="lower-temp",
        changes={
            "phase_settings": {
                ExecutionPhase.PLANNING.value: {"temperature": 0.2},
                ExecutionPhase.EXECUTION.value: {"temperature": 0.5},
            }
        },
    )

    variant_b = create_ab_test_config(
        base_config,
        variant_name="higher-temp",
        changes={
            "phase_settings": {
                ExecutionPhase.PLANNING.value: {"temperature": 0.4},
                ExecutionPhase.EXECUTION.value: {"temperature": 0.9},
            }
        },
    )

    print("\nVariant A (lower-temp):")
    print(f"  Planning temp: {variant_a.get_temperature(ExecutionPhase.PLANNING):.2f}")
    print(
        f"  Execution temp: {variant_a.get_temperature(ExecutionPhase.EXECUTION):.2f}"
    )

    print("\nVariant B (higher-temp):")
    print(f"  Planning temp: {variant_b.get_temperature(ExecutionPhase.PLANNING):.2f}")
    print(
        f"  Execution temp: {variant_b.get_temperature(ExecutionPhase.EXECUTION):.2f}"
    )


def example_registry_operations():
    """Example: Registry operations."""
    print("\n" + "=" * 60)
    print("Example 9: Registry Operations")
    print("=" * 60)

    print("\nAll registered models:")
    for model in registry.list_models():
        print(f"  - {model}")

    print("\nModels by family:")
    for family in ["gpt", "claude", "glm", "gemini"]:
        models = registry.get_models_by_family(family)
        print(f"\n  {family}:")
        for config in models:
            print(f"    - {config.display_name}")


def example_environment_config():
    """Example: Loading configuration from environment."""
    print("\n" + "=" * 60)
    print("Example 10: Environment Configuration")
    print("=" * 60)

    print("\nTo configure via environment variables, set:")
    print("  VIBEAGENT_MODEL_DEFAULT=gpt-4")
    print("  VIBEAGENT_MODEL_TEMPERATURE_PLANNING=0.3")
    print("  VIBEAGENT_MODEL_TEMPERATURE_EXECUTION=0.7")
    print("  VIBEAGENT_MODEL_MAX_TOKENS=3000")

    env_config = load_config_from_env()
    if env_config:
        print(f"\nLoaded {len(env_config)} configuration values from environment")
        for key, value in env_config.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo environment configuration found")


def example_integration_with_orchestrator():
    """Example: Integration with orchestrator (mock)."""
    print("\n" + "=" * 60)
    print("Example 11: Integration with Orchestrator")
    print("=" * 60)

    class MockOrchestrator:
        def __init__(self):
            self.llm_skill = type("obj", (object,), {"model": "gpt-4"})()
            self.react_config = {}
            self.parallel_executor = type(
                "obj",
                (object,),
                {"config": type("obj", (object,), {"max_parallel_calls": 0})()},
            )()

    orchestrator = MockOrchestrator()

    config = integrate_with_orchestrator(orchestrator, "gpt-4")

    print(f"\nIntegrated configuration for {config.model_name}:")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Max parallel calls: {config.max_parallel_calls}")
    print(f"  Supports function calling: {config.supports_function_calling}")


def example_complete_workflow():
    """Example: Complete workflow with model configuration."""
    print("\n" + "=" * 60)
    print("Example 12: Complete Workflow")
    print("=" * 60)

    model = "gpt-4-turbo"

    print(f"\nSetting up workflow for {model}...")

    config = get_model_config(model)
    print(f"\n1. Loaded configuration: {config.display_name}")

    storage = ModelConfigStorage()
    storage.save_config(config, version="workflow-v1")
    print(f"2. Saved configuration to storage")

    optimizer = ModelConfigOptimizer()

    for phase in [
        ExecutionPhase.PLANNING,
        ExecutionPhase.EXECUTION,
        ExecutionPhase.REFLECTION,
    ]:
        params = get_llm_params_for_phase(model, phase)
        print(f"\n3. Parameters for {phase.value}:")
        print(f"   Temperature: {params['temperature']}")
        print(f"   Max Tokens: {params['max_tokens']}")

        optimizer.track_performance(
            model_name=model,
            phase=phase,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            success=True,
            duration_ms=1500,
            iterations=5,
        )

    print(f"\n4. Tracked performance for {len(optimizer.performance_history)} phases")

    suggestions = optimizer.suggest_optimizations(config)
    print(f"\n5. Optimization suggestions:")
    for suggestion in suggestions:
        print(f"   - {suggestion}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Model Configuration System - All Examples")
    print("=" * 60)

    examples = [
        example_basic_config_usage,
        example_phase_specific_settings,
        example_capability_detection,
        example_llm_parameters,
        example_custom_config,
        example_config_storage,
        example_performance_tracking,
        example_ab_testing,
        example_registry_operations,
        example_environment_config,
        example_integration_with_orchestrator,
        example_complete_workflow,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
