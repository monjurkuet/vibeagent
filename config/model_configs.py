"""Model-specific configuration system for optimal LLM interaction.

This module provides per-model configuration including:
- Temperature and token settings for different phases
- Prompt templates optimized for each model
- Capability detection and adaptation
- Performance tracking and optimization
- A/B testing support
- Configuration versioning and history
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Execution phases for different LLM operations."""

    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    SUMMARIZATION = "summarization"
    ERROR_RECOVERY = "error_recovery"
    VALIDATION = "validation"


class ModelCapability(Enum):
    """Model capabilities for feature detection."""

    TOOL_CALLING = "tool_calling"
    REASONING = "reasoning"
    PARALLEL_EXECUTION = "parallel_execution"
    REFLECTION = "reflection"
    CODE_GENERATION = "code_generation"
    LONG_CONTEXT = "long_context"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"


@dataclass
class PhaseSettings:
    """Settings for a specific execution phase."""

    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    top_k: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    response_format: str | None = None
    enable_reasoning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RetryPolicy:
    """Retry policy for model interactions."""

    max_retries: int = 3
    retry_on_errors: list[str] = field(
        default_factory=lambda: ["timeout", "rate_limit", "server_error"]
    )
    backoff_multiplier: float = 2.0
    initial_delay_ms: int = 1000
    max_delay_ms: int = 10000
    retry_on_validation_failure: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PromptTemplate:
    """Prompt template for a specific model."""

    name: str
    template: str
    description: str
    model_type: str = "default"
    template_type: str = "system"
    variables: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return self.template

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelConfig:
    """Complete configuration for a specific LLM model."""

    model_name: str
    model_family: str
    display_name: str

    phase_settings: dict[str, PhaseSettings] = field(default_factory=dict)
    prompt_templates: dict[str, PromptTemplate] = field(default_factory=dict)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    capabilities: set[ModelCapability] = field(default_factory=set)
    max_iterations: int = 10
    max_parallel_calls: int = 5
    context_window: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = True

    special_instructions: str = ""
    optimization_tips: list[str] = field(default_factory=list)

    performance_metrics: dict[str, Any] = field(default_factory=dict)
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Initialize default phase settings if not provided."""
        if not self.phase_settings:
            self.phase_settings = self._get_default_phase_settings()

    def _get_default_phase_settings(self) -> dict[str, PhaseSettings]:
        """Get default phase settings."""
        return {
            ExecutionPhase.PLANNING.value: PhaseSettings(
                temperature=0.3,
                max_tokens=3000,
                enable_reasoning=True,
            ),
            ExecutionPhase.EXECUTION.value: PhaseSettings(
                temperature=0.7,
                max_tokens=2000,
            ),
            ExecutionPhase.REFLECTION.value: PhaseSettings(
                temperature=0.8,
                max_tokens=2500,
                enable_reasoning=True,
            ),
            ExecutionPhase.SUMMARIZATION.value: PhaseSettings(
                temperature=0.2,
                max_tokens=1000,
            ),
            ExecutionPhase.ERROR_RECOVERY.value: PhaseSettings(
                temperature=0.5,
                max_tokens=1500,
            ),
            ExecutionPhase.VALIDATION.value: PhaseSettings(
                temperature=0.1,
                max_tokens=1000,
            ),
        }

    def get_temperature(self, phase: ExecutionPhase) -> float:
        """Get temperature for specific phase."""
        phase_key = phase.value
        if phase_key in self.phase_settings:
            return self.phase_settings[phase_key].temperature
        return 0.7

    def get_max_tokens(self, phase: ExecutionPhase) -> int:
        """Get max tokens for specific phase."""
        phase_key = phase.value
        if phase_key in self.phase_settings:
            return self.phase_settings[phase_key].max_tokens
        return 2000

    def get_phase_settings(self, phase: ExecutionPhase) -> PhaseSettings:
        """Get complete settings for specific phase."""
        phase_key = phase.value
        return self.phase_settings.get(phase_key, PhaseSettings())

    def get_prompt_template(self, template_type: str) -> PromptTemplate | None:
        """Get prompt template by type."""
        return self.prompt_templates.get(template_type)

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if model has specific capability."""
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "display_name": self.display_name,
            "phase_settings": {k: v.to_dict() for k, v in self.phase_settings.items()},
            "prompt_templates": {k: v.to_dict() for k, v in self.prompt_templates.items()},
            "retry_policy": self.retry_policy.to_dict(),
            "capabilities": [c.value for c in self.capabilities],
            "max_iterations": self.max_iterations,
            "max_parallel_calls": self.max_parallel_calls,
            "context_window": self.context_window,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
            "special_instructions": self.special_instructions,
            "optimization_tips": self.optimization_tips,
            "performance_metrics": self.performance_metrics,
            "config_version": self.config_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        phase_settings = {k: PhaseSettings(**v) for k, v in data.get("phase_settings", {}).items()}
        prompt_templates = {
            k: PromptTemplate(**v) for k, v in data.get("prompt_templates", {}).items()
        }
        retry_policy = RetryPolicy(**data.get("retry_policy", {}))
        capabilities = {ModelCapability(c) for c in data.get("capabilities", [])}

        return cls(
            model_name=data["model_name"],
            model_family=data["model_family"],
            display_name=data["display_name"],
            phase_settings=phase_settings,
            prompt_templates=prompt_templates,
            retry_policy=retry_policy,
            capabilities=capabilities,
            max_iterations=data.get("max_iterations", 10),
            max_parallel_calls=data.get("max_parallel_calls", 5),
            context_window=data.get("context_window", 4096),
            supports_streaming=data.get("supports_streaming", True),
            supports_function_calling=data.get("supports_function_calling", True),
            special_instructions=data.get("special_instructions", ""),
            optimization_tips=data.get("optimization_tips", []),
            performance_metrics=data.get("performance_metrics", {}),
            config_version=data.get("config_version", "1.0.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class ModelConfigRegistry:
    """Registry for model configurations."""

    def __init__(self):
        self._configs: dict[str, ModelConfig] = {}
        self._load_predefined_configs()

    def _load_predefined_configs(self):
        """Load predefined model configurations."""
        self._register_gpt4_configs()
        self._register_gpt35_configs()
        self._register_claude_configs()
        self._register_local_llm_configs()
        self._register_gemini_configs()

    def _register_gpt4_configs(self):
        """Register GPT-4 model configurations."""
        gpt4_config = ModelConfig(
            model_name="gpt-4",
            model_family="gpt",
            display_name="GPT-4",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.REFLECTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.JSON_MODE,
                ModelCapability.STREAMING,
            },
            max_iterations=15,
            max_parallel_calls=5,
            context_window=128000,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="GPT-4 excels at complex reasoning and multi-step planning. Use lower temperatures for planning, higher for creative tasks.",
            optimization_tips=[
                "Use CoT (Chain of Thought) for complex tasks",
                "Enable reflection for multi-step reasoning",
                "Leverage JSON mode for structured outputs",
                "Use parallel tool calls when safe",
            ],
        )

        gpt4_turbo = ModelConfig(
            model_name="gpt-4-turbo",
            model_family="gpt",
            display_name="GPT-4 Turbo",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.REFLECTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.JSON_MODE,
                ModelCapability.STREAMING,
            },
            max_iterations=20,
            max_parallel_calls=10,
            context_window=128000,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="GPT-4 Turbo is faster with same capabilities. Can handle more parallel operations.",
        )

        self._configs["gpt-4"] = gpt4_config
        self._configs["gpt-4-turbo"] = gpt4_turbo
        self._configs["gpt-4-turbo-preview"] = gpt4_turbo

    def _register_gpt35_configs(self):
        """Register GPT-3.5 model configurations."""
        gpt35_config = ModelConfig(
            model_name="gpt-3.5-turbo",
            model_family="gpt",
            display_name="GPT-3.5 Turbo",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.JSON_MODE,
                ModelCapability.STREAMING,
            },
            max_iterations=10,
            max_parallel_calls=3,
            context_window=16385,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="GPT-3.5 is faster but less capable at complex reasoning. Keep tasks focused and avoid deep reflection.",
            optimization_tips=[
                "Break complex tasks into smaller steps",
                "Use fewer parallel calls",
                "Avoid deep reflection chains",
                "Focus on tool accuracy over reasoning depth",
            ],
        )

        self._configs["gpt-3.5-turbo"] = gpt35_config

    def _register_claude_configs(self):
        """Register Claude model configurations."""
        claude_opus = ModelConfig(
            model_name="claude-3-opus",
            model_family="claude",
            display_name="Claude 3 Opus",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.REFLECTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.STREAMING,
            },
            max_iterations=15,
            max_parallel_calls=5,
            context_window=200000,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="Claude Opus excels at nuanced reasoning and safety. Lower temperatures work best for consistent outputs.",
            optimization_tips=[
                "Use detailed system prompts",
                "Leverage strong safety alignment",
                "Enable reflection for complex tasks",
                "Use lower temperatures for factual tasks",
            ],
        )

        claude_sonnet = ModelConfig(
            model_name="claude-3-sonnet",
            model_family="claude",
            display_name="Claude 3 Sonnet",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.STREAMING,
            },
            max_iterations=12,
            max_parallel_calls=4,
            context_window=200000,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="Claude Sonnet balances capability and speed. Good for most tasks.",
        )

        claude_haiku = ModelConfig(
            model_name="claude-3-haiku",
            model_family="claude",
            display_name="Claude 3 Haiku",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.STREAMING,
            },
            max_iterations=8,
            max_parallel_calls=2,
            context_window=200000,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="Claude Haiku is fast for simple tasks. Avoid complex multi-step planning.",
            optimization_tips=[
                "Keep prompts simple and direct",
                "Avoid deep reflection",
                "Use for straightforward tool calls",
                "Limit parallel execution",
            ],
        )

        self._configs["claude-3-opus"] = claude_opus
        self._configs["claude-3-sonnet"] = claude_sonnet
        self._configs["claude-3-haiku"] = claude_haiku

    def _register_local_llm_configs(self):
        """Register local LLM configurations."""
        glm4_config = ModelConfig(
            model_name="glm-4",
            model_family="glm",
            display_name="GLM-4",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.JSON_MODE,
            },
            max_iterations=8,
            max_parallel_calls=2,
            context_window=32768,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="GLM-4 is capable but may need careful prompting. Use explicit instructions.",
            optimization_tips=[
                "Use clear, explicit instructions",
                "Provide examples in prompts",
                "Avoid ambiguous requests",
                "Enable JSON mode for structured outputs",
            ],
        )

        llama2_config = ModelConfig(
            model_name="llama-2",
            model_family="llama",
            display_name="Llama 2",
            capabilities={
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
            },
            max_iterations=6,
            max_parallel_calls=1,
            context_window=4096,
            supports_streaming=True,
            supports_function_calling=False,
            special_instructions="Llama 2 requires careful prompting. Use ReAct pattern explicitly.",
            optimization_tips=[
                "Use ReAct pattern with explicit steps",
                "Keep prompts simple and structured",
                "Provide few-shot examples",
                "Avoid complex tool chaining",
            ],
        )

        mistral_config = ModelConfig(
            model_name="mistral",
            model_family="mistral",
            display_name="Mistral",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.JSON_MODE,
            },
            max_iterations=8,
            max_parallel_calls=2,
            context_window=8192,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="Mistral is efficient and capable. Good balance of speed and quality.",
            optimization_tips=[
                "Use moderate temperatures",
                "Provide clear tool schemas",
                "Enable reflection for better outputs",
            ],
        )

        self._configs["glm-4"] = glm4_config
        self._configs["glm-4.7"] = glm4_config
        self._configs["llama-2"] = llama2_config
        self._configs["mistral"] = mistral_config

    def _register_gemini_configs(self):
        """Register Gemini model configurations."""
        gemini_pro = ModelConfig(
            model_name="gemini-pro",
            model_family="gemini",
            display_name="Gemini Pro",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.STREAMING,
            },
            max_iterations=12,
            max_parallel_calls=4,
            context_window=32768,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="Gemini Pro has strong multimodal capabilities. Use structured prompts.",
            optimization_tips=[
                "Leverage multimodal inputs when available",
                "Use structured prompts with examples",
                "Enable parallel execution for independent tasks",
            ],
        )

        gemini_ultra = ModelConfig(
            model_name="gemini-ultra",
            model_family="gemini",
            display_name="Gemini Ultra",
            capabilities={
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.REFLECTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.STREAMING,
            },
            max_iterations=15,
            max_parallel_calls=5,
            context_window=32768,
            supports_streaming=True,
            supports_function_calling=True,
            special_instructions="Gemini Ultra is Google's most capable model. Optimized for complex tasks.",
        )

        self._configs["gemini-pro"] = gemini_pro
        self._configs["gemini-ultra"] = gemini_ultra

    def register_config(self, config: ModelConfig):
        """Register a new model configuration."""
        self._configs[config.model_name] = config
        logger.info(f"Registered configuration for model: {config.model_name}")

    def get_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a model."""
        model_lower = model_name.lower()

        for key, config in self._configs.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return config

        for key, config in self._configs.items():
            if config.model_family.lower() in model_lower:
                return config

        return None

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self._configs.keys())

    def get_models_by_family(self, family: str) -> list[ModelConfig]:
        """Get all models in a family."""
        return [
            config
            for config in self._configs.values()
            if config.model_family.lower() == family.lower()
        ]


def detect_model_capabilities(
    model_name: str, base_url: str | None = None
) -> set[ModelCapability]:
    """Detect model capabilities from model name and optional API check.

    Args:
        model_name: Name of the model
        base_url: Optional base URL for API capability check

    Returns:
        Set of detected capabilities
    """
    model_lower = model_name.lower()
    capabilities = set()

    if "gpt-4" in model_lower:
        capabilities.update(
            {
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.REFLECTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.JSON_MODE,
                ModelCapability.STREAMING,
            }
        )
    elif "gpt-3.5" in model_lower:
        capabilities.update(
            {
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.JSON_MODE,
                ModelCapability.STREAMING,
            }
        )
    elif "claude" in model_lower:
        capabilities.update(
            {
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.STREAMING,
            }
        )
        if "opus" in model_lower or "sonnet" in model_lower:
            capabilities.add(ModelCapability.REFLECTION)
            capabilities.add(ModelCapability.LONG_CONTEXT)
    elif "gemini" in model_lower:
        capabilities.update(
            {
                ModelCapability.TOOL_CALLING,
                ModelCapability.REASONING,
                ModelCapability.PARALLEL_EXECUTION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.STREAMING,
            }
        )
        if "ultra" in model_lower:
            capabilities.add(ModelCapability.REFLECTION)
            capabilities.add(ModelCapability.LONG_CONTEXT)
    elif any(x in model_lower for x in ["llama", "mistral", "phi", "gemma"]):
        capabilities.update(
            {
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
            }
        )
        if "mistral" in model_lower or "glm" in model_lower:
            capabilities.add(ModelCapability.TOOL_CALLING)
            capabilities.add(ModelCapability.JSON_MODE)

    return capabilities


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model, creating default if not found.

    Args:
        model_name: Name of the model

    Returns:
        ModelConfig for the model
    """
    registry = ModelConfigRegistry()
    config = registry.get_config(model_name)

    if config:
        return config

    capabilities = detect_model_capabilities(model_name)

    default_config = ModelConfig(
        model_name=model_name,
        model_family="unknown",
        display_name=model_name,
        capabilities=capabilities,
        supports_function_calling=ModelCapability.TOOL_CALLING in capabilities,
        special_instructions="Default configuration for unknown model. May need manual tuning.",
    )

    logger.warning(f"No predefined config for {model_name}, using default")
    return default_config


def get_temperature_for_phase(model_name: str, phase: ExecutionPhase) -> float:
    """Get phase-specific temperature for a model.

    Args:
        model_name: Name of the model
        phase: Execution phase

    Returns:
        Temperature value
    """
    config = get_model_config(model_name)
    return config.get_temperature(phase)


def get_max_tokens_for_phase(model_name: str, phase: ExecutionPhase) -> int:
    """Get phase-specific max tokens for a model.

    Args:
        model_name: Name of the model
        phase: Execution phase

    Returns:
        Max tokens value
    """
    config = get_model_config(model_name)
    return config.get_max_tokens(phase)


def get_prompt_template(
    model_name: str,
    template_type: str,
    template_registry: dict[str, PromptTemplate] | None = None,
) -> PromptTemplate | None:
    """Get prompt template for a model.

    Args:
        model_name: Name of the model
        template_type: Type of template (system, few_shot, error_handling, etc.)
        template_registry: Optional external template registry

    Returns:
        PromptTemplate or None
    """
    if template_registry:
        return template_registry.get(template_type)

    from prompts.react_prompt import REACT_SYSTEM_PROMPTS

    model_type = "default"
    if "gpt-4" in model_name.lower():
        model_type = "gpt4"
    elif "claude" in model_name.lower():
        model_type = "claude"
    elif any(x in model_name.lower() for x in ["llama", "mistral", "phi"]):
        model_type = "local_llm"

    return REACT_SYSTEM_PROMPTS.get(model_type)


class ModelConfigStorage:
    """Storage manager for model configurations with versioning."""

    def __init__(self, db_manager=None, storage_path: str = "config/model_configs"):
        """Initialize configuration storage.

        Args:
            db_manager: Optional database manager for persistence
            storage_path: Path for file-based storage
        """
        self.db_manager = db_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: ModelConfig, version: str | None = None) -> str:
        """Save configuration to file storage.

        Args:
            config: ModelConfig to save
            version: Optional version string

        Returns:
            Path to saved file
        """
        if version:
            config.config_version = version
        config.updated_at = datetime.now().isoformat()

        filename = f"{config.model_name.replace('/', '_')}-{config.config_version}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved config to {filepath}")
        return str(filepath)

    def load_config(self, model_name: str, version: str | None = None) -> ModelConfig | None:
        """Load configuration from file storage.

        Args:
            model_name: Name of the model
            version: Optional version string (loads latest if None)

        Returns:
            ModelConfig or None
        """
        pattern = f"{model_name.replace('/', '_')}"
        if version:
            pattern += f"-{version}"

        config_files = list(self.storage_path.glob(f"{pattern}*.json"))

        if not config_files:
            return None

        config_files.sort(reverse=True)
        filepath = config_files[0]

        with open(filepath) as f:
            data = json.load(f)

        return ModelConfig.from_dict(data)

    def save_to_yaml(self, config: ModelConfig, filepath: str):
        """Save configuration to YAML file.

        Args:
            config: ModelConfig to save
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)

        logger.info(f"Saved config to YAML: {filepath}")

    def load_from_yaml(self, filepath: str) -> ModelConfig | None:
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            ModelConfig or None
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = yaml.safe_load(f)

        return ModelConfig.from_dict(data)

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions of a model configuration.

        Args:
            model_name: Name of the model

        Returns:
            List of version strings
        """
        pattern = f"{model_name.replace('/', '_')}*.json"
        config_files = list(self.storage_path.glob(pattern))

        versions = []
        for f in config_files:
            parts = f.stem.split("-")
            if len(parts) > 1:
                versions.append(parts[-1])

        return sorted(set(versions))

    def delete_config(self, model_name: str, version: str | None = None):
        """Delete configuration file.

        Args:
            model_name: Name of the model
            version: Optional version (deletes all if None)
        """
        pattern = f"{model_name.replace('/', '_')}"
        if version:
            pattern += f"-{version}"

        config_files = list(self.storage_path.glob(f"{pattern}*.json"))

        for f in config_files:
            f.unlink()
            logger.info(f"Deleted config: {f}")


class ModelConfigOptimizer:
    """Optimizer for model configurations based on performance data."""

    def __init__(self, db_manager=None):
        """Initialize optimizer.

        Args:
            db_manager: Optional database manager for performance data
        """
        self.db_manager = db_manager
        self.performance_history: dict[str, list[dict]] = defaultdict(list)

    def track_performance(
        self,
        model_name: str,
        phase: ExecutionPhase,
        temperature: float,
        max_tokens: int,
        success: bool,
        duration_ms: float,
        iterations: int,
    ):
        """Track performance of a configuration.

        Args:
            model_name: Model name
            phase: Execution phase
            temperature: Temperature used
            max_tokens: Max tokens used
            success: Whether the operation was successful
            duration_ms: Duration in milliseconds
            iterations: Number of iterations
        """
        key = f"{model_name}:{phase.value}"
        self.performance_history[key].append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": success,
                "duration_ms": duration_ms,
                "iterations": iterations,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_optimal_settings(
        self,
        model_name: str,
        phase: ExecutionPhase,
        min_samples: int = 10,
    ) -> dict[str, Any] | None:
        """Get optimal settings based on performance history.

        Args:
            model_name: Model name
            phase: Execution phase
            min_samples: Minimum samples required for optimization

        Returns:
            Dictionary with optimal settings or None
        """
        key = f"{model_name}:{phase.value}"
        history = self.performance_history.get(key, [])

        if len(history) < min_samples:
            return None

        successful_runs = [r for r in history if r["success"]]

        if not successful_runs:
            return None

        avg_temp = sum(r["temperature"] for r in successful_runs) / len(successful_runs)
        avg_tokens = sum(r["max_tokens"] for r in successful_runs) / len(successful_runs)
        avg_duration = sum(r["duration_ms"] for r in successful_runs) / len(successful_runs)
        avg_iterations = sum(r["iterations"] for r in successful_runs) / len(successful_runs)

        return {
            "temperature": round(avg_temp, 2),
            "max_tokens": int(avg_tokens),
            "avg_duration_ms": round(avg_duration, 2),
            "avg_iterations": round(avg_iterations, 2),
            "sample_count": len(successful_runs),
        }

    def suggest_optimizations(self, config: ModelConfig) -> list[str]:
        """Suggest optimizations for a model configuration.

        Args:
            config: ModelConfig to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        for phase_name, phase_settings in config.phase_settings.items():
            phase = ExecutionPhase(phase_name)
            optimal = self.get_optimal_settings(config.model_name, phase)

            if optimal:
                if abs(phase_settings.temperature - optimal["temperature"]) > 0.2:
                    suggestions.append(
                        f"{phase_name}: Consider adjusting temperature from "
                        f"{phase_settings.temperature} to {optimal['temperature']}"
                    )

                if abs(phase_settings.max_tokens - optimal["max_tokens"]) > 500:
                    suggestions.append(
                        f"{phase_name}: Consider adjusting max_tokens from "
                        f"{phase_settings.max_tokens} to {optimal['max_tokens']}"
                    )

        if not suggestions:
            suggestions.append("Configuration appears optimal based on current data")

        return suggestions

    def auto_tune_config(self, config: ModelConfig) -> ModelConfig:
        """Automatically tune configuration based on performance data.

        Args:
            config: ModelConfig to tune

        Returns:
            Tuned ModelConfig
        """
        import copy

        tuned_config = copy.deepcopy(config)

        for phase_name, phase_settings in tuned_config.phase_settings.items():
            phase = ExecutionPhase(phase_name)
            optimal = self.get_optimal_settings(config.model_name, phase)

            if optimal:
                phase_settings.temperature = optimal["temperature"]
                phase_settings.max_tokens = optimal["max_tokens"]

        tuned_config.updated_at = datetime.now().isoformat()
        logger.info(f"Auto-tuned configuration for {config.model_name}")

        return tuned_config


def load_config_from_env() -> dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Dictionary of configuration values
    """
    config = {}

    for key, value in os.environ.items():
        if key.startswith("VIBEAGENT_MODEL_"):
            config_key = key[17:].lower()
            config[config_key] = value

    return config


def validate_config(config: ModelConfig) -> tuple[bool, list[str]]:
    """Validate a model configuration.

    Args:
        config: ModelConfig to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not config.model_name:
        errors.append("model_name is required")

    if not config.model_family:
        errors.append("model_family is required")

    for phase_name, phase_settings in config.phase_settings.items():
        if phase_settings.temperature < 0 or phase_settings.temperature > 2:
            errors.append(f"{phase_name}: temperature must be between 0 and 2")

        if phase_settings.max_tokens <= 0:
            errors.append(f"{phase_name}: max_tokens must be positive")

    if config.context_window <= 0:
        errors.append("context_window must be positive")

    if config.max_iterations <= 0:
        errors.append("max_iterations must be positive")

    if config.max_parallel_calls <= 0:
        errors.append("max_parallel_calls must be positive")

    return (len(errors) == 0, errors)


def create_ab_test_config(
    base_config: ModelConfig,
    variant_name: str,
    changes: dict[str, Any],
) -> ModelConfig:
    """Create an A/B test variant configuration.

    Args:
        base_config: Base configuration to modify
        variant_name: Name for the variant
        changes: Changes to apply to the configuration

    Returns:
        New ModelConfig variant
    """
    import copy

    variant = copy.deepcopy(base_config)
    variant.model_name = f"{base_config.model_name}-{variant_name}"
    variant.config_version = f"ab-{variant_name}"

    for phase_name, phase_changes in changes.get("phase_settings", {}).items():
        if phase_name in variant.phase_settings:
            for key, value in phase_changes.items():
                setattr(variant.phase_settings[phase_name], key, value)

    if "max_iterations" in changes:
        variant.max_iterations = changes["max_iterations"]

    if "max_parallel_calls" in changes:
        variant.max_parallel_calls = changes["max_parallel_calls"]

    return variant


def integrate_with_orchestrator(orchestrator, model_name: str) -> ModelConfig:
    """Integrate model configuration with an orchestrator.

    Args:
        orchestrator: Orchestrator instance (ToolOrchestrator, PlanExecuteOrchestrator, etc.)
        model_name: Name of the model

    Returns:
        ModelConfig for the orchestrator
    """
    config = get_model_config(model_name)

    if hasattr(orchestrator, "llm_skill"):
        orchestrator.llm_skill.model_config = config

    if hasattr(orchestrator, "react_config"):
        orchestrator.react_config["max_reasoning_steps"] = config.max_iterations

    if hasattr(orchestrator, "parallel_executor"):
        orchestrator.parallel_executor.config.max_parallel_calls = config.max_parallel_calls

    logger.info(f"Integrated model config for {model_name} with orchestrator")

    return config


def get_llm_params_for_phase(
    model_name: str,
    phase: ExecutionPhase,
    additional_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get complete LLM parameters for a specific phase.

    Args:
        model_name: Name of the model
        phase: Execution phase
        additional_params: Additional parameters to include

    Returns:
        Dictionary of LLM parameters
    """
    config = get_model_config(model_name)
    phase_settings = config.get_phase_settings(phase)

    params = {
        "model": model_name,
        "temperature": phase_settings.temperature,
        "max_tokens": phase_settings.max_tokens,
        "top_p": phase_settings.top_p,
    }

    if phase_settings.top_k:
        params["top_k"] = phase_settings.top_k

    if phase_settings.presence_penalty != 0:
        params["presence_penalty"] = phase_settings.presence_penalty

    if phase_settings.frequency_penalty != 0:
        params["frequency_penalty"] = phase_settings.frequency_penalty

    if phase_settings.stop_sequences:
        params["stop"] = phase_settings.stop_sequences

    if additional_params:
        params.update(additional_params)

    return params


registry = ModelConfigRegistry()

__all__ = [
    "ExecutionPhase",
    "ModelCapability",
    "PhaseSettings",
    "RetryPolicy",
    "PromptTemplate",
    "ModelConfig",
    "ModelConfigRegistry",
    "ModelConfigStorage",
    "ModelConfigOptimizer",
    "get_model_config",
    "get_temperature_for_phase",
    "get_max_tokens_for_phase",
    "get_prompt_template",
    "detect_model_capabilities",
    "validate_config",
    "create_ab_test_config",
    "integrate_with_orchestrator",
    "get_llm_params_for_phase",
    "load_config_from_env",
    "registry",
]
