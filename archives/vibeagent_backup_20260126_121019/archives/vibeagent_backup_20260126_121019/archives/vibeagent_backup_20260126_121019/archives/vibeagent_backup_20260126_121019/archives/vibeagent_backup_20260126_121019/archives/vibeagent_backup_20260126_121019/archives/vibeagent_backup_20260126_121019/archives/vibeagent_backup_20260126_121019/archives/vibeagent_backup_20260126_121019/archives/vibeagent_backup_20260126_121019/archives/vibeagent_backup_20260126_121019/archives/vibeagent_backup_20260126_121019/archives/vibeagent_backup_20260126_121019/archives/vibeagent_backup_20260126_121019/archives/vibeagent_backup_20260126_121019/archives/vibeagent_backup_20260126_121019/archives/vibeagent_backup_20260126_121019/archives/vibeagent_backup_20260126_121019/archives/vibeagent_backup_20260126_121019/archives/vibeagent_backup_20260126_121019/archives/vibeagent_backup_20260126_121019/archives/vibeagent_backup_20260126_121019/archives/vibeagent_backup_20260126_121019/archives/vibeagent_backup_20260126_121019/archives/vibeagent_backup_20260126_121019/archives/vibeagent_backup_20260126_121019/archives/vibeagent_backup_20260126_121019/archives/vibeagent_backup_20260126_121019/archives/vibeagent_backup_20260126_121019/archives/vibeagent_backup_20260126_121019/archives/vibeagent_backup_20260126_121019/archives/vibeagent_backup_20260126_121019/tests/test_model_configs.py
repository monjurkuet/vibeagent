"""Tests for model configuration system."""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_configs import (
    ExecutionPhase,
    ModelCapability,
    PhaseSettings,
    RetryPolicy,
    PromptTemplate,
    ModelConfig,
    ModelConfigRegistry,
    ModelConfigStorage,
    ModelConfigOptimizer,
    get_model_config,
    get_temperature_for_phase,
    get_max_tokens_for_phase,
    detect_model_capabilities,
    validate_config,
    create_ab_test_config,
    get_llm_params_for_phase,
)


class TestPhaseSettings(unittest.TestCase):
    """Test PhaseSettings class."""

    def test_default_settings(self):
        """Test default phase settings."""
        settings = PhaseSettings()
        self.assertEqual(settings.temperature, 0.7)
        self.assertEqual(settings.max_tokens, 2000)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = PhaseSettings(temperature=0.5, max_tokens=1500)
        data = settings.to_dict()
        self.assertEqual(data["temperature"], 0.5)
        self.assertEqual(data["max_tokens"], 1500)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""

    def test_default_config(self):
        """Test default model configuration."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )

        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.max_parallel_calls, 5)

    def test_get_temperature(self):
        """Test getting phase-specific temperature."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )

        temp = config.get_temperature(ExecutionPhase.PLANNING)
        self.assertEqual(temp, 0.3)

    def test_get_max_tokens(self):
        """Test getting phase-specific max tokens."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )

        tokens = config.get_max_tokens(ExecutionPhase.PLANNING)
        self.assertEqual(tokens, 3000)

    def test_has_capability(self):
        """Test capability checking."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
            capabilities={ModelCapability.TOOL_CALLING, ModelCapability.REASONING},
        )

        self.assertTrue(config.has_capability(ModelCapability.TOOL_CALLING))
        self.assertFalse(config.has_capability(ModelCapability.STREAMING))


class TestModelConfigRegistry(unittest.TestCase):
    """Test ModelConfigRegistry class."""

    def test_get_config_gpt4(self):
        """Test getting GPT-4 configuration."""
        registry = ModelConfigRegistry()
        config = registry.get_config("gpt-4")

        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "gpt-4")
        self.assertTrue(config.has_capability(ModelCapability.TOOL_CALLING))
        self.assertTrue(config.has_capability(ModelCapability.REASONING))

    def test_get_config_claude(self):
        """Test getting Claude configuration."""
        registry = ModelConfigRegistry()
        config = registry.get_config("claude-3-opus")

        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "claude-3-opus")

    def test_list_models(self):
        """Test listing all models."""
        registry = ModelConfigRegistry()
        models = registry.list_models()

        self.assertGreater(len(models), 0)
        self.assertIn("gpt-4", models)
        self.assertIn("gpt-3.5-turbo", models)


class TestCapabilityDetection(unittest.TestCase):
    """Test model capability detection."""

    def test_detect_gpt4_capabilities(self):
        """Test GPT-4 capability detection."""
        capabilities = detect_model_capabilities("gpt-4")

        self.assertIn(ModelCapability.TOOL_CALLING, capabilities)
        self.assertIn(ModelCapability.REASONING, capabilities)
        self.assertIn(ModelCapability.CODE_GENERATION, capabilities)

    def test_detect_claude_capabilities(self):
        """Test Claude capability detection."""
        capabilities = detect_model_capabilities("claude-3-opus")

        self.assertIn(ModelCapability.TOOL_CALLING, capabilities)
        self.assertIn(ModelCapability.REASONING, capabilities)

    def test_detect_local_llm_capabilities(self):
        """Test local LLM capability detection."""
        capabilities = detect_model_capabilities("llama-2")

        self.assertIn(ModelCapability.REASONING, capabilities)
        self.assertNotIn(ModelCapability.TOOL_CALLING, capabilities)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_get_model_config(self):
        """Test getting model configuration."""
        config = get_model_config("gpt-4")

        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "gpt-4")

    def test_get_temperature_for_phase(self):
        """Test getting phase-specific temperature."""
        temp = get_temperature_for_phase("gpt-4", ExecutionPhase.PLANNING)
        self.assertEqual(temp, 0.3)

    def test_get_max_tokens_for_phase(self):
        """Test getting phase-specific max tokens."""
        tokens = get_max_tokens_for_phase("gpt-4", ExecutionPhase.PLANNING)
        self.assertEqual(tokens, 3000)

    def test_get_llm_params_for_phase(self):
        """Test getting complete LLM parameters."""
        params = get_llm_params_for_phase("gpt-4", ExecutionPhase.PLANNING)

        self.assertEqual(params["model"], "gpt-4")
        self.assertEqual(params["temperature"], 0.3)
        self.assertEqual(params["max_tokens"], 3000)
        self.assertIn("top_p", params)


class TestModelConfigStorage(unittest.TestCase):
    """Test ModelConfigStorage class."""

    def setUp(self):
        """Set up test storage directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ModelConfigStorage(storage_path=self.temp_dir)

    def tearDown(self):
        """Clean up test storage directory."""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )

        filepath = self.storage.save_config(config, version="v1.0")
        self.assertTrue(Path(filepath).exists())

        loaded = self.storage.load_config("test-model", version="v1.0")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.model_name, "test-model")

    def test_list_versions(self):
        """Test listing configuration versions."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )

        self.storage.save_config(config, version="v1.0")
        self.storage.save_config(config, version="v2.0")

        versions = self.storage.list_versions("test-model")
        self.assertEqual(len(versions), 2)
        self.assertIn("v1.0", versions)
        self.assertIn("v2.0", versions)


class TestModelConfigOptimizer(unittest.TestCase):
    """Test ModelConfigOptimizer class."""

    def test_track_performance(self):
        """Test performance tracking."""
        optimizer = ModelConfigOptimizer()

        optimizer.track_performance(
            model_name="gpt-4",
            phase=ExecutionPhase.EXECUTION,
            temperature=0.7,
            max_tokens=2000,
            success=True,
            duration_ms=1000,
            iterations=3,
        )

        self.assertEqual(len(optimizer.performance_history), 1)

    def test_get_optimal_settings(self):
        """Test getting optimal settings."""
        optimizer = ModelConfigOptimizer()

        for _ in range(15):
            optimizer.track_performance(
                model_name="gpt-4",
                phase=ExecutionPhase.EXECUTION,
                temperature=0.7,
                max_tokens=2000,
                success=True,
                duration_ms=1000,
                iterations=3,
            )

        optimal = optimizer.get_optimal_settings("gpt-4", ExecutionPhase.EXECUTION)

        self.assertIsNotNone(optimal)
        self.assertEqual(optimal["temperature"], 0.7)
        self.assertEqual(optimal["max_tokens"], 2000)


class TestABTesting(unittest.TestCase):
    """Test A/B testing functionality."""

    def test_create_ab_test_config(self):
        """Test creating A/B test variant."""
        base_config = get_model_config("gpt-4")

        variant = create_ab_test_config(
            base_config,
            variant_name="test-variant",
            changes={
                "phase_settings": {ExecutionPhase.PLANNING.value: {"temperature": 0.1}}
            },
        )

        self.assertEqual(variant.model_name, "gpt-4-test-variant")
        self.assertEqual(variant.get_temperature(ExecutionPhase.PLANNING), 0.1)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_valid_config(self):
        """Test validating a valid configuration."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )

        is_valid, errors = validate_config(config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_temperature(self):
        """Test validating configuration with invalid temperature."""
        config = ModelConfig(
            model_name="test-model",
            model_family="test",
            display_name="Test Model",
        )
        config.phase_settings[ExecutionPhase.PLANNING.value].temperature = 3.0

        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
