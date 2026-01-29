"""Tests for ContextManager."""

import pytest
from core.context_manager import (
    CompressionStrategy,
    ContextConfig,
    ContextManager,
    ContextType,
    MessageScore,
)


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I help you today?",
        },
        {
            "role": "user",
            "content": "I need to search for information about Python programming.",
        },
        {
            "role": "assistant",
            "content": "I can help you search for Python programming information. Let me use the search tool.",
        },
        {
            "role": "tool",
            "content": "Success: Found 10 results about Python programming.",
        },
        {
            "role": "assistant",
            "content": "I found 10 results about Python programming. Here are some key points...",
        },
        {
            "role": "user",
            "content": "That's great! Can you find more about machine learning?",
        },
        {
            "role": "assistant",
            "content": "Certainly! Let me search for machine learning information.",
        },
        {
            "role": "tool",
            "content": "Success: Found 15 results about machine learning.",
        },
        {
            "role": "assistant",
            "content": "Final Answer: Here's what I found about machine learning...",
        },
    ]


@pytest.fixture
def context_manager():
    """Create a ContextManager instance for testing."""
    config = ContextConfig(
        max_context_tokens=1000,
        summary_threshold=500,
        recency_weight=0.4,
    )
    return ContextManager(config=config)


class TestContextManager:
    """Test cases for ContextManager."""

    def test_initialization(self, context_manager):
        """Test ContextManager initialization."""
        assert context_manager is not None
        assert context_manager.config.max_context_tokens == 1000
        assert context_manager.config.summary_threshold == 500
        assert context_manager.config.recency_weight == 0.4

    def test_estimate_tokens(self, context_manager):
        """Test token estimation."""
        text = "Hello world!"
        tokens = context_manager.estimate_tokens(text)
        assert tokens > 0

        long_text = " ".join(["word"] * 100)
        tokens = context_manager.estimate_tokens(long_text)
        assert tokens >= 100

    def test_get_token_usage(self, context_manager, sample_messages):
        """Test token usage calculation."""
        usage = context_manager.get_token_usage(sample_messages)
        assert usage > 0

    def test_calculate_importance(self, context_manager):
        """Test importance calculation."""
        user_msg = {"role": "user", "content": "This is important!"}
        score = context_manager.calculate_importance(user_msg)

        assert isinstance(score, MessageScore)
        assert score.importance_score > 0
        assert score.final_score > 0
        assert "role_weight" in score.factors

        error_msg = {
            "role": "tool",
            "content": "Error: Something went wrong that is critical and needs attention",
        }
        error_score = context_manager.calculate_importance(error_msg)
        assert error_score.importance_score > 0

    def test_score_messages(self, context_manager, sample_messages):
        """Test scoring all messages."""
        scores = context_manager._score_messages(sample_messages)

        assert len(scores) == len(sample_messages)
        assert all(isinstance(s, MessageScore) for s in scores)
        assert all(s.final_score > 0 for s in scores)

        last_score = scores[-1]
        first_score = scores[0]
        assert last_score.recency_score >= first_score.recency_score

    def test_manage_context_within_limits(self, context_manager, sample_messages):
        """Test context management when within token limits."""
        result = context_manager.manage_context(sample_messages, max_tokens=10000)

        assert len(result) == len(sample_messages)

    def test_manage_context_exceeds_limits(self, context_manager, sample_messages):
        """Test context management when exceeding token limits."""
        result = context_manager.manage_context(sample_messages, max_tokens=100)

        assert len(result) <= len(sample_messages)
        assert context_manager.get_token_usage(result) <= 100

    def test_summarize_messages(self, context_manager, sample_messages):
        """Test message summarization."""
        summary = context_manager.summarize_messages(sample_messages)

        from core.context_manager import ContextSummary

        assert isinstance(summary, ContextSummary)
        assert summary.original_messages == len(sample_messages)
        assert summary.token_reduction > 0
        assert len(summary.summary_text) > 0

    def test_extract_key_points(self, context_manager, sample_messages):
        """Test key point extraction."""
        key_points = context_manager._extract_key_points(sample_messages)

        assert isinstance(key_points, list)
        assert len(key_points) > 0

    def test_compress_context(self, context_manager, sample_messages):
        """Test context compression."""
        compressed = context_manager.compress_context(sample_messages)

        assert isinstance(compressed, list)
        assert len(compressed) <= len(sample_messages)

    def test_retrieve_relevant_context(self, context_manager, sample_messages):
        """Test relevant context retrieval."""
        query = "machine learning"
        relevant = context_manager.retrieve_relevant_context(query, sample_messages)

        assert isinstance(relevant, list)
        assert len(relevant) > 0
        assert all(isinstance(msg, dict) for msg in relevant)

    def test_get_context_full(self, context_manager, sample_messages):
        """Test getting full context."""
        context = context_manager.get_context(sample_messages, ContextType.FULL)

        assert isinstance(context, list)
        assert len(context) > 0

    def test_get_context_summary(self, context_manager, sample_messages):
        """Test getting summary context."""
        context = context_manager.get_context(sample_messages, ContextType.SUMMARY)

        assert isinstance(context, list)
        assert len(context) >= 0

    def test_get_context_relevant(self, context_manager, sample_messages):
        """Test getting relevant context."""
        context = context_manager.get_context(sample_messages, ContextType.RELEVANT)

        assert isinstance(context, list)
        assert len(context) >= 0

    def test_get_context_minimal(self, context_manager, sample_messages):
        """Test getting minimal context."""
        context = context_manager.get_context(sample_messages, ContextType.MINIMAL)

        assert isinstance(context, list)
        assert len(context) <= len(sample_messages)

    def test_optimize_for_tokens(self, context_manager, sample_messages):
        """Test token optimization."""
        optimized = context_manager.optimize_for_tokens(sample_messages, max_tokens=200)

        assert context_manager.get_token_usage(optimized) <= 200

    def test_analyze_context(self, context_manager, sample_messages):
        """Test context analysis."""
        analysis = context_manager.analyze_context(sample_messages)

        assert analysis.total_messages == len(sample_messages)
        assert analysis.total_tokens > 0
        assert analysis.quality_score > 0
        assert isinstance(analysis.gaps, list)
        assert isinstance(analysis.suggestions, list)

    def test_is_tool_result_important(self, context_manager):
        """Test tool result importance check."""
        error_result = {"error": "Something went wrong"}
        assert context_manager.is_tool_result_important(error_result) is True

        success_result = {"success": "Done"}
        assert context_manager.is_tool_result_important(success_result) is False

        long_result = {"data": "x" * 3000}
        assert context_manager.is_tool_result_important(long_result) is False

    def test_get_essential_messages(self, context_manager, sample_messages):
        """Test getting essential messages."""
        essential = context_manager.get_essential_messages(sample_messages)

        assert isinstance(essential, list)
        assert len(essential) <= len(sample_messages)
        assert all(msg.get("role") in ["user", "assistant", "tool"] for msg in essential)

    def test_merge_similar_messages(self, context_manager):
        """Test merging similar messages."""
        messages = [
            {"role": "tool", "content": "Result 1"},
            {"role": "tool", "content": "Result 2"},
            {"role": "user", "content": "Query"},
        ]

        merged = context_manager.merge_similar_messages(messages)

        assert isinstance(merged, list)
        assert len(merged) <= len(messages)

    def test_detect_redundancy(self, context_manager):
        """Test redundancy detection."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        redundancy = context_manager.detect_redundancy(messages)

        assert isinstance(redundancy, list)
        assert len(redundancy) > 0

    def test_get_usage_statistics(self, context_manager):
        """Test usage statistics."""
        stats = context_manager.get_usage_statistics()

        assert isinstance(stats, dict)
        assert "cache_size" in stats
        assert "config" in stats

    def test_clear_cache(self, context_manager, sample_messages):
        """Test cache clearing."""
        context_manager.manage_context(sample_messages)
        context_manager.clear_cache()

        stats = context_manager.get_usage_statistics()
        assert stats["cache_size"] == 0
        assert stats["summary_cache_size"] == 0


class TestContextConfig:
    """Test cases for ContextConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ContextConfig()

        assert config.max_context_tokens == 8000
        assert config.summary_threshold == 4000
        assert config.recency_weight == 0.4
        assert config.compression_strategy == CompressionStrategy.HYBRID

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextConfig(
            max_context_tokens=5000,
            summary_threshold=2000,
            recency_weight=0.6,
            compression_strategy=CompressionStrategy.IMPORTANCE_BASED,
        )

        assert config.max_context_tokens == 5000
        assert config.summary_threshold == 2000
        assert config.recency_weight == 0.6
        assert config.compression_strategy == CompressionStrategy.IMPORTANCE_BASED


class TestContextType:
    """Test cases for ContextType enum."""

    def test_context_types(self):
        """Test all context types."""
        assert ContextType.FULL.value == "full"
        assert ContextType.SUMMARY.value == "summary"
        assert ContextType.RELEVANT.value == "relevant"
        assert ContextType.MINIMAL.value == "minimal"


class TestCompressionStrategy:
    """Test cases for CompressionStrategy enum."""

    def test_compression_strategies(self):
        """Test all compression strategies."""
        assert CompressionStrategy.IMPORTANCE_BASED.value == "importance_based"
        assert CompressionStrategy.TEMPORAL.value == "temporal"
        assert CompressionStrategy.SEMANTIC.value == "semantic"
        assert CompressionStrategy.HYBRID.value == "hybrid"


class TestIntegration:
    """Integration tests for ContextManager."""

    def test_full_workflow(self, context_manager, sample_messages):
        """Test complete context management workflow."""
        original_tokens = context_manager.get_token_usage(sample_messages)

        analysis = context_manager.analyze_context(sample_messages)
        assert analysis.total_messages == len(sample_messages)

        compressed = context_manager.compress_context(sample_messages)
        assert len(compressed) <= len(sample_messages)

        managed = context_manager.manage_context(compressed, max_tokens=500)
        assert context_manager.get_token_usage(managed) <= 500

        summary = context_manager.summarize_messages(managed)
        assert summary.token_reduction > 0

        minimal = context_manager.get_context(managed, ContextType.MINIMAL)
        assert len(minimal) <= len(managed)

    def test_long_conversation(self, context_manager):
        """Test handling of long conversations."""
        long_messages = []
        for i in range(50):
            long_messages.append({"role": "user", "content": f"Message {i}"})
            long_messages.append({"role": "assistant", "content": f"Response to message {i}"})

        result = context_manager.manage_context(long_messages, max_tokens=1000)
        assert context_manager.get_token_usage(result) <= 1000

        summary = context_manager.summarize_messages(long_messages)
        assert summary.token_reduction > 0.5

    def test_error_handling(self, context_manager):
        """Test error handling."""
        empty_messages = []
        result = context_manager.manage_context(empty_messages)
        assert result == []

        result = context_manager.summarize_messages(empty_messages)
        assert result.original_messages == 0

        result = context_manager.retrieve_relevant_context("query", [])
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
