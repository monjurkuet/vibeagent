"""Integration test showing ContextManager with ToolOrchestrator pattern."""

import sys

sys.path.insert(0, ".")

from core.context_manager import (
    ContextManager,
    ContextType,
    ContextConfig,
)


class MockLLMSkill:
    """Mock LLM skill for testing."""

    def __init__(self):
        self.model = "gpt-4"
        self.base_url = "http://localhost:11434"


class MockToolOrchestrator:
    """Mock ToolOrchestrator demonstrating ContextManager integration."""

    def __init__(self):
        self.llm_skill = MockLLMSkill()
        self.skills = {}
        self.db_manager = None

        # Initialize ContextManager
        config = ContextConfig(
            max_context_tokens=8000,
            summary_threshold=4000,
            recency_weight=0.4,
        )
        self.context_manager = ContextManager(
            config=config,
            db_manager=self.db_manager,
            llm_skill=self.llm_skill,
        )

        # Conversation history
        self.conversation_history = []

    def process_message(self, user_message: str):
        """Process user message with context management."""
        print(f"\n{'=' * 60}")
        print(f"Processing: {user_message}")
        print(f"{'=' * 60}")

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Analyze current context
        print(f"\n📊 Context Analysis:")
        analysis = self.context_manager.analyze_context(self.conversation_history)
        print(f"  Total messages: {analysis.total_messages}")
        print(f"  Total tokens: {analysis.total_tokens}")
        print(f"  Quality score: {analysis.quality_score:.2f}")

        # Optimize context for LLM call
        print(f"\n🔄 Optimizing context...")
        optimized_context = self.context_manager.optimize_for_tokens(
            self.conversation_history, max_tokens=4000
        )
        optimized_tokens = self.context_manager.get_token_usage(optimized_context)
        original_tokens = analysis.total_tokens
        reduction = (1 - optimized_tokens / original_tokens) * 100
        print(f"  Original tokens: {original_tokens}")
        print(f"  Optimized tokens: {optimized_tokens}")
        print(f"  Reduction: {reduction:.1f}%")

        # Simulate LLM response
        assistant_response = f"I understand: {user_message}"
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        # Get minimal context for next iteration
        minimal_context = self.context_manager.get_context(
            self.conversation_history, ContextType.MINIMAL
        )
        print(f"\n📋 Minimal context for next turn: {len(minimal_context)} messages")

        return assistant_response

    def get_context_summary(self):
        """Get summary of conversation context."""
        if not self.conversation_history:
            return "No conversation history"

        summary = self.context_manager.summarize_messages(self.conversation_history)
        return summary.summary_text

    def get_usage_stats(self):
        """Get context usage statistics."""
        return self.context_manager.get_usage_statistics()


def test_integration():
    """Test ContextManager integration with ToolOrchestrator pattern."""
    print("🚀 Testing ContextManager Integration with ToolOrchestrator")
    print("=" * 60)

    # Create orchestrator with context manager
    orchestrator = MockToolOrchestrator()

    # Simulate a conversation
    messages = [
        "Hello, I need help with Python programming",
        "Can you explain data structures?",
        "How do I use lists and dictionaries?",
        "What about sets and tuples?",
        "Can you give me examples?",
        "That's helpful, thanks!",
        "Now I need to understand functions",
        "How do I define and call functions?",
        "What about parameters and return values?",
        "Can you show me lambda functions?",
        "What are decorators?",
        "How do I handle errors?",
        "What are try/except blocks?",
        "How do I raise exceptions?",
        "That's all for now, thanks!",
    ]

    # Process each message
    for msg in messages:
        response = orchestrator.process_message(msg)
        print(f"\n💬 Response: {response[:60]}...")

    # Get final statistics
    print(f"\n{'=' * 60}")
    print("📈 Final Statistics")
    print(f"{'=' * 60}")

    stats = orchestrator.get_usage_stats()
    print(f"Cache size: {stats['cache_size']}")
    print(f"Summary cache size: {stats['summary_cache_size']}")

    # Get conversation summary
    print(f"\n📝 Conversation Summary:")
    summary = orchestrator.get_context_summary()
    print(summary[:300] + "..." if len(summary) > 300 else summary)

    # Analyze final context
    print(f"\n🔍 Final Context Analysis:")
    analysis = orchestrator.context_manager.analyze_context(
        orchestrator.conversation_history
    )
    print(f"Total messages: {analysis.total_messages}")
    print(f"Total tokens: {analysis.total_tokens}")
    print(f"Quality score: {analysis.quality_score:.2f}")
    print(f"Compression potential: {analysis.compression_potential:.1%}")

    # Show suggestions
    if analysis.suggestions:
        print(f"\n💡 Suggestions:")
        for suggestion in analysis.suggestions:
            print(f"  - {suggestion}")

    print(f"\n{'=' * 60}")
    print("✅ Integration test complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_integration()
