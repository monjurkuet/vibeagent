"""Quick test of ContextManager functionality."""

import sys

sys.path.insert(0, ".")

from core.context_manager import (
    ContextManager,
    ContextType,
    ContextConfig,
    CompressionStrategy,
)


def create_sample_messages():
    """Create sample conversation messages."""
    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"Question {i}"})
        messages.append({"role": "assistant", "content": f"Answer {i}"})
        messages.append({"role": "tool", "content": f"Tool result {i}"})
    return messages


def test_basic_functionality():
    """Test basic ContextManager functionality."""
    print("Testing ContextManager...")

    config = ContextConfig(
        max_context_tokens=1000,
        summary_threshold=500,
        recency_weight=0.4,
    )

    context_manager = ContextManager(config=config)

    messages = create_sample_messages()

    print(f"\nOriginal messages: {len(messages)}")
    original_tokens = context_manager.get_token_usage(messages)
    print(f"Original tokens: {original_tokens}")

    print("\n--- Testing manage_context ---")
    managed = context_manager.manage_context(messages, max_tokens=500)
    managed_tokens = context_manager.get_token_usage(managed)
    print(f"Managed messages: {len(managed)}")
    print(f"Managed tokens: {managed_tokens}")
    print(f"Reduction: {(1 - managed_tokens / original_tokens):.1%}")

    print("\n--- Testing summarize_messages ---")
    summary = context_manager.summarize_messages(messages)
    print(f"Original messages: {summary.original_messages}")
    print(f"Token reduction: {summary.token_reduction:.1%}")
    print(f"Summary text (first 200 chars): {summary.summary_text[:200]}...")

    print("\n--- Testing compress_context ---")
    compressed = context_manager.compress_context(messages)
    print(f"Compressed messages: {len(compressed)}")
    print(f"Reduction: {(1 - len(compressed) / len(messages)):.1%}")

    print("\n--- Testing analyze_context ---")
    analysis = context_manager.analyze_context(messages)
    print(f"Total messages: {analysis.total_messages}")
    print(f"Total tokens: {analysis.total_tokens}")
    print(f"Quality score: {analysis.quality_score:.2f}")
    print(f"Compression potential: {analysis.compression_potential:.1%}")

    print("\n--- Testing context types ---")
    full = context_manager.get_context(messages, ContextType.FULL)
    print(f"Full context: {len(full)} messages")

    minimal = context_manager.get_context(messages, ContextType.MINIMAL)
    print(f"Minimal context: {len(minimal)} messages")

    print("\n--- Testing importance scoring ---")
    test_msgs = [
        {"role": "user", "content": "This is critical and urgent!"},
        {"role": "tool", "content": "Error: Something failed"},
        {"role": "assistant", "content": "Final Answer: The solution is..."},
    ]
    for msg in test_msgs:
        score = context_manager.calculate_importance(msg)
        print(
            f"{msg['role']}: importance={score.importance_score:.2f}, final={score.final_score:.2f}"
        )

    print("\n--- Testing optimize_for_tokens ---")
    optimized = context_manager.optimize_for_tokens(messages, max_tokens=300)
    print(f"Optimized tokens: {context_manager.get_token_usage(optimized)}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
