"""Example demonstrating ContextManager integration with ToolOrchestrator."""

import logging
from core.context_manager import (
    ContextManager,
    ContextType,
    ContextConfig,
    CompressionStrategy,
)
from core.tool_orchestrator import ToolOrchestrator
from core.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_conversation():
    """Create a sample long conversation."""
    messages = []

    messages.append({"role": "user", "content": "I need help with a Python project"})

    messages.append(
        {
            "role": "assistant",
            "content": "I'd be happy to help you with your Python project! What specifically do you need assistance with?",
        }
    )

    messages.append(
        {
            "role": "user",
            "content": "I need to search for information about data processing libraries",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Let me search for information about Python data processing libraries for you.",
        }
    )

    messages.append(
        {
            "role": "tool",
            "content": "Success: Found information about pandas, numpy, and other libraries.",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "I found several popular Python data processing libraries:\n- pandas: Data manipulation and analysis\n- numpy: Numerical computing\n- matplotlib: Data visualization",
        }
    )

    messages.append(
        {
            "role": "user",
            "content": "That's helpful! Can you find more about machine learning?",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Let me search for machine learning libraries and frameworks.",
        }
    )

    messages.append(
        {
            "role": "tool",
            "content": "Success: Found information about scikit-learn, TensorFlow, PyTorch.",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Here are popular machine learning libraries:\n- scikit-learn: Traditional ML algorithms\n- TensorFlow: Deep learning framework\n- PyTorch: Deep learning framework",
        }
    )

    messages.append(
        {
            "role": "user",
            "content": "I need to understand how to use pandas for data analysis",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "I can help you understand pandas for data analysis. Let me search for tutorials and examples.",
        }
    )

    messages.append(
        {
            "role": "tool",
            "content": "Success: Found pandas tutorials and documentation.",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Final Answer: Here's how to use pandas for data analysis:\n1. Import pandas: import pandas as pd\n2. Load data: df = pd.read_csv('data.csv')\n3. Explore data: df.head(), df.describe()\n4. Clean data: df.dropna(), df.fillna()\n5. Analyze data: df.groupby(), df.pivot_table()",
        }
    )

    for i in range(20):
        messages.append({"role": "user", "content": f"Additional question {i}"})
        messages.append(
            {"role": "assistant", "content": f"Response to additional question {i}"}
        )

    return messages


def example_basic_usage():
    """Demonstrate basic ContextManager usage."""
    logger.info("=== Basic ContextManager Usage ===")

    config = ContextConfig(
        max_context_tokens=1000,
        summary_threshold=500,
        recency_weight=0.4,
    )

    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    original_tokens = context_manager.get_token_usage(messages)
    logger.info(f"Original messages: {len(messages)}")
    logger.info(f"Original tokens: {original_tokens}")

    managed = context_manager.manage_context(messages, max_tokens=1000)
    managed_tokens = context_manager.get_token_usage(managed)
    logger.info(f"Managed messages: {len(managed)}")
    logger.info(f"Managed tokens: {managed_tokens}")
    logger.info(f"Reduction: {(1 - managed_tokens / original_tokens):.1%}")


def example_context_types():
    """Demonstrate different context types."""
    logger.info("\n=== Different Context Types ===")

    config = ContextConfig(max_context_tokens=1000)
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    full_context = context_manager.get_context(messages, ContextType.FULL)
    logger.info(f"Full context: {len(full_context)} messages")

    summary_context = context_manager.get_context(messages, ContextType.SUMMARY)
    logger.info(f"Summary context: {len(summary_context)} messages")

    relevant_context = context_manager.get_context(messages, ContextType.RELEVANT)
    logger.info(f"Relevant context: {len(relevant_context)} messages")

    minimal_context = context_manager.get_context(messages, ContextType.MINIMAL)
    logger.info(f"Minimal context: {len(minimal_context)} messages")


def example_summarization():
    """Demonstrate message summarization."""
    logger.info("\n=== Message Summarization ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    original_tokens = context_manager.get_token_usage(messages)
    logger.info(f"Original tokens: {original_tokens}")

    summary = context_manager.summarize_messages(messages)
    logger.info(
        f"Summary tokens: {context_manager.estimate_tokens(summary.summary_text)}"
    )
    logger.info(f"Token reduction: {summary.token_reduction:.1%}")
    logger.info(f"Key points: {len(summary.key_points)}")
    logger.info(f"Summary:\n{summary.summary_text}")


def example_compression():
    """Demonstrate context compression."""
    logger.info("\n=== Context Compression ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": "Hello"})
        messages.append({"role": "user", "content": "Hello"})

    original_count = len(messages)
    compressed = context_manager.compress_context(messages)

    logger.info(f"Original messages: {original_count}")
    logger.info(f"Compressed messages: {len(compressed)}")
    logger.info(f"Reduction: {(1 - len(compressed) / original_count):.1%}")


def example_relevance_retrieval():
    """Demonstrate relevant context retrieval."""
    logger.info("\n=== Relevance Retrieval ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    query = "machine learning"
    relevant = context_manager.retrieve_relevant_context(query, messages)

    logger.info(f"Query: {query}")
    logger.info(f"Relevant messages found: {len(relevant)}")
    for msg in relevant:
        logger.info(f"  - {msg.get('role')}: {msg.get('content')[:50]}...")


def example_importance_scoring():
    """Demonstrate importance scoring."""
    logger.info("\n=== Importance Scoring ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    test_messages = [
        {"role": "user", "content": "This is critical and urgent!"},
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "Error: Something failed badly"},
        {"role": "tool", "content": "Success: Done"},
        {"role": "assistant", "content": "Final Answer: The solution is..."},
    ]

    for msg in test_messages:
        score = context_manager.calculate_importance(msg)
        logger.info(
            f"{msg['role']}: {msg['content'][:30]}... -> "
            f"Importance: {score.importance_score:.2f}, "
            f"Final: {score.final_score:.2f}"
        )


def example_analysis():
    """Demonstrate context analysis."""
    logger.info("\n=== Context Analysis ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    analysis = context_manager.analyze_context(messages)

    logger.info(f"Total messages: {analysis.total_messages}")
    logger.info(f"Total tokens: {analysis.total_tokens}")
    logger.info(f"Redundant messages: {analysis.redundant_messages}")
    logger.info(f"Compression potential: {analysis.compression_potential:.1%}")
    logger.info(f"Quality score: {analysis.quality_score:.2f}")
    logger.info(f"Gaps: {analysis.gaps}")
    logger.info(f"Suggestions: {analysis.suggestions}")


def example_essential_messages():
    """Demonstrate extracting essential messages."""
    logger.info("\n=== Essential Messages ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    essential = context_manager.get_essential_messages(messages)

    logger.info(f"Original messages: {len(messages)}")
    logger.info(f"Essential messages: {len(essential)}")
    logger.info(f"Reduction: {(1 - len(essential) / len(messages)):.1%}")


def example_with_tool_orchestrator():
    """Demonstrate integration with ToolOrchestrator."""
    logger.info("\n=== Integration with ToolOrchestrator ===")

    db_manager = DatabaseManager()
    config = ContextConfig(max_context_tokens=4000)
    context_manager = ContextManager(config=config, db_manager=db_manager)

    logger.info("ContextManager configured for ToolOrchestrator integration")
    logger.info(f"Max context tokens: {config.max_context_tokens}")
    logger.info(f"Summary threshold: {config.summary_threshold}")
    logger.info(f"Compression strategy: {config.compression_strategy.value}")


def example_usage_statistics():
    """Demonstrate usage statistics."""
    logger.info("\n=== Usage Statistics ===")

    config = ContextConfig()
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    context_manager.manage_context(messages)
    context_manager.summarize_messages(messages)
    context_manager.compress_context(messages)

    stats = context_manager.get_usage_statistics()

    logger.info(f"Cache size: {stats['cache_size']}")
    logger.info(f"Summary cache size: {stats['summary_cache_size']}")
    logger.info(f"Total usage patterns: {stats['total_usage_patterns']}")
    logger.info(f"Recent token history entries: {len(stats['recent_token_history'])}")


def example_optimization_workflow():
    """Demonstrate complete optimization workflow."""
    logger.info("\n=== Complete Optimization Workflow ===")

    config = ContextConfig(
        max_context_tokens=2000,
        summary_threshold=1000,
        compression_strategy=CompressionStrategy.HYBRID,
    )
    context_manager = ContextManager(config=config)

    messages = create_sample_conversation()

    logger.info("Step 1: Analyze context")
    analysis = context_manager.analyze_context(messages)
    logger.info(f"  Quality score: {analysis.quality_score:.2f}")
    logger.info(f"  Compression potential: {analysis.compression_potential:.1%}")

    logger.info("\nStep 2: Compress context")
    compressed = context_manager.compress_context(messages)
    logger.info(f"  Reduced from {len(messages)} to {len(compressed)} messages")

    logger.info("\nStep 3: Optimize for tokens")
    optimized = context_manager.optimize_for_tokens(compressed, max_tokens=2000)
    logger.info(f"  Final token count: {context_manager.get_token_usage(optimized)}")

    logger.info("\nStep 4: Get minimal context")
    minimal = context_manager.get_context(optimized, ContextType.MINIMAL)
    logger.info(f"  Essential messages: {len(minimal)}")

    logger.info("\nOptimization complete!")
    logger.info(
        f"Total reduction: "
        f"{(1 - context_manager.get_token_usage(minimal) / context_manager.get_token_usage(messages)):.1%}"
    )


def main():
    """Run all examples."""
    logger.info("ContextManager Examples\n")

    example_basic_usage()
    example_context_types()
    example_summarization()
    example_compression()
    example_relevance_retrieval()
    example_importance_scoring()
    example_analysis()
    example_essential_messages()
    example_with_tool_orchestrator()
    example_usage_statistics()
    example_optimization_workflow()

    logger.info("\n=== All Examples Complete ===")


if __name__ == "__main__":
    main()
