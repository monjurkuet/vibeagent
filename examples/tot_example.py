"""Example usage of Tree of Thoughts orchestrator.

This example demonstrates how to use the TreeOfThoughtsOrchestrator
for complex multi-step reasoning tasks.
"""

from vibeagent.core import (
    Agent,
    DatabaseManager,
    ExplorationStrategy,
    LLMSkill,
    ToTConfig,
    TreeOfThoughtsOrchestrator,
)
from vibeagent.skills.arxiv_skill import ArxivSkill


def main():
    """Run Tree of Thoughts example."""
    print("Tree of Thoughts Orchestrator Example")
    print("=" * 50)

    db_manager = DatabaseManager("data/vibeagent.db")

    llm_skill = LLMSkill(
        base_url="http://localhost:11434/v1",
        model="llama3.2:latest",
    )

    agent = Agent("ToTExampleAgent")
    arxiv_skill = ArxivSkill()
    agent.register_skill(arxiv_skill)

    tot_config = ToTConfig(
        max_tree_depth=8,
        branching_factor=3,
        max_nodes=50,
        exploration_strategy=ExplorationStrategy.BEST_FIRST,
        beam_width=3,
        pruning_threshold=0.3,
        enable_backtracking=True,
        enable_visualization=True,
        fallback_to_sequential=True,
    )

    tot_orchestrator = TreeOfThoughtsOrchestrator(
        llm_skill=llm_skill,
        skills=agent.skills,
        db_manager=db_manager,
        tot_config=tot_config,
    )

    print("\nConfiguration:")
    print(f"  Strategy: {tot_config.exploration_strategy.value}")
    print(f"  Max depth: {tot_config.max_tree_depth}")
    print(f"  Branching factor: {tot_config.branching_factor}")
    print(f"  Max nodes: {tot_config.max_nodes}")

    complex_task = (
        "Research recent papers on transformer architectures for computer vision, "
        "find the most cited ones from 2023, and summarize their key innovations."
    )

    print(f"\nTask: {complex_task}")

    print("\nExecuting with Tree of Thoughts...")
    result = tot_orchestrator.execute_with_tot(complex_task, max_iterations=15)

    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls made: {result.tool_calls_made}")
    print(f"Tree stats: {result.tree_stats}")

    if result.selected_path:
        print(f"\nSelected path length: {len(result.selected_path)}")
        print("Path steps:")
        for i, node in enumerate(result.selected_path):
            print(f"  {i + 1}. [{node.score:.2f}] {node.thought[:80]}...")

    if result.final_response:
        print(f"\nFinal response:\n{result.final_response}")

    if tot_config.enable_visualization and tot_orchestrator.current_tree:
        print("\n" + tot_orchestrator.visualize_tree())

    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    main()
