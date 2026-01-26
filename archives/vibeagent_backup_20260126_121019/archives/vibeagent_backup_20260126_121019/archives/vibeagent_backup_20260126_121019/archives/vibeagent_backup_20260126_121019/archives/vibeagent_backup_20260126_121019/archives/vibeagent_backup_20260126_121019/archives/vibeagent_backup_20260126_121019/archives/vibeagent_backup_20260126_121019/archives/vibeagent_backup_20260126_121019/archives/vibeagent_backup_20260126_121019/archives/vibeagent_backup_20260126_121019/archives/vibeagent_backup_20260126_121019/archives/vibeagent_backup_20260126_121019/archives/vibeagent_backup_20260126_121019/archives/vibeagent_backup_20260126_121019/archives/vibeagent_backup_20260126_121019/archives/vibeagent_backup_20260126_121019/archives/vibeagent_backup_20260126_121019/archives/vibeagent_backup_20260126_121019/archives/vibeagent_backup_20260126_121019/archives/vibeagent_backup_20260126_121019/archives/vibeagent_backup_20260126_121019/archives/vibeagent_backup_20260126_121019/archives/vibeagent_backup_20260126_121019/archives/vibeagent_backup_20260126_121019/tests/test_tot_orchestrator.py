"""Unit tests for Tree of Thoughts orchestrator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tot_orchestrator import (
    ThoughtNode,
    ThoughtTree,
    ToTConfig,
    ExplorationStrategy,
)


def test_thought_node():
    """Test ThoughtNode creation and methods."""
    print("Testing ThoughtNode...")

    node = ThoughtNode(
        node_id="node1",
        thought="Initial thought about the problem",
        action="search",
        depth=0,
        score=0.8,
        confidence=0.7,
    )

    assert node.node_id == "node1"
    assert node.thought == "Initial thought about the problem"
    assert node.action == "search"
    assert node.depth == 0
    assert node.score == 0.8
    assert node.confidence == 0.7
    assert node.is_root()
    assert node.is_leaf()

    print("  ✅ ThoughtNode creation works")
    print("  ✅ ThoughtNode methods work")


def test_thought_tree():
    """Test ThoughtTree creation and management."""
    print("\nTesting ThoughtTree...")

    root = ThoughtNode(
        node_id="root",
        thought="Root thought",
        depth=0,
        score=0.5,
    )

    tree = ThoughtTree(root_id="root", max_depth=5, max_nodes=20)
    tree.add_node(root)

    assert tree.get_root() == root
    assert len(tree.nodes) == 1
    assert tree.get_root().is_root()

    child1 = ThoughtNode(
        node_id="child1",
        thought="First branch",
        depth=1,
        parent_id="root",
        score=0.7,
    )
    child2 = ThoughtNode(
        node_id="child2",
        thought="Second branch",
        depth=1,
        parent_id="root",
        score=0.6,
    )

    tree.add_node(child1)
    tree.add_node(child2)
    root.children.extend(["child1", "child2"])

    assert len(tree.nodes) == 3
    assert len(tree.get_children("root")) == 2
    assert len(tree.get_leaves()) == 2

    stats = tree.get_stats()
    assert stats["total_nodes"] == 3
    assert stats["max_depth_reached"] == 1
    assert stats["leaf_count"] == 2

    print("  ✅ ThoughtTree creation works")
    print("  ✅ ThoughtTree node management works")
    print("  ✅ ThoughtTree statistics work")


def test_tree_pruning():
    """Test tree pruning functionality."""
    print("\nTesting tree pruning...")

    root = ThoughtNode(
        node_id="root",
        thought="Root",
        depth=0,
        score=0.8,
    )

    tree = ThoughtTree(root_id="root", max_depth=5, max_nodes=20)
    tree.add_node(root)

    low_score_child = ThoughtNode(
        node_id="low",
        thought="Low score branch",
        depth=1,
        parent_id="root",
        score=0.2,
    )
    high_score_child = ThoughtNode(
        node_id="high",
        thought="High score branch",
        depth=1,
        parent_id="root",
        score=0.9,
    )

    tree.add_node(low_score_child)
    tree.add_node(high_score_child)
    root.children.extend(["low", "high"])

    assert len(tree.nodes) == 3

    tree.prune_node("low")

    assert len(tree.nodes) == 2
    assert "low" not in tree.nodes
    assert "high" in tree.nodes

    print("  ✅ Tree pruning works")


def test_path_traversal():
    """Test path traversal in tree."""
    print("\nTesting path traversal...")

    root = ThoughtNode(
        node_id="root",
        thought="Root",
        depth=0,
        score=0.5,
    )

    tree = ThoughtTree(root_id="root", max_depth=5, max_nodes=20)
    tree.add_node(root)

    child = ThoughtNode(
        node_id="child",
        thought="Child",
        depth=1,
        parent_id="root",
        score=0.7,
    )

    tree.add_node(child)
    root.children.append("child")

    path = child.get_path_to_root(tree)
    assert len(path) == 2
    assert path[0] == root
    assert path[1] == child

    all_paths = tree.get_all_paths()
    assert len(all_paths) == 1
    assert len(all_paths[0]) == 2

    print("  ✅ Path traversal works")
    print("  ✅ All paths retrieval works")


def test_tot_config():
    """Test ToTConfig creation."""
    print("\nTesting ToTConfig...")

    config = ToTConfig(
        max_tree_depth=8,
        branching_factor=3,
        exploration_strategy=ExplorationStrategy.BEAM_SEARCH,
        beam_width=4,
    )

    assert config.max_tree_depth == 8
    assert config.branching_factor == 3
    assert config.exploration_strategy == ExplorationStrategy.BEAM_SEARCH
    assert config.beam_width == 4
    assert config.enable_backtracking == True

    print("  ✅ ToTConfig creation works")


def test_exploration_strategy_enum():
    """Test ExplorationStrategy enum."""
    print("\nTesting ExplorationStrategy enum...")

    assert ExplorationStrategy.BFS.value == "bfs"
    assert ExplorationStrategy.DFS.value == "dfs"
    assert ExplorationStrategy.BEST_FIRST.value == "best_first"
    assert ExplorationStrategy.BEAM_SEARCH.value == "beam_search"

    strategies = [s for s in ExplorationStrategy]
    assert len(strategies) == 4

    print("  ✅ ExplorationStrategy enum works")


def test_node_scoring():
    """Test node scoring logic."""
    print("\nTesting node scoring...")

    node1 = ThoughtNode(
        node_id="node1",
        thought="High confidence thought",
        depth=0,
        score=0.9,
        confidence=0.95,
    )

    node2 = ThoughtNode(
        node_id="node2",
        thought="Low confidence thought",
        depth=0,
        score=0.3,
        confidence=0.4,
    )

    assert node1.score > node2.score
    assert node1.confidence > node2.confidence

    print("  ✅ Node scoring works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Tree of Thoughts Orchestrator Unit Tests")
    print("=" * 60)

    try:
        test_thought_node()
        test_thought_tree()
        test_tree_pruning()
        test_path_traversal()
        test_tot_config()
        test_exploration_strategy_enum()
        test_node_scoring()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

        return True

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
