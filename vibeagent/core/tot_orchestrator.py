"""Tree of Thoughts orchestrator for multi-branch reasoning and exploration.

This module implements the Tree of Thoughts (ToT) paradigm, which allows the agent
to explore multiple reasoning paths in parallel, evaluate them, and select the best
approach for complex problem-solving tasks.
"""

import heapq
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .skill import BaseSkill, SkillResult
from .tool_orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Tree exploration strategies."""

    BFS = "bfs"
    DFS = "dfs"
    BEST_FIRST = "best_first"
    BEAM_SEARCH = "beam_search"


@dataclass
class ThoughtNode:
    """A node in the thought tree representing a reasoning branch."""

    node_id: str
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    score: float = 0.0
    confidence: float = 0.5
    depth: int = 0
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    status: str = "pending"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is root (no parent)."""
        return self.parent_id is None

    def get_path_to_root(self, tree: "ThoughtTree") -> list["ThoughtNode"]:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current:
            path.append(current)
            if current.parent_id is None:
                break
            current = tree.get_node(current.parent_id)
        return list(reversed(path))


@dataclass
class ThoughtTree:
    """A tree structure for storing and managing thought nodes."""

    root_id: str
    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    max_depth: int = 10
    max_nodes: int = 100
    created_at: float = field(default_factory=time.time)

    def add_node(self, node: ThoughtNode) -> bool:
        """Add a node to the tree."""
        if len(self.nodes) >= self.max_nodes:
            return False
        if node.depth > self.max_depth:
            return False
        self.nodes[node.node_id] = node
        return True

    def get_node(self, node_id: str) -> ThoughtNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_root(self) -> ThoughtNode | None:
        """Get the root node."""
        return self.nodes.get(self.root_id)

    def get_children(self, node_id: str) -> list[ThoughtNode]:
        """Get all children of a node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes.get(child_id) for child_id in node.children if child_id in self.nodes]

    def get_leaves(self) -> list[ThoughtNode]:
        """Get all leaf nodes."""
        return [node for node in self.nodes.values() if node.is_leaf()]

    def get_all_paths(self) -> list[list[ThoughtNode]]:
        """Get all paths from root to leaves."""
        root = self.get_root()
        if not root:
            return []

        paths = []
        stack = [(root, [root])]

        while stack:
            current, path = stack.pop()
            children = self.get_children(current.node_id)

            if not children:
                paths.append(path)
            else:
                for child in children:
                    stack.append((child, path + [child]))

        return paths

    def prune_node(self, node_id: str) -> bool:
        """Remove a node and all its descendants."""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent and node_id in parent.children:
                parent.children.remove(node_id)

        to_remove = [node_id]
        queue = deque([node_id])

        while queue:
            current = queue.popleft()
            children = self.nodes.get(current, ThoughtNode("", "")).children
            to_remove.extend(children)
            queue.extend(children)

        for node_id in to_remove:
            self.nodes.pop(node_id, None)

        return True

    def get_stats(self) -> dict:
        """Get tree statistics."""
        depths = [node.depth for node in self.nodes.values()]
        return {
            "total_nodes": len(self.nodes),
            "max_depth_reached": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "leaf_count": len(self.get_leaves()),
            "root_score": self.get_root().score if self.get_root() else 0,
        }


@dataclass
class ToTConfig:
    """Configuration for Tree of Thoughts orchestrator."""

    max_tree_depth: int = 10
    branching_factor: int = 3
    max_nodes: int = 100
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.BEST_FIRST
    beam_width: int = 3
    pruning_threshold: float = 0.3
    evaluation_timeout: int = 30
    enable_backtracking: bool = True
    enable_visualization: bool = True
    fallback_to_sequential: bool = True


@dataclass
class ToTResult:
    """Result from Tree of Thoughts orchestration."""

    success: bool
    final_response: str
    selected_path: list[ThoughtNode]
    tree_stats: dict
    iterations: int
    tool_calls_made: int
    tool_results: list[dict]
    reasoning_trace: list[dict]
    error: str | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TreeOfThoughtsOrchestrator(ToolOrchestrator):
    """Orchestrator implementing Tree of Thoughts reasoning paradigm.

    This orchestrator extends ToolOrchestrator to support multi-branch reasoning
    by exploring multiple thought paths in parallel, evaluating them, and selecting
    the best approach for complex problem-solving.
    """

    def __init__(
        self,
        llm_skill,
        skills: dict[str, BaseSkill],
        db_manager=None,
        tot_config: ToTConfig | None = None,
    ):
        """Initialize the ToT orchestrator.

        Args:
            llm_skill: LLMSkill instance for interacting with LLM
            skills: Dictionary of skill_name -> BaseSkill for available tools
            db_manager: Optional DatabaseManager for tracking operations
            tot_config: Optional ToTConfig for Tree of Thoughts behavior
        """
        super().__init__(llm_skill, skills, db_manager, use_react=True)

        self.tot_config = tot_config or ToTConfig()
        self.current_tree: ThoughtTree | None = None
        self.session_db_id: int | None = None
        self.node_db_ids: dict[str, int] = {}

        logger.info(
            f"TreeOfThoughtsOrchestrator initialized with strategy: "
            f"{self.tot_config.exploration_strategy.value}"
        )

    def execute_with_tot(self, user_message: str, max_iterations: int = 20) -> ToTResult:
        """Execute user message using Tree of Thoughts reasoning.

        Args:
            user_message: The user's message to process
            max_iterations: Maximum number of reasoning iterations

        Returns:
            ToTResult with final response and tree statistics
        """
        session_id_str = str(uuid.uuid4())
        start_time = time.time()

        reasoning_trace = []
        tool_calls_made = 0
        tool_results = []

        try:
            if self.db_manager:
                self.session_db_id = self.db_manager.create_session(
                    session_id=session_id_str,
                    session_type="tot_orchestration",
                    model=self.llm_skill.model,
                    orchestrator_type="TreeOfThoughtsOrchestrator",
                    metadata={
                        "max_iterations": max_iterations,
                        "tot_config": {
                            "max_tree_depth": self.tot_config.max_tree_depth,
                            "branching_factor": self.tot_config.branching_factor,
                            "exploration_strategy": self.tot_config.exploration_strategy.value,
                        },
                    },
                )
                self.db_manager.add_message(
                    session_id=self.session_db_id,
                    role="user",
                    content=user_message,
                    message_index=0,
                    model=self.llm_skill.model,
                )

            root_node = self._create_root_node(user_message)
            self.current_tree = ThoughtTree(
                root_id=root_node.node_id,
                max_depth=self.tot_config.max_tree_depth,
                max_nodes=self.tot_config.max_nodes,
            )
            self.current_tree.add_node(root_node)
            self._track_node(root_node, 0)

            reasoning_trace.append(
                {
                    "iteration": 0,
                    "type": "tree_init",
                    "thought": root_node.thought,
                    "node_id": root_node.node_id,
                }
            )

            selected_path = self._explore_tree(
                root_node,
                max_iterations,
                reasoning_trace,
                tool_calls_made,
                tool_results,
            )

            if not selected_path and self.tot_config.fallback_to_sequential:
                logger.warning("ToT exploration failed, falling back to sequential execution")
                return self._fallback_to_sequential(
                    user_message,
                    reasoning_trace,
                    tool_calls_made,
                    tool_results,
                    start_time,
                )

            if not selected_path:
                return ToTResult(
                    success=False,
                    final_response="",
                    selected_path=[],
                    tree_stats=self.current_tree.get_stats(),
                    iterations=len(reasoning_trace),
                    tool_calls_made=tool_calls_made,
                    tool_results=tool_results,
                    reasoning_trace=reasoning_trace,
                    error="Tree exploration failed to find a valid path",
                )

            final_response = self._generate_final_response(selected_path, user_message)

            if self.db_manager and self.session_db_id:
                try:
                    self.db_manager.update_session(
                        self.session_db_id,
                        final_status="completed",
                        total_duration_ms=int((time.time() - start_time) * 1000),
                        total_iterations=len(reasoning_trace),
                        total_tool_calls=tool_calls_made,
                    )
                    self._store_tree_visualization()
                except Exception as e:
                    logger.error(f"Failed to update session: {e}")

            return ToTResult(
                success=True,
                final_response=final_response,
                selected_path=selected_path,
                tree_stats=self.current_tree.get_stats(),
                iterations=len(reasoning_trace),
                tool_calls_made=tool_calls_made,
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
            )

        except Exception as e:
            logger.error(f"ToT orchestration failed: {e}", exc_info=True)
            if self.db_manager and self.session_db_id:
                try:
                    self.db_manager.update_session(
                        self.session_db_id,
                        final_status="error",
                        total_duration_ms=int((time.time() - start_time) * 1000),
                        total_iterations=len(reasoning_trace),
                        total_tool_calls=tool_calls_made,
                        error_message=str(e),
                    )
                except Exception as db_error:
                    logger.error(f"Failed to update session on error: {db_error}")

            return ToTResult(
                success=False,
                final_response="",
                selected_path=[],
                tree_stats=self.current_tree.get_stats() if self.current_tree else {},
                iterations=len(reasoning_trace),
                tool_calls_made=tool_calls_made,
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                error=f"ToT orchestration failed: {str(e)}",
            )

    def _create_root_node(self, user_message: str) -> ThoughtNode:
        """Create the root thought node from user message."""
        thought_prompt = f"""Analyze the following user request and generate an initial thought about how to approach it.

User request: {user_message}

Provide a brief initial thought about the approach needed. Consider:
- What is the user asking for?
- What tools or information might be needed?
- What are the main steps to solve this?

Thought:"""

        try:
            response = self._call_llm_for_thought(thought_prompt)
            if response.success:
                initial_thought = response.data.get("content", user_message)
            else:
                initial_thought = f"Need to address: {user_message}"
        except Exception as e:
            logger.warning(f"Failed to generate initial thought: {e}")
            initial_thought = f"Need to address: {user_message}"

        return ThoughtNode(
            node_id=str(uuid.uuid4()),
            thought=initial_thought,
            depth=0,
            score=0.5,
            confidence=0.5,
            status="pending",
            metadata={"user_message": user_message},
        )

    def _explore_tree(
        self,
        root: ThoughtNode,
        max_iterations: int,
        reasoning_trace: list[dict],
        tool_calls_made: int,
        tool_results: list[dict],
    ) -> list[ThoughtNode] | None:
        """Explore the thought tree using configured strategy.

        Args:
            root: Root node of the tree
            max_iterations: Maximum iterations
            reasoning_trace: Trace to update
            tool_calls_made: Tool call counter
            tool_results: Tool results list

        Returns:
            Selected path or None
        """
        strategy = self.tot_config.exploration_strategy

        if strategy == ExplorationStrategy.BFS:
            return self._explore_bfs(
                root, max_iterations, reasoning_trace, tool_calls_made, tool_results
            )
        if strategy == ExplorationStrategy.DFS:
            return self._explore_dfs(
                root, max_iterations, reasoning_trace, tool_calls_made, tool_results
            )
        if strategy == ExplorationStrategy.BEST_FIRST:
            return self._explore_best_first(
                root, max_iterations, reasoning_trace, tool_calls_made, tool_results
            )
        if strategy == ExplorationStrategy.BEAM_SEARCH:
            return self._explore_beam_search(
                root, max_iterations, reasoning_trace, tool_calls_made, tool_results
            )
        return self._explore_best_first(
            root, max_iterations, reasoning_trace, tool_calls_made, tool_results
        )

    def _explore_bfs(
        self,
        root: ThoughtNode,
        max_iterations: int,
        reasoning_trace: list[dict],
        tool_calls_made: int,
        tool_results: list[dict],
    ) -> list[ThoughtNode] | None:
        """Breadth-First Search exploration."""
        queue = deque([(root, 0)])
        visited = {root.node_id}
        iteration = 0

        while queue and iteration < max_iterations:
            current, depth = queue.popleft()
            iteration += 1

            branches = self._generate_branches(current, self.tot_config.branching_factor)
            reasoning_trace.append(
                {
                    "iteration": iteration,
                    "type": "branch_generation",
                    "node_id": current.node_id,
                    "branch_count": len(branches),
                }
            )

            for branch in branches:
                if branch.score < self.tot_config.pruning_threshold:
                    continue

                if not self.current_tree.add_node(branch):
                    continue

                current.children.append(branch.node_id)
                self._track_node(branch, iteration)

                if branch.status == "complete":
                    return branch.get_path_to_root(self.current_tree)

                if branch.depth < self.tot_config.max_tree_depth:
                    queue.append((branch, depth + 1))

            iteration += 1

        return self._select_best_path()

    def _explore_dfs(
        self,
        root: ThoughtNode,
        max_iterations: int,
        reasoning_trace: list[dict],
        tool_calls_made: int,
        tool_results: list[dict],
    ) -> list[ThoughtNode] | None:
        """Depth-First Search exploration."""
        stack = [(root, 0)]
        iteration = 0

        while stack and iteration < max_iterations:
            current, depth = stack.pop()
            iteration += 1

            branches = self._generate_branches(current, self.tot_config.branching_factor)
            reasoning_trace.append(
                {
                    "iteration": iteration,
                    "type": "branch_generation",
                    "node_id": current.node_id,
                    "branch_count": len(branches),
                }
            )

            for branch in reversed(branches):
                if branch.score < self.tot_config.pruning_threshold:
                    continue

                if not self.current_tree.add_node(branch):
                    continue

                current.children.append(branch.node_id)
                self._track_node(branch, iteration)

                if branch.status == "complete":
                    return branch.get_path_to_root(self.current_tree)

                if branch.depth < self.tot_config.max_tree_depth:
                    stack.append((branch, depth + 1))

            iteration += 1

        return self._select_best_path()

    def _explore_best_first(
        self,
        root: ThoughtNode,
        max_iterations: int,
        reasoning_trace: list[dict],
        tool_calls_made: int,
        tool_results: list[dict],
    ) -> list[ThoughtNode] | None:
        """Best-First Search exploration using node scores."""
        heap = [(-root.score, root.node_id, root)]
        iteration = 0

        while heap and iteration < max_iterations:
            _, _, current = heapq.heappop(heap)
            iteration += 1

            branches = self._generate_branches(current, self.tot_config.branching_factor)
            reasoning_trace.append(
                {
                    "iteration": iteration,
                    "type": "branch_generation",
                    "node_id": current.node_id,
                    "branch_count": len(branches),
                }
            )

            for branch in branches:
                if branch.score < self.tot_config.pruning_threshold:
                    continue

                if not self.current_tree.add_node(branch):
                    continue

                current.children.append(branch.node_id)
                self._track_node(branch, iteration)

                if branch.status == "complete":
                    return branch.get_path_to_root(self.current_tree)

                if branch.depth < self.tot_config.max_tree_depth:
                    heapq.heappush(heap, (-branch.score, branch.node_id, branch))

            iteration += 1

        return self._select_best_path()

    def _explore_beam_search(
        self,
        root: ThoughtNode,
        max_iterations: int,
        reasoning_trace: list[dict],
        tool_calls_made: int,
        tool_results: list[dict],
    ) -> list[ThoughtNode] | None:
        """Beam Search exploration keeping top K branches."""
        beam = [root]
        iteration = 0

        while beam and iteration < max_iterations:
            new_beam = []

            for current in beam:
                branches = self._generate_branches(current, self.tot_config.branching_factor)

                for branch in branches:
                    if branch.score < self.tot_config.pruning_threshold:
                        continue

                    if not self.current_tree.add_node(branch):
                        continue

                    current.children.append(branch.node_id)
                    self._track_node(branch, iteration)

                    if branch.status == "complete":
                        return branch.get_path_to_root(self.current_tree)

                    if branch.depth < self.tot_config.max_tree_depth:
                        new_beam.append(branch)

            beam = sorted(new_beam, key=lambda x: x.score, reverse=True)[
                : self.tot_config.beam_width
            ]
            iteration += 1

        return self._select_best_path()

    def _generate_branches(self, node: ThoughtNode, num_branches: int) -> list[ThoughtNode]:
        """Generate multiple thought branches from a node.

        Args:
            node: Parent node
            num_branches: Number of branches to generate

        Returns:
            List of child nodes
        """
        branch_prompt = f"""Given the current thought, generate {num_branches} alternative next steps.

Current thought: {node.thought}
Current depth: {node.depth}
Previous observation: {node.observation or "None"}

Generate {num_branches} different approaches or next thoughts. Each should:
1. Be distinct from the others
2. Make progress toward solving the problem
3. Consider alternative strategies

Format as JSON array:
[
  {{"thought": "First approach", "action": "tool_name", "confidence": 0.8}},
  {{"thought": "Second approach", "action": "tool_name", "confidence": 0.7}}
]"""

        try:
            response = self._call_llm_for_thought(branch_prompt)
            if not response.success:
                return []

            content = response.data.get("content", "")
            branches_data = json.loads(content)

            if not isinstance(branches_data, list):
                return []

            branches = []
            for i, branch_data in enumerate(branches_data[:num_branches]):
                child_node = ThoughtNode(
                    node_id=str(uuid.uuid4()),
                    thought=branch_data.get("thought", f"Alternative approach {i + 1}"),
                    action=branch_data.get("action"),
                    depth=node.depth + 1,
                    parent_id=node.node_id,
                    confidence=branch_data.get("confidence", 0.5),
                    status="pending",
                )

                child_node.score = self._evaluate_branch(child_node, node)
                branches.append(child_node)

            return branches

        except Exception as e:
            logger.warning(f"Failed to generate branches: {e}")
            return []

    def _evaluate_branch(self, node: ThoughtNode, parent: ThoughtNode) -> float:
        """Evaluate the quality of a branch.

        Args:
            node: Node to evaluate
            parent: Parent node for context

        Returns:
            Score between 0 and 1
        """
        evaluation_prompt = f"""Evaluate the quality and potential of this thought branch.

Parent thought: {parent.thought}
Current thought: {node.thought}
Proposed action: {node.action or "None"}

Evaluate on a scale of 0.0 to 1.0 based on:
1. Progress toward goal (0.3)
2. Feasibility of action (0.3)
3. Confidence in approach (0.2)
4. Novelty/creativity (0.2)

Return JSON: {{"score": 0.8, "reasoning": "Brief explanation"}}"""

        try:
            response = self._call_llm_for_thought(evaluation_prompt)
            if response.success:
                content = response.data.get("content", "")
                result = json.loads(content)
                return min(max(result.get("score", 0.5), 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Failed to evaluate branch: {e}")

        return node.confidence

    def _select_best_path(self) -> list[ThoughtNode] | None:
        """Select the best reasoning path from the tree.

        Returns:
            Best path or None
        """
        if not self.current_tree:
            return None

        paths = self.current_tree.get_all_paths()

        if not paths:
            return None

        def path_score(path: list[ThoughtNode]) -> float:
            if not path:
                return 0.0

            avg_score = sum(node.score for node in path) / len(path)
            depth_bonus = 1.0 / (len(path) + 1)
            leaf_score = path[-1].score if path else 0.0

            return (avg_score * 0.5) + (leaf_score * 0.3) + (depth_bonus * 0.2)

        best_path = max(paths, key=path_score)

        return best_path

    def _backtrack(self, node: ThoughtNode) -> ThoughtNode | None:
        """Backtrack from a failed node to try alternative branches.

        Args:
            node: Failed node

        Returns:
            Alternative node to try or None
        """
        if not self.tot_config.enable_backtracking:
            return None

        parent = self.current_tree.get_node(node.parent_id) if node.parent_id else None
        if not parent:
            return None

        siblings = [
            child
            for child in self.current_tree.get_children(parent.node_id)
            if child.node_id != node.node_id
        ]

        if not siblings:
            return self._backtrack(parent)

        best_sibling = max(siblings, key=lambda x: x.score)
        return best_sibling

    def _prune_tree(self) -> int:
        """Prune low-scoring branches from the tree.

        Returns:
            Number of nodes pruned
        """
        if not self.current_tree:
            return 0

        pruned = 0
        nodes_to_check = list(self.current_tree.nodes.values())

        for node in nodes_to_check:
            if node.score < self.tot_config.pruning_threshold and not node.is_root():
                if self.current_tree.prune_node(node.node_id):
                    pruned += 1

        return pruned

    def _track_node(self, node: ThoughtNode, iteration: int):
        """Track a thought node in the database.

        Args:
            node: Node to track
            iteration: Iteration number
        """
        if not self.db_manager or not self.session_db_id:
            return

        try:
            step_id = self.db_manager.add_reasoning_step(
                session_id=self.session_db_id,
                iteration=iteration,
                step_type="thought_node",
                content=json.dumps(
                    {
                        "node_id": node.node_id,
                        "thought": node.thought,
                        "action": node.action,
                        "score": node.score,
                        "depth": node.depth,
                    }
                ),
                metadata={
                    "parent_id": node.parent_id,
                    "confidence": node.confidence,
                },
            )
            self.node_db_ids[node.node_id] = step_id
        except Exception as e:
            logger.error(f"Failed to track node: {e}")

    def _store_tree_visualization(self):
        """Store tree visualization in database."""
        if not self.db_manager or not self.session_db_id or not self.current_tree:
            return

        try:
            visualization = self.visualize_tree()
            self.db_manager.add_reasoning_step(
                session_id=self.session_db_id,
                iteration=0,
                step_type="tree_visualization",
                content=visualization,
                metadata={
                    "tree_stats": self.current_tree.get_stats(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to store tree visualization: {e}")

    def visualize_tree(self) -> str:
        """Generate a text-based tree visualization.

        Returns:
            String representation of the tree
        """
        if not self.current_tree:
            return "No tree to visualize"

        lines = []
        lines.append("Tree of Thoughts Visualization")
        lines.append("=" * 50)

        def print_node(node: ThoughtNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}[{node.score:.2f}] {node.thought[:60]}...")

            children = self.current_tree.get_children(node.node_id)
            if children:
                sorted_children = sorted(children, key=lambda x: x.score, reverse=True)
                for i, child in enumerate(sorted_children):
                    is_last_child = i == len(sorted_children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print_node(child, new_prefix, is_last_child)

        root = self.current_tree.get_root()
        if root:
            print_node(root)

        lines.append("=" * 50)
        lines.append(f"Total nodes: {len(self.current_tree.nodes)}")
        lines.append(f"Max depth: {self.current_tree.get_stats()['max_depth_reached']}")
        lines.append(f"Leaves: {len(self.current_tree.get_leaves())}")

        return "\n".join(lines)

    def _generate_final_response(self, path: list[ThoughtNode], user_message: str) -> str:
        """Generate final response from selected path.

        Args:
            path: Selected reasoning path
            user_message: Original user message

        Returns:
            Final response string
        """
        path_summary = "\n".join([f"Step {i + 1}: {node.thought}" for i, node in enumerate(path)])

        response_prompt = f"""Based on the following reasoning path, provide a final response to the user.

User request: {user_message}

Reasoning path:
{path_summary}

Provide a clear, comprehensive final answer that addresses the user's request based on the reasoning above."""

        try:
            response = self._call_llm_for_thought(response_prompt)
            if response.success:
                return response.data.get("content", "Unable to generate response")
        except Exception as e:
            logger.warning(f"Failed to generate final response: {e}")

        return path[-1].thought if path else "Unable to complete the task"

    def _call_llm_for_thought(self, prompt: str) -> SkillResult:
        """Call LLM for thought generation.

        Args:
            prompt: Prompt to send

        Returns:
            SkillResult with LLM response
        """
        try:
            import requests

            chat_url = f"{self.llm_skill.base_url}/chat/completions"

            payload = {
                "model": self.llm_skill.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
            }

            response = requests.post(chat_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            return SkillResult(
                success=True,
                data={
                    "content": content,
                    "model": self.llm_skill.model,
                    "usage": result.get("usage", {}),
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"LLM request failed: {str(e)}")

    def _fallback_to_sequential(
        self,
        user_message: str,
        reasoning_trace: list[dict],
        tool_calls_made: int,
        tool_results: list[dict],
        start_time: float,
    ) -> ToTResult:
        """Fallback to sequential tool orchestration.

        Args:
            user_message: User message
            reasoning_trace: Existing trace
            tool_calls_made: Counter
            tool_results: Results list
            start_time: Start time

        Returns:
            ToTResult from sequential execution
        """
        logger.info("Falling back to sequential ToolOrchestrator")

        try:
            result = self.execute_with_tools(user_message, max_iterations=10, use_react=True)

            return ToTResult(
                success=result.success,
                final_response=result.final_response,
                selected_path=[],
                tree_stats={},
                iterations=result.iterations,
                tool_calls_made=result.tool_calls_made,
                tool_results=result.tool_results,
                reasoning_trace=reasoning_trace + (result.reasoning_trace or []),
                error=result.error,
                metadata={"fallback": True, "original_method": "ToT"},
            )
        except Exception as e:
            return ToTResult(
                success=False,
                final_response="",
                selected_path=[],
                tree_stats={},
                iterations=len(reasoning_trace),
                tool_calls_made=tool_calls_made,
                tool_results=tool_results,
                reasoning_trace=reasoning_trace,
                error=f"Fallback failed: {str(e)}",
            )
