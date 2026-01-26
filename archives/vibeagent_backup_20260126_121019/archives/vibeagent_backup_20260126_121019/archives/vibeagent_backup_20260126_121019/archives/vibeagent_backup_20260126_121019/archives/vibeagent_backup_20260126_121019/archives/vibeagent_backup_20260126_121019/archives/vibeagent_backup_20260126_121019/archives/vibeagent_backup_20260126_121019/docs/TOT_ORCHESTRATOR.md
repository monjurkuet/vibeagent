# Tree of Thoughts Orchestrator

The Tree of Thoughts (ToT) orchestrator is an advanced reasoning framework that enables multi-branch exploration and evaluation of reasoning paths. It extends the ToolOrchestrator to implement the ToT paradigm for complex problem-solving tasks.

## Overview

The ToT orchestrator improves complex reasoning success rates by:
- Exploring multiple reasoning paths in parallel
- Evaluating and scoring each branch
- Selecting the optimal path for execution
- Supporting backtracking from failed branches
- Pruning low-quality branches

## Key Components

### ThoughtNode
Represents a single reasoning step in the tree:
- `thought`: The reasoning content
- `action`: Optional tool/action to execute
- `observation`: Result from action execution
- `score`: Quality score (0.0 to 1.0)
- `confidence`: Confidence in this thought
- `depth`: Depth in the tree
- `parent_id`: Reference to parent node
- `children`: List of child node IDs

### ThoughtTree
Manages the entire thought tree structure:
- Root node and all descendants
- Tree statistics (depth, node count, etc.)
- Path traversal and selection
- Pruning operations

### ToTConfig
Configuration options for ToT behavior:
- `max_tree_depth`: Maximum tree depth (default: 10)
- `branching_factor`: Number of branches per node (default: 3)
- `max_nodes`: Maximum total nodes (default: 100)
- `exploration_strategy`: Search strategy (default: BEST_FIRST)
- `beam_width`: Width for beam search (default: 3)
- `pruning_threshold`: Minimum score to keep (default: 0.3)
- `enable_backtracking`: Enable backtracking (default: True)
- `enable_visualization`: Generate tree visualization (default: True)
- `fallback_to_sequential`: Fallback to sequential if needed (default: True)

## Exploration Strategies

### BFS (Breadth-First Search)
Explores all branches at the same depth before going deeper. Good for:
- Finding shortest paths
- Exploring all options equally
- Tasks with unknown optimal depth

### DFS (Depth-First Search)
Explores one branch deeply before trying others. Good for:
- Deep reasoning chains
- Memory-constrained environments
- Tasks with clear depth bounds

### Best-First Search
Always explores the highest-scoring branch next. Good for:
- Quality-focused exploration
- Finding optimal solutions
- Tasks with clear evaluation metrics

### Beam Search
Keeps only the top K branches at each level. Good for:
- Balancing breadth and depth
- Computational efficiency
- Large search spaces

## Usage Example

```python
from core import (
    Agent,
    DatabaseManager,
    TreeOfThoughtsOrchestrator,
    ToTConfig,
    ExplorationStrategy,
    LLMSkill,
)
from skills.arxiv_skill import ArxivSkill

# Initialize components
db_manager = DatabaseManager("data/vibeagent.db")
llm_skill = LLMSkill(
    base_url="http://localhost:11434/v1",
    model="llama3.2:latest",
)

# Create agent and register skills
agent = Agent("ToTAgent")
arxiv_skill = ArxivSkill()
agent.register_skill(arxiv_skill)

# Configure ToT
tot_config = ToTConfig(
    max_tree_depth=8,
    branching_factor=3,
    exploration_strategy=ExplorationStrategy.BEST_FIRST,
    beam_width=3,
)

# Create orchestrator
tot_orchestrator = TreeOfThoughtsOrchestrator(
    llm_skill=llm_skill,
    skills=agent.skills,
    db_manager=db_manager,
    tot_config=tot_config,
)

# Execute complex task
result = tot_orchestrator.execute_with_tot(
    "Research papers on transformers and summarize key innovations",
    max_iterations=15,
)

# Access results
print(f"Success: {result.success}")
print(f"Final response: {result.final_response}")
print(f"Tree stats: {result.tree_stats}")

# Visualize the tree
print(tot_orchestrator.visualize_tree())
```

## ToTResult Structure

```python
@dataclass
class ToTResult:
    success: bool
    final_response: str
    selected_path: List[ThoughtNode]
    tree_stats: Dict
    iterations: int
    tool_calls_made: int
    tool_results: List[Dict]
    reasoning_trace: List[Dict]
    error: Optional[str] = None
    metadata: Dict = None
```

## Key Methods

### execute_with_tot(user_message, max_iterations)
Main entry point for ToT reasoning:
- Creates initial thought tree
- Explores using configured strategy
- Evaluates and selects best path
- Generates final response

### generate_branches(node, num_branches)
Generates N alternative thought branches from a node:
- Uses LLM to generate diverse approaches
- Each branch represents a different reasoning direction
- Limits branching factor to prevent explosion

### evaluate_branch(node)
Scores the quality of a reasoning branch:
- Progress toward goal (30%)
- Feasibility of action (30%)
- Confidence in approach (20%)
- Novelty/creativity (20%)

### select_best_path()
Selects optimal reasoning path:
- Considers average path score
- Rewards shorter paths
- Values high-scoring leaf nodes

### backtrack(node)
Recovers from failed branches:
- Moves to parent node
- Tries alternative siblings
- Learns from failed paths

### prune_tree()
Removes low-quality branches:
- Prunes nodes below threshold
- Reduces tree size
- Maintains tree quality

### visualize_tree()
Generates text-based tree visualization:
- Shows thought hierarchy
- Displays branch scores
- Highlights selected path

## Database Tracking

The ToT orchestrator tracks:
- Tree structure and nodes
- Branch generation and evaluation
- Tree exploration decisions
- Tree visualizations
- Performance metrics

Stored in `reasoning_steps` table with step_type:
- `thought_node`: Individual thought nodes
- `tree_visualization`: Tree structure representation

## Performance Characteristics

### Expected Improvements
- **15% improvement** in complex reasoning success rate
- Better handling of multi-step decision making
- Exploration of alternative approaches
- Selection of optimal reasoning paths

### Complexity
- **Time**: O(b^d) where b=branching_factor, d=max_depth
- **Space**: O(n) where n=max_nodes
- **Beam Search**: O(k * d) where k=beam_width

### Optimization Tips
1. Use appropriate branching factor (2-4 recommended)
2. Set reasonable max_depth for your task
3. Enable pruning to reduce search space
4. Use Beam Search for large problems
5. Set appropriate pruning threshold (0.3-0.5)

## Configuration Guidelines

### Simple Tasks
```python
ToTConfig(
    max_tree_depth=5,
    branching_factor=2,
    exploration_strategy=ExplorationStrategy.BFS,
)
```

### Medium Complexity
```python
ToTConfig(
    max_tree_depth=8,
    branching_factor=3,
    exploration_strategy=ExplorationStrategy.BEST_FIRST,
    pruning_threshold=0.3,
)
```

### Complex Tasks
```python
ToTConfig(
    max_tree_depth=10,
    branching_factor=3,
    exploration_strategy=ExplorationStrategy.BEAM_SEARCH,
    beam_width=3,
    enable_backtracking=True,
    pruning_threshold=0.4,
)
```

## Integration with ToolOrchestrator

The ToT orchestrator extends ToolOrchestrator:
- Inherits tool execution capabilities
- Uses ReAct prompts for thought generation
- Leverages database for tracking
- Falls back to sequential if ToT fails

## Error Handling

1. **Branch generation failure**: Returns empty branch list
2. **Evaluation failure**: Uses confidence as fallback score
3. **Tree exploration failure**: Falls back to sequential execution
4. **LLM timeout**: Uses cached or default values

## Best Practices

1. **Start with Best-First** for most tasks
2. **Use Beam Search** for large search spaces
3. **Enable pruning** to control tree growth
4. **Set reasonable limits** on depth and nodes
5. **Enable visualization** for debugging
6. **Use fallback** for reliability

## Troubleshooting

### Tree grows too large
- Reduce `branching_factor`
- Lower `max_tree_depth`
- Decrease `max_nodes`
- Enable stronger pruning

### Exploration is too slow
- Use Beam Search instead of BFS/DFS
- Reduce `branching_factor`
- Lower `max_iterations`
- Increase `pruning_threshold`

### Poor path selection
- Adjust evaluation criteria
- Increase `beam_width`
- Enable backtracking
- Review scoring weights

### Memory issues
- Reduce `max_nodes`
- Use Beam Search (more memory efficient)
- Enable aggressive pruning
- Lower `branching_factor`

## Future Enhancements

- [ ] Parallel branch evaluation
- [ ] Adaptive branching factor
- [ ] Learned evaluation models
- [ ] Tree caching and reuse
- [ ] Interactive tree exploration
- [ ] Visual tree editor

## References

- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
- ReAct: Reasoning + Acting paradigm
- Beam Search in NLP
- Monte Carlo Tree Search (MCTS) inspiration