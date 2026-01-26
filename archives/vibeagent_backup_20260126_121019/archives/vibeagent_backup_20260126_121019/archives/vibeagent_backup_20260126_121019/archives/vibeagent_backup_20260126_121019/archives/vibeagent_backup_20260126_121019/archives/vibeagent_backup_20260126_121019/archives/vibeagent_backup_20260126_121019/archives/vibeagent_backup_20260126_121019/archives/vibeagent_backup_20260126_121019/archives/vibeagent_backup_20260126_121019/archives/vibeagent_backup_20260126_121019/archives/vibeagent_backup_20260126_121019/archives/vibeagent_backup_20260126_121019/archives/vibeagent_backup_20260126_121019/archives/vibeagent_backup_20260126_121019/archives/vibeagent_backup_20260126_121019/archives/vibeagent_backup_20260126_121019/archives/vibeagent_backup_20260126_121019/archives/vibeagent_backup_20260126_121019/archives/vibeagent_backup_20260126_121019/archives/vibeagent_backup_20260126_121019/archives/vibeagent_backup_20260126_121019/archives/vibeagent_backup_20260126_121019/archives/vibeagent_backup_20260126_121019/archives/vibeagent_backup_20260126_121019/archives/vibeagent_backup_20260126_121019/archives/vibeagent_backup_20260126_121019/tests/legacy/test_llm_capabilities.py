#!/usr/bin/env python3
"""
Comprehensive LLM Capability Test - Tests multi-tool calling with all orchestration strategies.

This test measures:
- Tool calling accuracy
- Multi-step reasoning
- Parallel execution capability
- ReAct reasoning quality
- Plan-and-Execute effectiveness
- Tree-of-Thought exploration
- Error recovery
"""

import sys
import logging
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("llm_capability_test.log"),
    ],
)

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test run."""

    test_name: str
    strategy: str
    success: bool
    iterations: int
    tool_calls_made: int
    parallel_calls_used: bool
    reasoning_steps: int
    response_time_ms: float
    final_response: str
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""

    strategy: str
    total_tests: int
    passed: int
    failed: int
    avg_iterations: float
    avg_tool_calls: float
    avg_response_time_ms: float
    parallel_usage_count: int
    total_reasoning_steps: int
    results: List[TestResult] = field(default_factory=list)


# Test cases requiring different capabilities
TEST_CASES = [
    {
        "name": "Simple Single Tool",
        "description": "Basic tool calling - single tool, single call",
        "complexity": "low",
        "expected_tools": 1,
        "expected_parallel": False,
        "prompt": "Search for papers about machine learning on arXiv",
    },
    {
        "name": "Sequential Chaining",
        "description": "Multi-step task requiring tool chaining",
        "complexity": "medium",
        "expected_tools": 2,
        "expected_parallel": False,
        "prompt": "Search for papers about neural networks and save the results to the database",
    },
    {
        "name": "Parallel Independent Tasks",
        "description": "Multiple independent tasks that can run in parallel",
        "complexity": "medium",
        "expected_tools": 2,
        "expected_parallel": True,
        "prompt": "Search for papers about reinforcement learning AND computer vision simultaneously",
    },
    {
        "name": "Complex Multi-Step",
        "description": "Complex task requiring multiple tools and steps",
        "complexity": "high",
        "expected_tools": 3,
        "expected_parallel": False,
        "prompt": "Research recent papers on transformer architectures, extract their abstracts, save to database, and summarize the key findings",
    },
    {
        "name": "Tool Selection",
        "description": "Requires choosing the right tool from multiple options",
        "complexity": "medium",
        "expected_tools": 1,
        "expected_parallel": False,
        "prompt": "I need to find information about AI research. Should I search arXiv or scrape a specific website?",
    },
    {
        "name": "Error Recovery",
        "description": "Task that may require retry or error handling",
        "complexity": "high",
        "expected_tools": 2,
        "expected_parallel": False,
        "prompt": "Try to search for papers and if that fails, try a different search term",
    },
    {
        "name": "Context-Aware",
        "description": "Requires maintaining context across multiple steps",
        "complexity": "high",
        "expected_tools": 3,
        "expected_parallel": False,
        "prompt": "First search for papers about AI, then analyze the results, then save the best ones to the database",
    },
    {
        "name": "Decision Making",
        "description": "Requires making decisions based on tool results",
        "complexity": "high",
        "expected_tools": 2,
        "expected_parallel": False,
        "prompt": "Search for papers and only save the ones published after 2023",
    },
]


class LLMAbilityTester:
    """Test LLM capabilities with different orchestration strategies."""

    def __init__(
        self, llm_base_url: str = "http://localhost:8087/v1", llm_model: str = "glm-4.7"
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.stats: Dict[str, StrategyStats] = {}
        self.all_results: List[TestResult] = []

    def get_available_models(self) -> List[str]:
        """Get list of available models from the API."""
        try:
            models_url = f"{self.llm_base_url.rstrip('/')}/models"
            response = requests.get(models_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            logger.info(f"📋 Found {len(models)} available models: {', '.join(models)}")
            return models
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch available models: {e}")
            return []

    def setup_database(self):
        """Setup database for storing results."""
        from core.database_manager import DatabaseManager

        db_path = Path("data/llm_capability_test.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_manager = DatabaseManager(db_path=str(db_path))
        logger.info(f"✅ Database initialized at {db_path}")

    def setup_skills(self):
        """Setup mock skills for testing."""
        from core.skill import BaseSkill, SkillResult
        from typing import List

        class ArxivSearchSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="arxiv_search", version="1.0.0")
                self.call_count = 0

            def execute(self, **kwargs) -> SkillResult:
                self.call_count += 1
                query = kwargs.get("query", "")
                max_results = kwargs.get("max_results", 10)
                logger.info(
                    f"   📚 arxiv_search: query='{query}', max_results={max_results}"
                )
                return SkillResult(
                    success=True,
                    data={
                        "papers": [
                            {
                                "title": f"Paper about {query}",
                                "arxiv_id": f"2401.{self.call_count:04d}",
                                "published": "2024-01-15",
                            }
                        ]
                        * max_results
                    },
                )

            def validate(self) -> bool:
                return True

            def get_dependencies(self) -> List[str]:
                return []

            def get_tool_schema(self) -> Dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "arxiv_search",
                        "description": "Search for papers on arXiv",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                },
                                "max_results": {"type": "integer", "default": 10},
                                "categories": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }

        class DatabaseSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="database", version="1.0.0")
                self.call_count = 0
                self.stored_data = []

            def execute(self, **kwargs) -> SkillResult:
                self.call_count += 1
                action = kwargs.get("action", "")
                data = kwargs.get("data", {})

                if action == "save":
                    logger.info(
                        f"   💾 database: save action, data keys={list(data.keys())}"
                    )
                    self.stored_data.append(data)
                    return SkillResult(
                        success=True,
                        data={"saved": True, "count": len(self.stored_data)},
                    )
                elif action == "query":
                    logger.info(f"   🔍 database: query action")
                    return SkillResult(success=True, data={"results": self.stored_data})
                else:
                    logger.warning(f"   ⚠️ database: unknown action '{action}'")
                    return SkillResult(success=False, error=f"Unknown action: {action}")

            def validate(self) -> bool:
                return True

            def get_dependencies(self) -> List[str]:
                return []

            def get_tool_schema(self) -> Dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "database",
                        "description": "Store or query data in database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["save", "query"],
                                    "description": "Action to perform",
                                },
                                "data": {
                                    "type": "object",
                                    "description": "Data to save (for save action)",
                                },
                                "filters": {
                                    "type": "object",
                                    "description": "Query filters (for query action)",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                }

        class ExtractSkill(BaseSkill):
            def __init__(self):
                super().__init__(name="extract", version="1.0.0")
                self.call_count = 0

            def execute(self, **kwargs) -> SkillResult:
                self.call_count += 1
                text = kwargs.get("text", "")
                field = kwargs.get("field", "")
                logger.info(f"   📝 extract: field='{field}', text length={len(text)}")
                return SkillResult(
                    success=True, data={"extracted": f"Extracted {field} from text"}
                )

            def validate(self) -> bool:
                return True

            def get_dependencies(self) -> List[str]:
                return []

            def get_tool_schema(self) -> Dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": "Extract information from text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text to extract from",
                                },
                                "field": {
                                    "type": "string",
                                    "description": "Field to extract (e.g., abstract, title)",
                                },
                            },
                            "required": ["text", "field"],
                        },
                    },
                }

        self.skills = {
            "arxiv_search": ArxivSearchSkill(),
            "database": DatabaseSkill(),
            "extract": ExtractSkill(),
        }

        logger.info(f"✅ Skills configured: {', '.join(self.skills.keys())}")
        return self.skills

    def run_test_with_strategy(self, test_case: Dict, strategy: str) -> TestResult:
        """Run a single test case with a specific orchestration strategy."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TEST: {test_case['name']}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Prompt: {test_case['prompt'][:100]}...")
        logger.info(f"{'=' * 80}")

        start_time = time.time()

        try:
            from skills import LLMSkill

            # Setup LLM skill
            llm_skill = LLMSkill(base_url=self.llm_base_url, model=self.llm_model)

            # Create orchestrator based on strategy
            if strategy == "basic":
                from core.tool_orchestrator import ToolOrchestrator

                orchestrator = ToolOrchestrator(
                    llm_skill=llm_skill,
                    skills=self.skills,
                    db_manager=self.db_manager,
                    use_react=False,
                )
            elif strategy == "react":
                from core.tool_orchestrator import ToolOrchestrator

                orchestrator = ToolOrchestrator(
                    llm_skill=llm_skill,
                    skills=self.skills,
                    db_manager=self.db_manager,
                    use_react=True,
                )
            elif strategy == "plan_execute":
                from core.plan_execute_orchestrator import PlanExecuteOrchestrator

                orchestrator = PlanExecuteOrchestrator(
                    llm_skill=llm_skill,
                    skills=self.skills,
                    db_manager=self.db_manager,
                )
            elif strategy == "tot":
                from core.tot_orchestrator import TreeOfThoughtsOrchestrator

                orchestrator = TreeOfThoughtsOrchestrator(
                    llm_skill=llm_skill,
                    skills=self.skills,
                    db_manager=self.db_manager,
                    tot_config=None,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Reset skill call counts
            for skill in self.skills.values():
                skill.call_count = 0

            # Execute test
            result = orchestrator.execute_with_tools(
                test_case["prompt"], max_iterations=10
            )

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            # Check if parallel calls were used
            total_skill_calls = sum(s.call_count for s in self.skills.values())
            parallel_used = result.iterations < total_skill_calls

            # Count reasoning steps (if available)
            reasoning_steps = 0
            if hasattr(result, "reasoning_trace") and result.reasoning_trace:
                reasoning_steps = len(result.reasoning_trace)

            # Determine success
            success = result.success

            # Build result
            test_result = TestResult(
                test_name=test_case["name"],
                strategy=strategy,
                success=success,
                iterations=result.iterations,
                tool_calls_made=result.tool_calls_made,
                parallel_calls_used=parallel_used,
                reasoning_steps=reasoning_steps,
                response_time_ms=response_time_ms,
                final_response=result.final_response or "",
                error=result.error,
                metrics={
                    "expected_tools": test_case["expected_tools"],
                    "expected_parallel": test_case["expected_parallel"],
                    "complexity": test_case["complexity"],
                    "skill_calls": {
                        name: skill.call_count for name, skill in self.skills.items()
                    },
                },
            )

            # Log results
            logger.info(f"\n📊 Results:")
            logger.info(f"   Success: {'✅' if success else '❌'}")
            logger.info(f"   Iterations: {result.iterations}")
            logger.info(f"   Tool calls: {result.tool_calls_made}")
            logger.info(f"   Parallel: {'✅' if parallel_used else '❌'}")
            logger.info(f"   Reasoning steps: {reasoning_steps}")
            logger.info(f"   Response time: {response_time_ms:.2f}ms")
            logger.info(
                f"   Final response: {result.final_response[:200] if result.final_response else 'N/A'}..."
            )

            if result.error:
                logger.warning(f"   Error: {result.error}")

            return test_result

        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            logger.error(f"❌ Test failed with exception: {e}", exc_info=True)

            return TestResult(
                test_name=test_case["name"],
                strategy=strategy,
                success=False,
                iterations=0,
                tool_calls_made=0,
                parallel_calls_used=False,
                reasoning_steps=0,
                response_time_ms=response_time_ms,
                final_response="",
                error=str(e),
            )

    def run_all_tests(
        self, strategies: Optional[List[str]] = None, test_all_models: bool = False
    ):
        """Run all test cases with all strategies.

        Args:
            strategies: List of strategies to test. If None, tests all strategies.
            test_all_models: If True, tests all available models from the API.
        """
        if strategies is None:
            strategies = ["basic", "react", "plan_execute", "tot"]

        # Determine models to test
        if test_all_models:
            models = self.get_available_models()
            if not models:
                logger.warning("⚠️ No models found, falling back to default model")
                models = [self.llm_model]
        else:
            models = [self.llm_model]

        self.setup_database()
        self.setup_skills()

        logger.info(f"\n🧪 Starting LLM Capability Test")
        logger.info(f"   LLM URL: {self.llm_base_url}")
        logger.info(f"   Models to test: {', '.join(models)}")
        logger.info(f"   Strategies: {', '.join(strategies)}")
        logger.info(f"   Test cases: {len(TEST_CASES)}")
        total_tests = len(TEST_CASES) * len(strategies) * len(models)
        logger.info(f"   Total tests: {total_tests}")

        # Run tests for each model
        for model in models:
            logger.info(f"\n\n{'=' * 80}")
            logger.info(f"🤖 TESTING MODEL: {model}")
            logger.info(f"{'=' * 80}\n")

            self.llm_model = model

            # Initialize stats for this model
            model_stats = {}
            for strategy in strategies:
                model_stats[strategy] = StrategyStats(
                    strategy=strategy,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    avg_iterations=0,
                    avg_tool_calls=0,
                    avg_response_time_ms=0,
                    parallel_usage_count=0,
                    total_reasoning_steps=0,
                )

            # Run tests for each strategy
            for strategy in strategies:
                logger.info(f"\n\n{'#' * 80}")
                logger.info(f"# STRATEGY: {strategy.upper()}")
                logger.info(f"{'#' * 80}\n")

                for test_case in TEST_CASES:
                    result = self.run_test_with_strategy(test_case, strategy)
                    self.all_results.append(result)

                    # Update stats
                    stats = model_stats[strategy]
                    stats.total_tests += 1
                    if result.success:
                        stats.passed += 1
                    else:
                        stats.failed += 1

                    stats.avg_iterations = (
                        stats.avg_iterations * (stats.total_tests - 1)
                        + result.iterations
                    ) / stats.total_tests
                    stats.avg_tool_calls = (
                        stats.avg_tool_calls * (stats.total_tests - 1)
                        + result.tool_calls_made
                    ) / stats.total_tests
                    stats.avg_response_time_ms = (
                        stats.avg_response_time_ms * (stats.total_tests - 1)
                        + result.response_time_ms
                    ) / stats.total_tests

                    if result.parallel_calls_used:
                        stats.parallel_usage_count += 1

                    stats.total_reasoning_steps += result.reasoning_steps
                    stats.results.append(result)

                    # Store in database
                    self._store_result_in_db(result)

            # Store model stats in overall stats with model prefix
            for strategy, stats in model_stats.items():
                key = f"{model}:{strategy}"
                self.stats[key] = stats

        # Generate report
        self._generate_report()

    def _store_result_in_db(self, result: TestResult):
        """Store test result in database with full metadata."""
        try:
            # Check if test case exists
            test_case_id = None
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM test_cases WHERE name = ?", (result.test_name,)
                )
                row = cursor.fetchone()
                if row:
                    test_case_id = row["id"]
                    logger.debug(f"   📊 Found existing test case: {result.test_name}")

            # Create test case if needed (outside connection block)
            if test_case_id is None:
                test_case_id = self.db_manager.create_test_case(
                    name=result.test_name,
                    category="llm_capability",
                    description=f"Test using {result.strategy} strategy",
                    messages=[{"role": "user", "content": result.test_name}],
                    tools=[],
                    metadata=result.metrics,
                )
                logger.debug(f"   📊 Created new test case: {result.test_name}")

            # Get next run number and create/update test run
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get next run number
                cursor.execute(
                    "SELECT COALESCE(MAX(run_number), 0) + 1 as next_run FROM test_runs WHERE test_case_id = ?",
                    (test_case_id,),
                )
                run_number = cursor.fetchone()["next_run"]

            # Create test run with metadata
            test_run_id = self.db_manager.create_test_run(
                test_case_id=test_case_id,
                run_number=run_number,
                status="completed" if result.success else "failed",
            )

            # Update test run with full metadata
            self.db_manager.update_test_run(
                test_run_id,
                status="completed" if result.success else "failed",
                completed_at=datetime.now(),
                total_iterations=result.iterations,
                total_tool_calls=result.tool_calls_made,
                final_status="success" if result.success else "failed",
                error_message=result.error,
            )

            # Store strategy in test_runs metadata
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE test_runs SET metadata = ? WHERE id = ?",
                    (
                        json.dumps({"strategy": result.strategy, **result.metrics}),
                        test_run_id,
                    ),
                )

            # Store performance metrics
            self.db_manager.add_performance_metric(
                session_id=test_run_id,
                metric_name="response_time",
                metric_value=result.response_time_ms,
                metric_unit="ms",
            )
            self.db_manager.add_performance_metric(
                session_id=test_run_id,
                metric_name="reasoning_steps",
                metric_value=result.reasoning_steps,
                metric_unit="steps",
            )
            self.db_manager.add_performance_metric(
                session_id=test_run_id,
                metric_name="iterations",
                metric_value=result.iterations,
                metric_unit="count",
            )
            self.db_manager.add_performance_metric(
                session_id=test_run_id,
                metric_name="tool_calls",
                metric_value=result.tool_calls_made,
                metric_unit="count",
            )

            # Store parallel usage as metric
            self.db_manager.add_performance_metric(
                session_id=test_run_id,
                metric_name="parallel_used",
                metric_value=1 if result.parallel_calls_used else 0,
                metric_unit="boolean",
            )

            logger.info(
                f"   📊 Stored in database (test_run_id: {test_run_id}, run #{run_number})"
            )

        except Exception as e:
            logger.warning(f"   ⚠️ Failed to store in database: {e}")

    def _generate_report(self):
        """Generate comprehensive report."""
        logger.info(f"\n\n{'=' * 80}")
        logger.info("📊 FINAL REPORT")
        logger.info(f"{'=' * 80}\n")

        # Group stats by model
        models = {}
        for key, stats in self.stats.items():
            if ":" in key:
                model, strategy = key.split(":", 1)
                if model not in models:
                    models[model] = {}
                models[model][strategy] = stats
            else:
                # Single model test (legacy format)
                if "single" not in models:
                    models["single"] = {}
                models["single"][key] = stats

        # Generate report for each model
        for model_name, model_strategies in models.items():
            if len(models) > 1:
                logger.info(f"\n\n{'🤖' * 40}")
                logger.info(f"MODEL: {model_name}")
                logger.info(f"{'🤖' * 40}\n")

            # Strategy comparison table
            logger.info("STRATEGY COMPARISON:")
            logger.info(
                f"{'Strategy':<15} | {'Pass':<5} | {'Fail':<5} | {'Success %':<10} | {'Avg Iter':<10} | {'Avg Tools':<10} | {'Avg Time (ms)':<15}"
            )
            logger.info("-" * 100)

            for strategy_name, stats in model_strategies.items():
                success_rate = (
                    (stats.passed / stats.total_tests * 100)
                    if stats.total_tests > 0
                    else 0
                )
                logger.info(
                    f"{strategy_name:<15} | {stats.passed:<5} | {stats.failed:<5} | {success_rate:<9.1f}% | "
                    f"{stats.avg_iterations:<10.1f} | {stats.avg_tool_calls:<10.1f} | {stats.avg_response_time_ms:<15.2f}"
                )

            # Detailed results by complexity
            logger.info(f"\n\nRESULTS BY COMPLEXITY:")
            logger.info("-" * 80)

            for complexity in ["low", "medium", "high"]:
                logger.info(f"\n{complexity.upper()} Complexity:")
                for strategy_name, stats in model_strategies.items():
                    complexity_results = [
                        r
                        for r in stats.results
                        if r.metrics.get("complexity") == complexity
                    ]
                    if complexity_results:
                        passed = sum(1 for r in complexity_results if r.success)
                        total = len(complexity_results)
                        rate = (passed / total * 100) if total > 0 else 0
                        logger.info(
                            f"   {strategy_name}: {passed}/{total} ({rate:.1f}%)"
                        )

            # Parallel execution capability
            logger.info(f"\n\nPARALLEL EXECUTION CAPABILITY:")
            logger.info("-" * 80)
            for strategy_name, stats in model_strategies.items():
                parallel_rate = (
                    (stats.parallel_usage_count / stats.total_tests * 100)
                    if stats.total_tests > 0
                    else 0
                )
                logger.info(
                    f"   {strategy_name}: {stats.parallel_usage_count}/{stats.total_tests} tests used parallel calls ({parallel_rate:.1f}%)"
                )

            # Reasoning steps
            logger.info(f"\n\nREASONING CAPABILITY:")
            logger.info("-" * 80)
            for strategy_name, stats in model_strategies.items():
                avg_reasoning = (
                    stats.total_reasoning_steps / stats.total_tests
                    if stats.total_tests > 0
                    else 0
                )
                logger.info(
                    f"   {strategy_name}: {avg_reasoning:.1f} avg reasoning steps"
                )

            # Model summary
            logger.info(f"\n\n{'=' * 80}")
            logger.info(f"SUMMARY FOR {model_name}")
            logger.info(f"{'=' * 80}")

            best_overall = max(
                model_strategies.items(),
                key=lambda x: x[1].passed / x[1].total_tests
                if x[1].total_tests > 0
                else 0,
            )
            logger.info(
                f"   Best overall strategy: {best_overall[0]} ({best_overall[1].passed}/{best_overall[1].total_tests} passed)"
            )

            fastest = min(
                model_strategies.items(), key=lambda x: x[1].avg_response_time_ms
            )
            logger.info(
                f"   Fastest strategy: {fastest[0]} ({fastest[1].avg_response_time_ms:.2f}ms avg)"
            )

            most_reasoning = max(
                model_strategies.items(),
                key=lambda x: x[1].total_reasoning_steps / x[1].total_tests
                if x[1].total_tests > 0
                else 0,
            )
            logger.info(
                f"   Most reasoning: {most_reasoning[0]} ({most_reasoning[1].total_reasoning_steps / most_reasoning[1].total_tests if most_reasoning[1].total_tests > 0 else 0:.1f} avg steps)"
            )

        # Cross-model comparison if testing multiple models
        if len(models) > 1:
            logger.info(f"\n\n{'=' * 80}")
            logger.info("🏆 CROSS-MODEL COMPARISON")
            logger.info(f"{'=' * 80}\n")

            # Find best model by overall success rate
            model_scores = []
            for model_name, model_strategies in models.items():
                total_passed = sum(s.passed for s in model_strategies.values())
                total_tests = sum(s.total_tests for s in model_strategies.values())
                score = (total_passed / total_tests * 100) if total_tests > 0 else 0
                model_scores.append((model_name, score, total_passed, total_tests))

            model_scores.sort(key=lambda x: x[1], reverse=True)

            logger.info("OVERALL SUCCESS RATE:")
            logger.info(
                f"{'Model':<20} | {'Passed':<8} | {'Total':<8} | {'Success %':<10}"
            )
            logger.info("-" * 60)
            for model_name, score, passed, total in model_scores:
                logger.info(
                    f"{model_name:<20} | {passed:<8} | {total:<8} | {score:<9.1f}%"
                )

            logger.info(
                f"\n🥇 Best model: {model_scores[0][0]} ({model_scores[0][1]:.1f}% success rate)"
            )

        # Save JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "llm_url": self.llm_base_url,
            "models_tested": list(models.keys())
            if len(models) > 1
            else [self.llm_model],
            "results": {
                model_name: {
                    "strategies": {
                        name: {
                            "total_tests": stats.total_tests,
                            "passed": stats.passed,
                            "failed": stats.failed,
                            "success_rate": stats.passed / stats.total_tests
                            if stats.total_tests > 0
                            else 0,
                            "avg_iterations": stats.avg_iterations,
                            "avg_tool_calls": stats.avg_tool_calls,
                            "avg_response_time_ms": stats.avg_response_time_ms,
                            "parallel_usage_count": stats.parallel_usage_count,
                            "avg_reasoning_steps": stats.total_reasoning_steps
                            / stats.total_tests
                            if stats.total_tests > 0
                            else 0,
                        }
                        for name, stats in model_strategies.items()
                    }
                }
                for model_name, model_strategies in models.items()
            },
            "all_results": [
                {
                    "test_name": r.test_name,
                    "strategy": r.strategy,
                    "success": r.success,
                    "iterations": r.iterations,
                    "tool_calls": r.tool_calls_made,
                    "parallel_used": r.parallel_calls_used,
                    "reasoning_steps": r.reasoning_steps,
                    "response_time_ms": r.response_time_ms,
                    "error": r.error,
                }
                for r in self.all_results
            ],
        }

        report_file = Path("llm_capability_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n\n💾 Report saved to: {report_file.absolute()}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LLM capabilities with different orchestration strategies"
    )
    parser.add_argument(
        "--llm-url", default="http://localhost:8087/v1", help="LLM API base URL"
    )
    parser.add_argument("--llm-model", default="glm-4.7", help="LLM model name")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["basic", "react", "plan_execute", "tot"],
        choices=["basic", "react", "plan_execute", "tot"],
        help="Strategies to test",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all available models from the API (overrides --llm-model)",
    )

    args = parser.parse_args()

    tester = LLMAbilityTester(llm_base_url=args.llm_url, llm_model=args.llm_model)
    tester.run_all_tests(strategies=args.strategies, test_all_models=args.all_models)


if __name__ == "__main__":
    main()
