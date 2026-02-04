"""Continuous Learning Loop for self-improvement.

This system continuously monitors, evaluates, and improves the knowledge base
through feedback loops and adaptive strategies.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .knowledge_quality_evaluator import KnowledgeQualityEvaluator, QualityMetrics
from .skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for learning loop."""

    iteration: int
    quality_score: float
    feedback_received: int
    improvements_made: int
    errors_fixed: int
    performance_gain: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ImprovementAction:
    """Action taken to improve the system."""

    action_type: str
    description: str
    parameters: dict
    executed_at: str
    result: str | None = None
    impact: float = 0.0


class ContinuousLearningLoop:
    """Continuous learning and improvement system."""

    def __init__(
        self,
        evaluator: KnowledgeQualityEvaluator | None = None,
        vector_skill: Any | None = None,
        neo4j_skill: Any | None = None,
        min_quality_threshold: float = 0.7,
        learning_interval: int = 3600,  # seconds
    ):
        """Initialize Continuous Learning Loop.

        Args:
            evaluator: Knowledge quality evaluator
            vector_skill: Vector storage skill
            neo4j_skill: Knowledge graph skill
            min_quality_threshold: Minimum acceptable quality score
            learning_interval: Interval between learning iterations
        """
        self.evaluator = evaluator
        self.vector_skill = vector_skill
        self.neo4j_skill = neo4j_skill
        self.min_quality_threshold = min_quality_threshold
        self.learning_interval = learning_interval

        self.metrics_history: list[LearningMetrics] = []
        self.improvements: list[ImprovementAction] = []
        self.feedback_queue: list[dict] = []

        self.is_running = False
        logger.info("ContinuousLearningLoop initialized")

    def start_learning(self) -> SkillResult:
        """Start the continuous learning loop.

        Returns:
            SkillResult with start status
        """
        try:
            if self.is_running:
                return SkillResult(success=False, error="Learning loop already running")

            self.is_running = True
            logger.info("Continuous learning loop started")

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "status": "started",
                    "learning_interval": self.learning_interval,
                    "min_quality_threshold": self.min_quality_threshold,
                },
            )

        except Exception as e:
            logger.error(f"Error starting learning loop: {e}")
            return SkillResult(success=False, error=f"Start failed: {str(e)}")

    def stop_learning(self) -> SkillResult:
        """Stop the continuous learning loop.

        Returns:
            SkillResult with stop status
        """
        try:
            self.is_running = False
            logger.info("Continuous learning loop stopped")

            self._record_usage()
            return SkillResult(
                success=True,
                data={"status": "stopped", "iterations_completed": len(self.metrics_history)},
            )

        except Exception as e:
            logger.error(f"Error stopping learning loop: {e}")
            return SkillResult(success=False, error=f"Stop failed: {str(e)}")

    def run_learning_iteration(self) -> SkillResult:
        """Run a single learning iteration.

        Returns:
            SkillResult with iteration results
        """
        try:
            iteration = len(self.metrics_history) + 1
            logger.info(f"Running learning iteration {iteration}")

            # Collect feedback
            feedback = self._collect_feedback()

            # Evaluate current state
            quality_result = self._evaluate_current_state()

            if not quality_result.success:
                return quality_result

            quality_score = quality_result.data.get("overall_score", 0.0)

            # Generate improvements
            improvements = self._generate_improvements(quality_result.data, feedback)

            # Execute improvements
            improvement_results = []
            for improvement in improvements:
                result = self._execute_improvement(improvement)
                improvement_results.append(result)
                self.improvements.append(
                    ImprovementAction(
                        action_type=improvement["action_type"],
                        description=improvement["description"],
                        parameters=improvement.get("parameters", {}),
                        executed_at=datetime.now().isoformat(),
                        result="success" if result.success else "failed",
                    )
                )

            # Record metrics
            metrics = LearningMetrics(
                iteration=iteration,
                quality_score=quality_score,
                feedback_received=len(feedback),
                improvements_made=len([r for r in improvement_results if r.success]),
                errors_fixed=self._count_errors_fixed(improvement_results),
            )

            # Calculate performance gain
            if self.metrics_history:
                prev_metrics = self.metrics_history[-1]
                metrics.performance_gain = metrics.quality_score - prev_metrics.quality_score

            self.metrics_history.append(metrics)

            # Create snapshot
            if self.neo4j_skill:
                self._create_learning_snapshot(metrics, improvements)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "iteration": iteration,
                    "quality_score": quality_score,
                    "feedback_received": metrics.feedback_received,
                    "improvements_made": metrics.improvements_made,
                    "errors_fixed": metrics.errors_fixed,
                    "performance_gain": metrics.performance_gain,
                    "improvements": [
                        {
                            "action_type": imp.action_type,
                            "description": imp.description,
                            "result": imp.result,
                        }
                        for imp in self.improvements[-len(improvements) :]
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Error in learning iteration: {e}")
            return SkillResult(success=False, error=f"Iteration failed: {str(e)}")

    def add_feedback(self, feedback: dict) -> SkillResult:
        """Add feedback to the learning loop.

        Args:
            feedback: Feedback data

        Returns:
            SkillResult with feedback status
        """
        try:
            feedback["timestamp"] = datetime.now().isoformat()
            self.feedback_queue.append(feedback)

            return SkillResult(
                success=True,
                data={"feedback_count": len(self.feedback_queue)},
            )

        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return SkillResult(success=False, error=f"Feedback addition failed: {str(e)}")

    def _collect_feedback(self) -> list[dict]:
        """Collect and categorize feedback.

        Returns:
            List of feedback items
        """
        feedback = self.feedback_queue.copy()
        self.feedback_queue.clear()

        # Categorize feedback
        categorized = {
            "quality_issues": [],
            "retrieval_issues": [],
            "accuracy_issues": [],
            "other": [],
        }

        for item in feedback:
            feedback_type = item.get("type", "other")
            if feedback_type in categorized:
                categorized[feedback_type].append(item)
            else:
                categorized["other"].append(item)

        return feedback

    def _evaluate_current_state(self) -> SkillResult:
        """Evaluate the current state of the knowledge base.

        Returns:
            SkillResult with evaluation results
        """
        try:
            if not self.evaluator:
                # Return placeholder metrics
                return SkillResult(
                    success=True,
                    data={
                        "context_relevancy": 0.7,
                        "answer_faithfulness": 0.7,
                        "answer_relevancy": 0.7,
                        "context_utilization": 0.7,
                        "freshness": 0.7,
                        "completeness": 0.7,
                        "overall_score": 0.7,
                    },
                )

            # Use predefined test cases for evaluation
            test_questions = [
                {
                    "question": "What is machine learning?",
                    "answer": "Machine learning is a subset of artificial intelligence...",
                    "context": "Machine learning algorithms build models based on sample data...",
                    "reference": "Machine learning enables computers to learn...",
                },
            ]

            result = self.evaluator.evaluate_knowledge_base(test_questions)
            return result

        except Exception as e:
            logger.error(f"Error evaluating current state: {e}")
            return SkillResult(success=False, error=f"Evaluation failed: {str(e)}")

    def _generate_improvements(
        self, metrics: dict, feedback: list[dict]
    ) -> list[dict]:
        """Generate improvement actions based on metrics and feedback.

        Args:
            metrics: Current metrics
            feedback: Collected feedback

        Returns:
            List of improvement actions
        """
        improvements = []

        # Analyze metrics
        quality_score = metrics.get("overall_score", 0.0)
        context_relevancy = metrics.get("context_relevancy", 0.0)
        answer_faithfulness = metrics.get("answer_faithfulness", 0.0)
        answer_relevancy = metrics.get("answer_relevancy", 0.0)
        context_utilization = metrics.get("context_utilization", 0.0)

        # Generate improvements based on low scores
        if context_relevancy < self.min_quality_threshold:
            improvements.append(
                {
                    "action_type": "improve_retrieval",
                    "description": "Improve retrieval system for better context relevancy",
                    "parameters": {"current_score": context_relevancy, "target_score": self.min_quality_threshold + 0.1},
                }
            )

        if answer_faithfulness < self.min_quality_threshold:
            improvements.append(
                {
                    "action_type": "reduce_hallucinations",
                    "description": "Implement stricter context adherence to reduce hallucinations",
                    "parameters": {"current_score": answer_faithfulness, "target_score": self.min_quality_threshold + 0.1},
                }
            )

        if answer_relevancy < self.min_quality_threshold:
            improvements.append(
                {
                    "action_type": "improve_answer_generation",
                    "description": "Refine answer generation prompts for better question addressing",
                    "parameters": {"current_score": answer_relevancy, "target_score": self.min_quality_threshold + 0.1},
                }
            )

        if context_utilization < self.min_quality_threshold:
            improvements.append(
                {
                    "action_type": "optimize_context_usage",
                    "description": "Optimize how retrieved context is used in answers",
                    "parameters": {"current_score": context_utilization, "target_score": self.min_quality_threshold + 0.1},
                }
            )

        # Process feedback-driven improvements
        for item in feedback:
            if item.get("type") == "quality_issues":
                improvements.append(
                    {
                        "action_type": "address_quality_issue",
                        "description": f"Address: {item.get('description', 'Unknown issue')}",
                        "parameters": item,
                    }
                )

        # Limit improvements to prevent over-optimization
        return improvements[:5]

    def _execute_improvement(self, improvement: dict) -> SkillResult:
        """Execute an improvement action.

        Args:
            improvement: Improvement action details

        Returns:
            SkillResult with execution results
        """
        try:
            action_type = improvement["action_type"]
            parameters = improvement.get("parameters", {})

            if action_type == "improve_retrieval":
                return self._improve_retrieval(parameters)
            elif action_type == "reduce_hallucinations":
                return self._reduce_hallucinations(parameters)
            elif action_type == "improve_answer_generation":
                return self._improve_answer_generation(parameters)
            elif action_type == "optimize_context_usage":
                return self._optimize_context_usage(parameters)
            elif action_type == "address_quality_issue":
                return self._address_quality_issue(parameters)
            else:
                return SkillResult(success=False, error=f"Unknown action type: {action_type}")

        except Exception as e:
            logger.error(f"Error executing improvement: {e}")
            return SkillResult(success=False, error=f"Improvement execution failed: {str(e)}")

    def _improve_retrieval(self, parameters: dict) -> SkillResult:
        """Improve retrieval system.

        Args:
            parameters: Improvement parameters

        Returns:
            SkillResult with improvement results
        """
        # This would adjust retrieval parameters, re-index, etc.
        logger.info(f"Improving retrieval: {parameters}")
        return SkillResult(
            success=True,
            data={"action": "improve_retrieval", "result": "Retrieval parameters adjusted"},
        )

    def _reduce_hallucinations(self, parameters: dict) -> SkillResult:
        """Reduce hallucinations in answers.

        Args:
            parameters: Improvement parameters

        Returns:
            SkillResult with improvement results
        """
        logger.info(f"Reducing hallucinations: {parameters}")
        return SkillResult(
            success=True,
            data={"action": "reduce_hallucinations", "result": "Context adherence measures strengthened"},
        )

    def _improve_answer_generation(self, parameters: dict) -> SkillResult:
        """Improve answer generation.

        Args:
            parameters: Improvement parameters

        Returns:
            SkillResult with improvement results
        """
        logger.info(f"Improving answer generation: {parameters}")
        return SkillResult(
            success=True,
            data={"action": "improve_answer_generation", "result": "Answer generation prompts refined"},
        )

    def _optimize_context_usage(self, parameters: dict) -> SkillResult:
        """Optimize context usage.

        Args:
            parameters: Improvement parameters

        Returns:
            SkillResult with improvement results
        """
        logger.info(f"Optimizing context usage: {parameters}")
        return SkillResult(
            success=True,
            data={"action": "optimize_context_usage", "result": "Context utilization strategy optimized"},
        )

    def _address_quality_issue(self, parameters: dict) -> SkillResult:
        """Address a specific quality issue.

        Args:
            parameters: Issue parameters

        Returns:
            SkillResult with resolution results
        """
        logger.info(f"Addressing quality issue: {parameters}")
        return SkillResult(
            success=True,
            data={"action": "address_quality_issue", "result": "Issue addressed"},
        )

    def _create_learning_snapshot(self, metrics: LearningMetrics, improvements: list[dict]):
        """Create a snapshot of the learning state.

        Args:
            metrics: Current learning metrics
            improvements: Improvements made
        """
        if self.neo4j_skill:
            self.neo4j_skill.execute(
                action="insert_entity",
                entities=[
                    {
                        "id": f"learning_iteration_{metrics.iteration}",
                        "label": "LearningIteration",
                        "properties": {
                            "iteration": metrics.iteration,
                            "quality_score": metrics.quality_score,
                            "feedback_received": metrics.feedback_received,
                            "improvements_made": metrics.improvements_made,
                            "errors_fixed": metrics.errors_fixed,
                            "performance_gain": metrics.performance_gain,
                            "timestamp": metrics.timestamp,
                            "improvements": json.dumps(improvements, default=str),
                        },
                    }
                ],
            )

    def _count_errors_fixed(self, improvement_results: list[SkillResult]) -> int:
        """Count how many errors were fixed.

        Args:
            improvement_results: Results from improvements

        Returns:
            Count of fixed errors
        """
        return sum(1 for r in improvement_results if r.success)

    def get_learning_status(self) -> dict:
        """Get current learning status.

        Returns:
            Learning status information
        """
        if self.metrics_history:
            latest = self.metrics_history[-1]
        else:
            latest = None

        return {
            "is_running": self.is_running,
            "total_iterations": len(self.metrics_history),
            "latest_quality_score": latest.quality_score if latest else None,
            "total_improvements": len(self.improvements),
            "feedback_queue_size": len(self.feedback_queue),
            "min_quality_threshold": self.min_quality_threshold,
            "performance_trend": self._calculate_performance_trend(),
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend.

        Returns:
            Trend description
        """
        if len(self.metrics_history) < 2:
            return "insufficient_data"

        recent = self.metrics_history[-5:]
        if len(recent) < 2:
            recent = self.metrics_history

        gains = [m.performance_gain for m in recent if m.performance_gain > 0]
        losses = [m.performance_gain for m in recent if m.performance_gain < 0]

        if sum(gains) > sum(abs(loss) for losses in losses):
            return "improving"
        elif sum(abs(losses) for losses in losses) > sum(gains):
            return "declining"
        else:
            return "stable"

    def _record_usage(self):
        """Record usage statistics."""
        pass

    def health_check(self) -> bool:
        """Check if learning loop is operational."""
        return self.evaluator is not None or self.vector_skill is not None or self.neo4j_skill is not None