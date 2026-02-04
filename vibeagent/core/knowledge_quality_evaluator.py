"""Knowledge Quality Evaluator for RAG systems.

This evaluator measures the quality of knowledge bases and RAG systems
using metrics like context relevancy, answer faithfulness, and freshness.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for knowledge base."""

    context_relevancy: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_utilization: float = 0.0
    freshness: float = 0.0
    completeness: float = 0.0
    overall_score: float = 0.0
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationResult:
    """Result of knowledge base evaluation."""

    metrics: QualityMetrics
    test_cases: int
    passed: int
    failed: int
    details: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class KnowledgeQualityEvaluator:
    """Evaluator for knowledge base quality and RAG metrics."""

    def __init__(self, llm_skill: Any | None = None, vector_skill: Any | None = None):
        """Initialize Knowledge Quality Evaluator.

        Args:
            llm_skill: LLM skill for evaluation
            vector_skill: Vector skill for retrieval testing
        """
        self.llm_skill = llm_skill
        self.vector_skill = vector_skill
        logger.info("KnowledgeQualityEvaluator initialized")

    def evaluate_rag(
        self,
        question: str,
        answer: str,
        context: str,
        reference_answer: str | None = None,
    ) -> SkillResult:
        """Evaluate RAG response quality.

        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            reference_answer: Reference answer for comparison

        Returns:
            SkillResult with evaluation metrics
        """
        try:
            metrics = {}

            # Evaluate context relevancy
            metrics["context_relevancy"] = self._evaluate_context_relevancy(question, context)

            # Evaluate answer faithfulness
            metrics["answer_faithfulness"] = self._evaluate_answer_faithfulness(answer, context)

            # Evaluate answer relevancy
            metrics["answer_relevancy"] = self._evaluate_answer_relevancy(question, answer)

            # Evaluate context utilization
            metrics["context_utilization"] = self._evaluate_context_utilization(answer, context)

            # Calculate overall score
            metrics["overall_score"] = self._calculate_overall_score(metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "question": question,
                    "answer": answer,
                    "metrics": metrics,
                    "recommendations": recommendations,
                },
            )

        except Exception as e:
            logger.error(f"Error evaluating RAG: {e}")
            return SkillResult(success=False, error=f"Evaluation failed: {str(e)}")

    def evaluate_knowledge_base(
        self, sample_questions: list[dict], limit: int = 50
    ) -> SkillResult:
        """Evaluate knowledge base quality.

        Args:
            sample_questions: List of test questions with expected answers
            limit: Number of questions to evaluate

        Returns:
            SkillResult with evaluation results
        """
        try:
            if not sample_questions:
                return SkillResult(success=False, error="No sample questions provided")

            test_cases = sample_questions[:limit]
            results = []
            metrics_list = []

            for i, test_case in enumerate(test_cases):
                logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")

                # Generate answer (would need a generation pipeline)
                answer = test_case.get("answer", "")
                context = test_case.get("context", "")
                reference = test_case.get("reference", "")

                # Evaluate
                evaluation = self.evaluate_rag(
                    question=test_case.get("question", ""),
                    answer=answer,
                    context=context,
                    reference_answer=reference,
                )

                if evaluation.success:
                    results.append(evaluation.data)
                    metrics_list.append(evaluation.data["metrics"])

            # Calculate aggregate metrics
            if metrics_list:
                aggregate_metrics = self._calculate_aggregate_metrics(metrics_list)
            else:
                aggregate_metrics = {}

            # Generate recommendations
            recommendations = self._generate_kb_recommendations(aggregate_metrics)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "total_test_cases": len(test_cases),
                    "evaluated": len(results),
                    "aggregate_metrics": aggregate_metrics,
                    "recommendations": recommendations,
                    "details": results,
                },
            )

        except Exception as e:
            logger.error(f"Error evaluating knowledge base: {e}")
            return SkillResult(success=False, error=f"Evaluation failed: {str(e)}")

    def _evaluate_context_relevancy(self, question: str, context: str) -> float:
        """Evaluate how relevant the context is to the question.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Relevancy score (0-1)
        """
        try:
            if not self.llm_skill:
                return 0.5  # Default score

            prompt = f"""Rate the relevancy of the context to the question on a scale of 0 to 1.

Question: {question}

Context: {context}

Respond with just the numeric score (0.0 to 1.0)."""

            result = self.llm_skill.execute(
                prompt=prompt,
                system_prompt="You are an expert evaluator. Respond with only a number.",
                max_tokens=10,
            )

            if result.success:
                try:
                    score = float(result.data.get("content", "0.5").strip())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    return 0.5

            return 0.5

        except Exception as e:
            logger.error(f"Error evaluating context relevancy: {e}")
            return 0.5

    def _evaluate_answer_faithfulness(self, answer: str, context: str) -> float:
        """Evaluate if the answer is faithful to the context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Faithfulness score (0-1)
        """
        try:
            if not self.llm_skill:
                return 0.5

            prompt = f"""Rate how faithful the answer is to the provided context on a scale of 0 to 1.
An answer is faithful if it only uses information present in the context.

Answer: {answer}

Context: {context}

Respond with just the numeric score (0.0 to 1.0)."""

            result = self.llm_skill.execute(
                prompt=prompt,
                system_prompt="You are an expert evaluator. Respond with only a number.",
                max_tokens=10,
            )

            if result.success:
                try:
                    score = float(result.data.get("content", "0.5").strip())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    return 0.5

            return 0.5

        except Exception as e:
            logger.error(f"Error evaluating answer faithfulness: {e}")
            return 0.5

    def _evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the question.

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevancy score (0-1)
        """
        try:
            if not self.llm_skill:
                return 0.5

            prompt = f"""Rate how relevant the answer is to the question on a scale of 0 to 1.

Question: {question}

Answer: {answer}

Respond with just the numeric score (0.0 to 1.0)."""

            result = self.llm_skill.execute(
                prompt=prompt,
                system_prompt="You are an expert evaluator. Respond with only a number.",
                max_tokens=10,
            )

            if result.success:
                try:
                    score = float(result.data.get("content", "0.5").strip())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    return 0.5

            return 0.5

        except Exception as e:
            logger.error(f"Error evaluating answer relevancy: {e}")
            return 0.5

    def _evaluate_context_utilization(self, answer: str, context: str) -> float:
        """Evaluate how much of the context is actually used in the answer.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Utilization score (0-1)
        """
        try:
            # Simple approach: count context words in answer
            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())

            if not context_words:
                return 0.0

            overlap = len(context_words & answer_words)
            utilization = min(1.0, overlap / len(context_words))

            return utilization

        except Exception as e:
            logger.error(f"Error evaluating context utilization: {e}")
            return 0.0

    def _calculate_overall_score(self, metrics: dict[str, float]) -> float:
        """Calculate overall quality score.

        Args:
            metrics: Individual metrics

        Returns:
            Overall score (0-1)
        """
        weights = {
            "context_relevancy": 0.3,
            "answer_faithfulness": 0.3,
            "answer_relevancy": 0.2,
            "context_utilization": 0.2,
        }

        weighted_sum = sum(metrics.get(key, 0.0) * weight for key, weight in weights.items())
        return weighted_sum

    def _calculate_aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Calculate aggregate metrics from multiple evaluations.

        Args:
            metrics_list: List of metric dictionaries

        Returns:
            Aggregate metrics
        """
        if not metrics_list:
            return {}

        aggregate = {}
        for key in metrics_list[0].keys():
            values = [m.get(key, 0.0) for m in metrics_list]
            aggregate[key] = sum(values) / len(values)

        return aggregate

    def _generate_recommendations(self, metrics: dict[str, float]) -> list[str]:
        """Generate improvement recommendations.

        Args:
            metrics: Quality metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        if metrics.get("context_relevancy", 0.0) < 0.7:
            recommendations.append(
                "Improve retrieval system to fetch more relevant context (consider hybrid search, better embeddings)"
            )

        if metrics.get("answer_faithfulness", 0.0) < 0.7:
            recommendations.append(
                "Reduce hallucinations by enforcing strict context adherence in answer generation"
            )

        if metrics.get("answer_relevancy", 0.0) < 0.7:
            recommendations.append(
                "Improve answer generation to better address user questions (refine prompts, use chain-of-thought)"
            )

        if metrics.get("context_utilization", 0.0) < 0.5:
            recommendations.append(
                "Optimize context usage - provide more context to the model or adjust prompting strategy"
            )

        if metrics.get("overall_score", 0.0) < 0.7:
            recommendations.append(
                "Overall quality below threshold - consider comprehensive evaluation and system optimization"
            )

        if not recommendations:
            recommendations.append("System performing well - continue monitoring and collecting feedback")

        return recommendations

    def _generate_kb_recommendations(self, aggregate_metrics: dict[str, float]) -> list[str]:
        """Generate knowledge base specific recommendations.

        Args:
            aggregate_metrics: Aggregate metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        overall_score = aggregate_metrics.get("overall_score", 0.0)

        if overall_score < 0.5:
            recommendations.append("Critical: Major knowledge base improvements needed")
            recommendations.append("Consider adding more diverse and comprehensive documentation")
            recommendations.append("Review and improve indexing and retrieval strategies")
        elif overall_score < 0.7:
            recommendations.append("Knowledge base quality needs improvement")
            recommendations.append("Expand coverage of topics and improve document quality")
            recommendations.append("Consider implementing RAG evaluation as continuous process")
        elif overall_score < 0.9:
            recommendations.append("Knowledge base quality is good but can be improved")
            recommendations.append("Focus on edge cases and improving context retrieval")
            recommendations.append("Implement continuous quality monitoring")
        else:
            recommendations.append("Excellent knowledge base quality")
            recommendations.append("Maintain current practices and monitor for regression")

        return recommendations

    def get_evaluation_report(self) -> dict:
        """Get evaluation system status.

        Returns:
            System status information
        """
        return {
            "llm_skill_available": self.llm_skill is not None,
            "vector_skill_available": self.vector_skill is not None,
            "metrics": [
                "context_relevancy",
                "answer_faithfulness",
                "answer_relevancy",
                "context_utilization",
                "freshness",
                "completeness",
                "overall_score",
            ],
        }

    def health_check(self) -> bool:
        """Check if evaluator is operational."""
        return self.llm_skill is not None