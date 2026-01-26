import json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Analytics engine for generating insights from the VibeAgent database."""

    def __init__(self, db_manager):
        """Initialize analytics engine.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager

    def get_success_rate_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get success rate trend over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with daily success rates
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        DATE(s.created_at) as date,
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions s
                    WHERE s.created_at >= datetime('now', '-' || ? || ' days')
                    GROUP BY DATE(s.created_at)
                    ORDER BY date DESC
                    """,
                    (days,),
                )
                results = [dict(row) for row in cursor.fetchall()]

            return {
                "period_days": days,
                "total_days": len(results),
                "data": results,
                "avg_success_rate": round(
                    mean([r["success_rate"] for r in results if r["success_rate"]]), 2
                )
                if results
                else 0,
            }
        except Exception as e:
            logger.error(f"Error getting success rate trend: {e}")
            return {"error": str(e), "data": []}

    def get_tool_performance(self, tool_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific tool.

        Args:
            tool_name: Name of the tool to analyze

        Returns:
            Dictionary with tool performance metrics
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        tool_name,
                        COUNT(*) as total_calls,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_calls,
                        ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                        AVG(execution_time_ms) as avg_execution_time_ms,
                        MIN(execution_time_ms) as min_execution_time_ms,
                        MAX(execution_time_ms) as max_execution_time_ms,
                        SUM(CASE WHEN success = 0 THEN 1 END) as failed_calls,
                        AVG(retry_count) as avg_retry_count
                    FROM tool_calls
                    WHERE tool_name = ?
                    GROUP BY tool_name
                    """,
                    (tool_name,),
                )
                result = cursor.fetchone()

                if not result:
                    return {
                        "error": f"Tool '{tool_name}' not found",
                        "tool_name": tool_name,
                    }

                data = dict(result)

                cursor.execute(
                    """
                    SELECT error_type, COUNT(*) as count
                    FROM tool_calls
                    WHERE tool_name = ? AND success = 0 AND error_type IS NOT NULL
                    GROUP BY error_type
                    ORDER BY count DESC
                    LIMIT 10
                    """,
                    (tool_name,),
                )
                data["error_breakdown"] = [dict(row) for row in cursor.fetchall()]

                return data
        except Exception as e:
            logger.error(f"Error getting tool performance: {e}")
            return {"error": str(e), "tool_name": tool_name}

    def get_model_comparison(self) -> List[Dict[str, Any]]:
        """Compare performance across different models.

        Returns:
            List of model performance metrics
        """
        try:
            return self.db.get_model_comparison()
        except Exception as e:
            logger.error(f"Error getting model comparison: {e}")
            return []

    def get_execution_time_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of execution times.

        Returns:
            Dictionary with execution time statistics
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT execution_time_ms
                    FROM tool_calls
                    WHERE execution_time_ms > 0
                    """
                )
                times = [row[0] for row in cursor.fetchall()]

            if not times:
                return {"error": "No execution time data available"}

            times_sorted = sorted(times)
            n = len(times)

            return {
                "count": n,
                "min_ms": min(times),
                "max_ms": max(times),
                "mean_ms": round(mean(times), 2),
                "median_ms": round(median(times), 2),
                "std_dev_ms": round(stdev(times), 2) if n > 1 else 0,
                "p25_ms": times_sorted[int(n * 0.25)],
                "p50_ms": times_sorted[int(n * 0.5)],
                "p75_ms": times_sorted[int(n * 0.75)],
                "p90_ms": times_sorted[int(n * 0.9)],
                "p95_ms": times_sorted[int(n * 0.95)],
                "p99_ms": times_sorted[int(n * 0.99)],
            }
        except Exception as e:
            logger.error(f"Error getting execution time distribution: {e}")
            return {"error": str(e)}

    def get_iteration_statistics(self) -> Dict[str, Any]:
        """Analyze iteration count statistics.

        Returns:
            Dictionary with iteration statistics
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT total_iterations
                    FROM sessions
                    WHERE total_iterations > 0
                    """
                )
                iterations = [row[0] for row in cursor.fetchall()]

            if not iterations:
                return {"error": "No iteration data available"}

            iterations_sorted = sorted(iterations)
            n = len(iterations)

            return {
                "count": n,
                "min_iterations": min(iterations),
                "max_iterations": max(iterations),
                "mean_iterations": round(mean(iterations), 2),
                "median_iterations": round(median(iterations), 2),
                "std_dev_iterations": round(stdev(iterations), 2) if n > 1 else 0,
                "p25_iterations": iterations_sorted[int(n * 0.25)],
                "p75_iterations": iterations_sorted[int(n * 0.75)],
                "p90_iterations": iterations_sorted[int(n * 0.9)],
            }
        except Exception as e:
            logger.error(f"Error getting iteration statistics: {e}")
            return {"error": str(e)}

    def find_failing_tools(self, threshold: float = 50.0) -> List[Dict[str, Any]]:
        """Find tools with high failure rates.

        Args:
            threshold: Failure rate threshold percentage

        Returns:
            List of tools exceeding the failure threshold
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        tool_name,
                        COUNT(*) as total_calls,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_calls,
                        ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                        ROUND(100 - CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as failure_rate
                    FROM tool_calls
                    GROUP BY tool_name
                    HAVING failure_rate >= ?
                    ORDER BY failure_rate DESC
                    """,
                    (threshold,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error finding failing tools: {e}")
            return []

    def find_successful_patterns(self) -> Dict[str, Any]:
        """Identify patterns that lead to successful outcomes.

        Returns:
            Dictionary with successful patterns
        """
        try:
            patterns = {}

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT s.model, 
                           COUNT(*) as total,
                           COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) as successful,
                           ROUND(CAST(COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) AS FLOAT) 
                                 / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions s
                    WHERE s.final_status IS NOT NULL
                    GROUP BY s.model
                    ORDER BY success_rate DESC
                    """
                )
                patterns["by_model"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT s.orchestrator_type,
                           COUNT(*) as total,
                           COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) as successful,
                           ROUND(CAST(COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) AS FLOAT) 
                                 / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions s
                    WHERE s.final_status IS NOT NULL AND s.orchestrator_type IS NOT NULL
                    GROUP BY s.orchestrator_type
                    ORDER BY success_rate DESC
                    """
                )
                patterns["by_orchestrator"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT s.session_type,
                           COUNT(*) as total,
                           COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) as successful,
                           ROUND(CAST(COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) AS FLOAT) 
                                 / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions s
                    WHERE s.final_status IS NOT NULL
                    GROUP BY s.session_type
                    ORDER BY success_rate DESC
                    """
                )
                patterns["by_session_type"] = [dict(row) for row in cursor.fetchall()]

            return patterns
        except Exception as e:
            logger.error(f"Error finding successful patterns: {e}")
            return {"error": str(e)}

    def find_error_patterns(self) -> Dict[str, Any]:
        """Identify common error patterns.

        Returns:
            Dictionary with error patterns
        """
        try:
            patterns = {}

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT error_type, COUNT(*) as count
                    FROM tool_calls
                    WHERE success = 0 AND error_type IS NOT NULL
                    GROUP BY error_type
                    ORDER BY count DESC
                    LIMIT 20
                    """
                )
                patterns["by_error_type"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT tool_name, error_type, COUNT(*) as count
                    FROM tool_calls
                    WHERE success = 0 AND error_type IS NOT NULL
                    GROUP BY tool_name, error_type
                    ORDER BY count DESC
                    LIMIT 20
                    """
                )
                patterns["by_tool_and_error"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT error_type, recovery_strategy, 
                           COUNT(*) as total_attempts,
                           COUNT(CASE WHEN success = 1 THEN 1 END) as successful_recoveries,
                           ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT) 
                                 / NULLIF(COUNT(*), 0) * 100, 2) as recovery_success_rate
                    FROM error_recovery
                    GROUP BY error_type, recovery_strategy
                    ORDER BY total_attempts DESC
                    LIMIT 20
                    """
                )
                patterns["recovery_strategies"] = [
                    dict(row) for row in cursor.fetchall()
                ]

            return patterns
        except Exception as e:
            logger.error(f"Error finding error patterns: {e}")
            return {"error": str(e)}

    def find_optimal_parameters(self) -> Dict[str, Any]:
        """Find best parameter combinations.

        Returns:
            Dictionary with optimal parameter insights
        """
        try:
            insights = {}

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT m.temperature,
                           COUNT(*) as total,
                           AVG(lr.response_time_ms) as avg_response_time_ms,
                           ROUND(CAST(SUM(lr.completion_tokens) AS FLOAT) 
                                 / NULLIF(SUM(lr.prompt_tokens), 0), 2) as avg_token_efficiency
                    FROM messages m
                    JOIN llm_responses lr ON m.id = lr.message_id
                    WHERE m.temperature IS NOT NULL
                    GROUP BY m.temperature
                    ORDER BY avg_token_efficiency DESC
                    """
                )
                insights["by_temperature"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT m.max_tokens,
                           COUNT(*) as total,
                           COUNT(CASE WHEN lr.finish_reason = 'stop' THEN 1 END) as completed,
                           ROUND(CAST(COUNT(CASE WHEN lr.finish_reason = 'stop' THEN 1 END) AS FLOAT) 
                                 / NULLIF(COUNT(*), 0) * 100, 2) as completion_rate
                    FROM messages m
                    JOIN llm_responses lr ON m.id = lr.message_id
                    WHERE m.max_tokens IS NOT NULL
                    GROUP BY m.max_tokens
                    ORDER BY completion_rate DESC
                    """
                )
                insights["by_max_tokens"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT lr.model,
                           AVG(lr.response_time_ms) as avg_response_time_ms,
                           AVG(lr.total_tokens) as avg_total_tokens,
                           ROUND(CAST(SUM(lr.completion_tokens) AS FLOAT) 
                                 / NULLIF(SUM(lr.prompt_tokens), 0), 2) as avg_token_efficiency
                    FROM llm_responses lr
                    GROUP BY lr.model
                    ORDER BY avg_token_efficiency DESC
                    """
                )
                insights["by_model_efficiency"] = [
                    dict(row) for row in cursor.fetchall()
                ]

            return insights
        except Exception as e:
            logger.error(f"Error finding optimal parameters: {e}")
            return {"error": str(e)}

    def detect_performance_degradation(self, days: int = 7) -> Dict[str, Any]:
        """Detect performance degradation over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with degradation insights
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        DATE(s.created_at) as date,
                        AVG(s.total_duration_ms) as avg_duration_ms,
                        AVG(s.total_iterations) as avg_iterations
                    FROM sessions s
                    WHERE s.created_at >= datetime('now', '-' || ? || ' days')
                        AND s.total_duration_ms > 0
                    GROUP BY DATE(s.created_at)
                    ORDER BY date ASC
                    """,
                    (days,),
                )
                daily_data = [dict(row) for row in cursor.fetchall()]

            if len(daily_data) < 2:
                return {
                    "error": "Insufficient data for trend analysis",
                    "data": daily_data,
                }

            durations = [
                d["avg_duration_ms"] for d in daily_data if d["avg_duration_ms"]
            ]
            iterations = [
                d["avg_iterations"] for d in daily_data if d["avg_iterations"]
            ]

            degradation_detected = False
            insights = []

            if len(durations) >= 2:
                duration_trend = (durations[-1] - durations[0]) / durations[0] * 100
                if duration_trend > 20:
                    degradation_detected = True
                    insights.append(
                        {
                            "metric": "execution_time",
                            "trend_percent": round(duration_trend, 2),
                            "severity": "high" if duration_trend > 50 else "medium",
                        }
                    )

            if len(iterations) >= 2:
                iteration_trend = (
                    (iterations[-1] - iterations[0]) / iterations[0] * 100
                    if iterations[0] > 0
                    else 0
                )
                if iteration_trend > 20:
                    degradation_detected = True
                    insights.append(
                        {
                            "metric": "iterations",
                            "trend_percent": round(iteration_trend, 2),
                            "severity": "high" if iteration_trend > 50 else "medium",
                        }
                    )

            return {
                "degradation_detected": degradation_detected,
                "period_days": days,
                "insights": insights,
                "daily_data": daily_data,
            }
        except Exception as e:
            logger.error(f"Error detecting performance degradation: {e}")
            return {"error": str(e)}

    def detect_success_rate_drop(self, days: int = 7) -> Dict[str, Any]:
        """Detect success rate decline over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with success rate drop insights
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        DATE(s.created_at) as date,
                        COUNT(*) as total,
                        COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) as successful,
                        ROUND(CAST(COUNT(CASE WHEN s.final_status = 'success' THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions s
                    WHERE s.created_at >= datetime('now', '-' || ? || ' days')
                        AND s.final_status IS NOT NULL
                    GROUP BY DATE(s.created_at)
                    ORDER BY date ASC
                    """,
                    (days,),
                )
                daily_data = [dict(row) for row in cursor.fetchall()]

            if len(daily_data) < 2:
                return {
                    "error": "Insufficient data for trend analysis",
                    "data": daily_data,
                }

            success_rates = [
                d["success_rate"] for d in daily_data if d["success_rate"] is not None
            ]

            if len(success_rates) < 2:
                return {"error": "Insufficient success rate data", "data": daily_data}

            drop_detected = False
            drop_percent = 0

            if success_rates[0] > 0:
                drop_percent = (
                    (success_rates[0] - success_rates[-1]) / success_rates[0] * 100
                )
                if drop_percent > 10:
                    drop_detected = True

            return {
                "drop_detected": drop_detected,
                "drop_percent": round(drop_percent, 2),
                "period_days": days,
                "initial_rate": success_rates[0],
                "current_rate": success_rates[-1],
                "daily_data": daily_data,
            }
        except Exception as e:
            logger.error(f"Error detecting success rate drop: {e}")
            return {"error": str(e)}

    def analyze_token_usage_trend(self, days: int = 30) -> Dict[str, Any]:
        """Analyze token usage over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with token usage trends
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        DATE(lr.created_at) as date,
                        lr.model,
                        SUM(lr.prompt_tokens) as total_prompt_tokens,
                        SUM(lr.completion_tokens) as total_completion_tokens,
                        SUM(lr.total_tokens) as total_tokens,
                        AVG(lr.response_time_ms) as avg_response_time_ms
                    FROM llm_responses lr
                    WHERE lr.created_at >= datetime('now', '-' || ? || ' days')
                    GROUP BY DATE(lr.created_at), lr.model
                    ORDER BY date DESC, model
                    """,
                    (days,),
                )
                daily_data = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT 
                    model,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(response_time_ms) as avg_response_time_ms
                FROM llm_responses
                WHERE created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY model
                """,
                (days,),
            )
            model_totals = [dict(row) for row in cursor.fetchall()]

            return {
                "period_days": days,
                "daily_data": daily_data,
                "model_totals": model_totals,
                "total_tokens": sum(d["total_tokens"] for d in model_totals),
            }
        except Exception as e:
            logger.error(f"Error analyzing token usage trend: {e}")
            return {"error": str(e)}

    def compare_periods(self, days1: int = 7, days2: int = 7) -> Dict[str, Any]:
        """Compare two time periods.

        Args:
            days1: Number of days for period 1 (recent)
            days2: Number of days for period 2 (previous)

        Returns:
            Dictionary with period comparison
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                        AVG(total_duration_ms) as avg_duration_ms,
                        AVG(total_iterations) as avg_iterations,
                        AVG(total_tool_calls) as avg_tool_calls
                    FROM sessions
                    WHERE created_at >= datetime('now', '-' || ? || ' days')
                    """,
                    (days1,),
                )
                period1 = dict(cursor.fetchone())

                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                        AVG(total_duration_ms) as avg_duration_ms,
                        AVG(total_iterations) as avg_iterations,
                        AVG(total_tool_calls) as avg_tool_calls
                    FROM sessions
                    WHERE created_at >= datetime('now', '-' || (? + ?) || ' days')
                        AND created_at < datetime('now', '-' || ? || ' days')
                    """,
                    (days1, days2, days1),
                )
                period2 = dict(cursor.fetchone())

            comparison = {}
            for key in [
                "success_rate",
                "avg_duration_ms",
                "avg_iterations",
                "avg_tool_calls",
            ]:
                if period1.get(key) and period2.get(key):
                    change = ((period1[key] - period2[key]) / period2[key]) * 100
                    comparison[key] = {
                        "period1": round(period1[key], 2),
                        "period2": round(period2[key], 2),
                        "change_percent": round(change, 2),
                        "trend": "improved"
                        if (key == "success_rate" and change > 0)
                        or (key != "success_rate" and change < 0)
                        else "degraded",
                    }

            return {
                "period1_days": days1,
                "period2_days": days2,
                "period1": period1,
                "period2": period2,
                "comparison": comparison,
            }
        except Exception as e:
            logger.error(f"Error comparing periods: {e}")
            return {"error": str(e)}

    def generate_daily_insights(self, days: int = 1) -> Dict[str, Any]:
        """Generate daily insights.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with daily insights
        """
        try:
            insights = {
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "summary": {},
                "recommendations": [],
            }

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                        AVG(total_duration_ms) as avg_duration_ms,
                        SUM(total_tool_calls) as total_tool_calls,
                        AVG(total_iterations) as avg_iterations
                    FROM sessions
                    WHERE created_at >= datetime('now', '-' || ? || ' days')
                    """,
                    (days,),
                )
                insights["summary"] = dict(cursor.fetchone())

                cursor.execute(
                    """
                    SELECT tool_name, COUNT(*) as failed_count
                    FROM tool_calls
                    WHERE success = 0 
                        AND created_at >= datetime('now', '-' || ? || ' days')
                    GROUP BY tool_name
                    ORDER BY failed_count DESC
                    LIMIT 5
                    """,
                    (days,),
                )
                failing_tools = [dict(row) for row in cursor.fetchall()]
                if failing_tools:
                    insights["failing_tools"] = failing_tools
                    insights["recommendations"].append(
                        {
                            "type": "tool_issues",
                            "message": f"Top failing tool: {failing_tools[0]['tool_name']} ({failing_tools[0]['failed_count']} failures)",
                            "priority": "high"
                            if failing_tools[0]["failed_count"] > 10
                            else "medium",
                        }
                    )

                cursor.execute(
                    """
                    SELECT model, COUNT(*) as count
                    FROM sessions
                    WHERE created_at >= datetime('now', '-' || ? || ' days')
                    GROUP BY model
                    ORDER BY count DESC
                    """,
                    (days,),
                )
                insights["model_usage"] = [dict(row) for row in cursor.fetchall()]

            return insights
        except Exception as e:
            logger.error(f"Error generating daily insights: {e}")
            return {"error": str(e)}

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly summary report.

        Returns:
            Dictionary with weekly report
        """
        try:
            report = {
                "period_days": 7,
                "generated_at": datetime.now().isoformat(),
                "overview": {},
                "performance": {},
                "trends": {},
                "recommendations": [],
            }

            report["overview"] = self.generate_daily_insights(days=7).get("summary", {})

            report["performance"]["tools"] = self.db.get_tool_success_rate()
            report["performance"]["models"] = self.get_model_comparison()
            report["performance"]["execution_times"] = (
                self.get_execution_time_distribution()
            )

            degradation = self.detect_performance_degradation(days=7)
            report["trends"]["degradation"] = degradation

            success_drop = self.detect_success_rate_drop(days=7)
            report["trends"]["success_rate"] = success_drop

            period_comparison = self.compare_periods(7, 7)
            report["trends"]["period_comparison"] = period_comparison

            failing_tools = self.find_failing_tools(threshold=30)
            if failing_tools:
                report["recommendations"].append(
                    {
                        "type": "failing_tools",
                        "count": len(failing_tools),
                        "tools": [t["tool_name"] for t in failing_tools[:3]],
                        "priority": "high",
                    }
                )

            return report
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return {"error": str(e)}

    def generate_model_recommendations(self) -> Dict[str, Any]:
        """Generate model usage recommendations.

        Returns:
            Dictionary with model recommendations
        """
        try:
            recommendations = {
                "generated_at": datetime.now().isoformat(),
                "best_for_speed": {},
                "best_for_quality": {},
                "best_for_efficiency": {},
                "overall_recommendation": "",
            }

            models = self.get_model_comparison()
            if not models:
                return {"error": "No model data available"}

            speed_sorted = sorted(
                models, key=lambda x: x.get("avg_response_time_ms", float("inf"))
            )
            quality_sorted = sorted(
                models, key=lambda x: x.get("total_requests", 0), reverse=True
            )
            efficiency_sorted = sorted(
                models, key=lambda x: x.get("avg_token_efficiency", 0), reverse=True
            )

            if speed_sorted:
                recommendations["best_for_speed"] = {
                    "model": speed_sorted[0]["model"],
                    "avg_response_time_ms": speed_sorted[0]["avg_response_time_ms"],
                }

            if quality_sorted:
                recommendations["best_for_quality"] = {
                    "model": quality_sorted[0]["model"],
                    "total_requests": quality_sorted[0]["total_requests"],
                }

            if efficiency_sorted:
                recommendations["best_for_efficiency"] = {
                    "model": efficiency_sorted[0]["model"],
                    "token_efficiency": efficiency_sorted[0]["avg_token_efficiency"],
                }

            if efficiency_sorted:
                recommendations["overall_recommendation"] = efficiency_sorted[0][
                    "model"
                ]

            return recommendations
        except Exception as e:
            logger.error(f"Error generating model recommendations: {e}")
            return {"error": str(e)}

    def generate_optimization_suggestions(self) -> Dict[str, Any]:
        """Generate performance optimization suggestions.

        Returns:
            Dictionary with optimization suggestions
        """
        try:
            suggestions = {
                "generated_at": datetime.now().isoformat(),
                "suggestions": [],
            }

            failing_tools = self.find_failing_tools(threshold=20)
            for tool in failing_tools[:5]:
                suggestions.append(
                    {
                        "category": "tool_reliability",
                        "priority": "high",
                        "tool": tool["tool_name"],
                        "failure_rate": tool["failure_rate"],
                        "suggestion": f"Investigate and fix issues with {tool['tool_name']} (failure rate: {tool['failure_rate']}%)",
                    }
                )

            degradation = self.detect_performance_degradation()
            if degradation.get("degradation_detected"):
                for insight in degradation.get("insights", []):
                    suggestions.append(
                        {
                            "category": "performance",
                            "priority": insight["severity"],
                            "metric": insight["metric"],
                            "suggestion": f"Performance degradation detected in {insight['metric']} ({insight['trend_percent']}% increase)",
                        }
                    )

            optimal_params = self.find_optimal_parameters()
            temp_data = optimal_params.get("by_temperature", [])
            if temp_data:
                best_temp = temp_data[0]
                suggestions.append(
                    {
                        "category": "parameters",
                        "priority": "medium",
                        "suggestion": f"Consider using temperature {best_temp['temperature']} for better token efficiency",
                    }
                )

            execution_times = self.get_execution_time_distribution()
            if execution_times.get("p95_ms") and execution_times.get("mean_ms"):
                if execution_times["p95_ms"] > execution_times["mean_ms"] * 3:
                    suggestions.append(
                        {
                            "category": "outliers",
                            "priority": "medium",
                            "suggestion": "High variance in execution times detected. Investigate outlier cases.",
                        }
                    )

            return suggestions
        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return {"error": str(e)}

    def get_test_performance_summary(self) -> Dict[str, Any]:
        """Get overall test performance summary.

        Returns:
            Dictionary with test performance summary
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_test_cases,
                        COUNT(CASE WHEN total_runs > 0 THEN 1 END) as executed_test_cases,
                        SUM(total_runs) as total_runs,
                        SUM(successful_runs) as total_successful_runs,
                        ROUND(AVG(success_rate), 2) as avg_success_rate
                    FROM test_performance
                    """
                )
                summary = dict(cursor.fetchone())

                cursor.execute(
                    """
                    SELECT category, COUNT(*) as count,
                           ROUND(AVG(success_rate), 2) as avg_success_rate
                    FROM test_performance
                    WHERE total_runs > 0
                    GROUP BY category
                    """
                )
                by_category = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT test_case_name, success_rate, total_runs, last_run
                    FROM test_performance
                    WHERE total_runs > 0
                    ORDER BY success_rate ASC
                    LIMIT 10
                    """
                )
                worst_performing = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT test_case_name, success_rate, total_runs, last_run
                    FROM test_performance
                    WHERE total_runs > 0
                    ORDER BY success_rate DESC
                    LIMIT 10
                    """
                )
                best_performing = [dict(row) for row in cursor.fetchall()]

            return {
                "summary": summary,
                "by_category": by_category,
                "best_performing": best_performing,
                "worst_performing": worst_performing,
            }
        except Exception as e:
            logger.error(f"Error getting test performance summary: {e}")
            return {"error": str(e)}

    def get_failing_test_cases(
        self, threshold: float = 50.0, min_runs: int = 3
    ) -> List[Dict[str, Any]]:
        """Get test cases that consistently fail.

        Args:
            threshold: Success rate threshold below which tests are considered failing
            min_runs: Minimum number of runs required

        Returns:
            List of failing test cases
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        test_case_id,
                        test_case_name,
                        category,
                        total_runs,
                        successful_runs,
                        success_rate,
                        last_run
                    FROM test_performance
                    WHERE total_runs >= ? AND success_rate < ?
                    ORDER BY success_rate ASC, total_runs DESC
                    """,
                    (min_runs, threshold),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting failing test cases: {e}")
            return []

    def get_regression_issues(self, days: int = 14) -> List[Dict[str, Any]]:
        """Get tests that have regressed (success rate dropped).

        Args:
            days: Number of days to analyze for regression

        Returns:
            List of regressed test cases
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        tc.id as test_case_id,
                        tc.name as test_case_name,
                        tc.category,
                        COUNT(CASE WHEN tr.completed_at >= datetime('now', '-' || ? || ' days') THEN 1 END) as recent_runs,
                        COUNT(CASE WHEN tr.completed_at >= datetime('now', '-' || ? || ' days') 
                                    AND tr.final_status = 'success' THEN 1 END) as recent_successful,
                        ROUND(CAST(COUNT(CASE WHEN tr.completed_at >= datetime('now', '-' || ? || ' days') 
                                        AND tr.final_status = 'success' THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(CASE WHEN tr.completed_at >= datetime('now', '-' || ? || ' days') THEN 1 END), 0) * 100, 2) as recent_success_rate,
                        tp.success_rate as historical_success_rate
                    FROM test_cases tc
                    LEFT JOIN test_runs tr ON tc.id = tr.test_case_id
                    LEFT JOIN test_performance tp ON tc.id = tp.test_case_id
                    WHERE tp.total_runs >= 5
                    GROUP BY tc.id
                    HAVING recent_runs >= 3 AND (recent_success_rate < historical_success_rate * 0.9)
                    ORDER BY (historical_success_rate - recent_success_rate) DESC
                    """,
                    (days, days, days, days),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting regression issues: {e}")
            return []

    def compare_test_runs(self, test_case_id: int) -> Dict[str, Any]:
        """Compare test runs over time for a specific test case.

        Args:
            test_case_id: ID of the test case

        Returns:
            Dictionary with test run comparison
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT tc.name as test_case_name, tc.category
                    FROM test_cases tc
                    WHERE tc.id = ?
                    """,
                    (test_case_id,),
                )
                test_info = dict(cursor.fetchone())

                cursor.execute(
                    """
                    SELECT 
                        tr.id,
                        tr.run_number,
                        tr.status,
                        tr.final_status,
                        tr.total_iterations,
                        tr.total_tool_calls,
                        tr.started_at,
                        tr.completed_at,
                        COUNT(je.id) as total_evaluations,
                        COUNT(CASE WHEN je.passed = 1 THEN 1 END) as passed_evaluations,
                        ROUND(CAST(COUNT(CASE WHEN je.passed = 1 THEN 1 END) AS FLOAT) 
                              / NULLIF(COUNT(je.id), 0) * 100, 2) as evaluation_pass_rate
                    FROM test_runs tr
                    LEFT JOIN judge_evaluations je ON tr.id = je.test_run_id
                    WHERE tr.test_case_id = ?
                    GROUP BY tr.id
                    ORDER BY tr.run_number ASC
                    """,
                    (test_case_id,),
                )
                runs = [dict(row) for row in cursor.fetchall()]

                if not runs:
                    return {
                        "error": f"No runs found for test case {test_case_id}",
                        "test_case_id": test_case_id,
                    }

                success_rates = [
                    r["evaluation_pass_rate"]
                    for r in runs
                    if r["evaluation_pass_rate"] is not None
                ]
                iterations = [
                    r["total_iterations"]
                    for r in runs
                    if r["total_iterations"] is not None
                ]

                trend = {}
                if len(success_rates) >= 2:
                    trend["success_rate_change"] = round(
                        success_rates[-1] - success_rates[0], 2
                    )
                    trend["success_rate_trend"] = (
                        "improving" if trend["success_rate_change"] > 0 else "declining"
                    )

                if len(iterations) >= 2:
                    trend["iteration_change"] = round(iterations[-1] - iterations[0], 2)
                    trend["iteration_trend"] = (
                        "increasing" if trend["iteration_change"] > 0 else "decreasing"
                    )

                return {
                    "test_case_id": test_case_id,
                    "test_info": test_info,
                    "total_runs": len(runs),
                    "runs": runs,
                    "trend": trend,
                }
        except Exception as e:
            logger.error(f"Error comparing test runs: {e}")
            return {"error": str(e), "test_case_id": test_case_id}

    def export_metrics_to_json(self, file_path: str) -> str:
        """Export all metrics to JSON file.

        Args:
            file_path: Path to output JSON file

        Returns:
            Path to exported file
        """
        try:
            metrics = {
                "exported_at": datetime.now().isoformat(),
                "success_rate_trend": self.get_success_rate_trend(days=30),
                "tool_performance": self.db.get_tool_success_rate(),
                "model_comparison": self.get_model_comparison(),
                "execution_time_distribution": self.get_execution_time_distribution(),
                "iteration_statistics": self.get_iteration_statistics(),
                "test_performance": self.get_test_performance_summary(),
            }

            with open(file_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"Metrics exported to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise

    def export_insights_to_json(self, file_path: str) -> str:
        """Export insights to JSON file.

        Args:
            file_path: Path to output JSON file

        Returns:
            Path to exported file
        """
        try:
            insights = {
                "exported_at": datetime.now().isoformat(),
                "daily_insights": self.generate_daily_insights(days=1),
                "weekly_report": self.generate_weekly_report(),
                "model_recommendations": self.generate_model_recommendations(),
                "optimization_suggestions": self.generate_optimization_suggestions(),
                "failing_tools": self.find_failing_tools(threshold=30),
                "successful_patterns": self.find_successful_patterns(),
                "error_patterns": self.find_error_patterns(),
            }

            with open(file_path, "w") as f:
                json.dump(insights, f, indent=2, default=str)

            logger.info(f"Insights exported to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error exporting insights: {e}")
            raise

    def generate_report(self, format: str = "json") -> str:
        """Generate comprehensive report.

        Args:
            format: Output format ('json' or 'dict')

        Returns:
            Report as JSON string or dictionary
        """
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "success_rate_trend": self.get_success_rate_trend(days=30),
                    "performance_comparison": self.compare_periods(7, 7),
                },
                "performance": {
                    "tools": self.db.get_tool_success_rate(),
                    "models": self.get_model_comparison(),
                    "execution_times": self.get_execution_time_distribution(),
                    "iterations": self.get_iteration_statistics(),
                },
                "patterns": {
                    "failing_tools": self.find_failing_tools(threshold=30),
                    "successful_patterns": self.find_successful_patterns(),
                    "error_patterns": self.find_error_patterns(),
                    "optimal_parameters": self.find_optimal_parameters(),
                },
                "trends": {
                    "degradation": self.detect_performance_degradation(days=7),
                    "success_rate_drop": self.detect_success_rate_drop(days=7),
                    "token_usage": self.analyze_token_usage_trend(days=30),
                },
                "tests": {
                    "summary": self.get_test_performance_summary(),
                    "failing_tests": self.get_failing_test_cases(threshold=50),
                    "regressions": self.get_regression_issues(days=14),
                },
                "insights": {
                    "daily": self.generate_daily_insights(days=1),
                    "weekly": self.generate_weekly_report(),
                    "model_recommendations": self.generate_model_recommendations(),
                    "optimization_suggestions": self.generate_optimization_suggestions(),
                },
            }

            if format == "json":
                return json.dumps(report, indent=2, default=str)
            else:
                return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            if format == "json":
                return json.dumps({"error": str(e)})
            else:
                return {"error": str(e)}

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0
        values_sorted = sorted(values)
        n = len(values_sorted)
        index = int(n * percentile / 100)
        return values_sorted[min(index, n - 1)]

    def _calculate_moving_average(
        self, values: List[float], window: int
    ) -> List[float]:
        """Calculate moving average.

        Args:
            values: List of numeric values
            window: Window size for moving average

        Returns:
            List of moving average values
        """
        if not values or window <= 0:
            return []
        moving_avg = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = values[start : i + 1]
            moving_avg.append(round(mean(window_values), 2) if window_values else 0)
        return moving_avg

    def _detect_anomalies(
        self, values: List[float], threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect statistical anomalies using standard deviation.

        Args:
            values: List of numeric values
            threshold: Standard deviation threshold for anomaly detection

        Returns:
            List of anomalies with indices and values
        """
        if len(values) < 3:
            return []

        try:
            mean_val = mean(values)
            std_val = stdev(values)

            if std_val == 0:
                return []

            anomalies = []
            for i, val in enumerate(values):
                z_score = abs((val - mean_val) / std_val)
                if z_score > threshold:
                    anomalies.append(
                        {
                            "index": i,
                            "value": val,
                            "z_score": round(z_score, 2),
                            "deviation": round(val - mean_val, 2),
                        }
                    )

            return anomalies
        except Exception:
            return []

    def _format_insight(self, insight: Dict[str, Any]) -> str:
        """Format insight for display.

        Args:
            insight: Insight dictionary

        Returns:
            Formatted string
        """
        if not insight:
            return "No insight available"

        parts = []
        for key, value in insight.items():
            if key not in ["generated_at", "error"]:
                parts.append(f"{key}: {value}")

        return " | ".join(parts) if parts else str(insight)
