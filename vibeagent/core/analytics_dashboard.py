import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    metric: str
    threshold: float
    operator: str
    enabled: bool = True
    notification_channels: list[str] = None

    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = []


@dataclass
class Alert:
    id: str
    metric: str
    threshold: float
    actual_value: float
    message: str
    severity: str
    triggered_at: datetime
    resolved: bool = False
    resolved_at: datetime | None = None


@dataclass
class DashboardConfig:
    time_range: str = "24h"
    filters: dict[str, Any] = None
    refresh_interval: int = 60

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class AnalyticsDashboard:
    def __init__(self, db_manager, analytics_engine):
        self.db = db_manager
        self.analytics = analytics_engine
        self.config = DashboardConfig()
        self.alerts: list[Alert] = []
        self.alert_configs: dict[str, AlertConfig] = {}
        self._cache = {}
        self._cache_ttl = 300

    def _get_time_filter(self) -> tuple[str, str]:
        time_ranges = {
            "1h": ("datetime('now', '-1 hours')", "1 hour"),
            "24h": ("datetime('now', '-1 days')", "24 hours"),
            "7d": ("datetime('now', '-7 days')", "7 days"),
            "30d": ("datetime('now', '-30 days')", "30 days"),
            "all": ("datetime('1970-01-01')", "all time"),
        }
        return time_ranges.get(self.config.time_range, time_ranges["24h"])

    def _get_cache_key(self, method_name: str, **kwargs) -> str:
        key = f"{method_name}_{self.config.time_range}"
        for k, v in sorted(kwargs.items()):
            key += f"_{k}_{v}"
        return key

    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            cached_time, data = self._cache[key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                return data
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (datetime.now(), data)

    def _clear_cache(self):
        self._cache.clear()

    def set_time_range(self, time_range: str):
        valid_ranges = ["1h", "24h", "7d", "30d", "all"]
        if time_range in valid_ranges:
            self.config.time_range = time_range
            self._clear_cache()
            logger.info(f"Time range set to {time_range}")

    def set_filters(self, **filters):
        self.config.filters.update(filters)
        self._clear_cache()

    def clear_filters(self):
        self.config.filters = {}
        self._clear_cache()

    def get_overview_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("overview_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        time_filter, _ = self._get_time_filter()

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT)
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                        AVG(total_duration_ms) as avg_duration_ms,
                        SUM(total_tool_calls) as total_tool_calls,
                        AVG(total_iterations) as avg_iterations,
                        COUNT(DISTINCT model) as unique_models,
                        COUNT(DISTINCT session_type) as unique_session_types
                    FROM sessions
                    WHERE created_at >= {time_filter}
                    """
                )
                overview = dict(cursor.fetchone())

                cursor.execute(
                    f"""
                    SELECT
                        SUM(prompt_tokens) as total_prompt_tokens,
                        SUM(completion_tokens) as total_completion_tokens,
                        SUM(total_tokens) as total_tokens,
                        AVG(response_time_ms) as avg_response_time_ms
                    FROM llm_responses
                    WHERE created_at >= {time_filter}
                    """
                )
                token_data = dict(cursor.fetchone())
                overview.update(token_data)

                cursor.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_tool_calls,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_tool_calls,
                        ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT)
                              / NULLIF(COUNT(*), 0) * 100, 2) as tool_success_rate,
                        AVG(execution_time_ms) as avg_tool_execution_time_ms
                    FROM tool_calls
                    WHERE created_at >= {time_filter}
                    """
                )
                tool_data = dict(cursor.fetchone())
                overview.update(tool_data)

            self._set_cache(cache_key, overview)
            return overview
        except Exception as e:
            logger.error(f"Error getting overview panel: {e}")
            return {"error": str(e)}

    def get_performance_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("performance_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        time_filter, _ = self._get_time_filter()

        try:
            performance = {
                "execution_times": self.analytics.get_execution_time_distribution(),
                "iteration_stats": self.analytics.get_iteration_statistics(),
                "daily_trends": [],
                "hourly_patterns": [],
            }

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                    SELECT
                        DATE(created_at) as date,
                        AVG(total_duration_ms) as avg_duration_ms,
                        AVG(total_iterations) as avg_iterations,
                        COUNT(*) as session_count
                    FROM sessions
                    WHERE created_at >= {time_filter}
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                    """
                )
                performance["daily_trends"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    f"""
                    SELECT
                        strftime('%H', created_at) as hour,
                        AVG(total_duration_ms) as avg_duration_ms,
                        COUNT(*) as session_count
                    FROM sessions
                    WHERE created_at >= {time_filter}
                    GROUP BY hour
                    ORDER BY hour
                    """
                )
                performance["hourly_patterns"] = [dict(row) for row in cursor.fetchall()]

            self._set_cache(cache_key, performance)
            return performance
        except Exception as e:
            logger.error(f"Error getting performance panel: {e}")
            return {"error": str(e)}

    def get_test_results_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("test_results_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            test_results = {
                "summary": self.analytics.get_test_performance_summary(),
                "failing_tests": self.analytics.get_failing_test_cases(threshold=50),
                "regressions": self.analytics.get_regression_issues(days=14),
                "recent_runs": [],
            }

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        tr.id,
                        tc.name as test_case_name,
                        tc.category,
                        tr.status,
                        tr.final_status,
                        tr.started_at,
                        tr.completed_at
                    FROM test_runs tr
                    JOIN test_cases tc ON tr.test_case_id = tc.id
                    ORDER BY tr.started_at DESC
                    LIMIT 20
                    """
                )
                test_results["recent_runs"] = [dict(row) for row in cursor.fetchall()]

            self._set_cache(cache_key, test_results)
            return test_results
        except Exception as e:
            logger.error(f"Error getting test results panel: {e}")
            return {"error": str(e)}

    def get_tool_usage_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("tool_usage_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        time_filter, _ = self._get_time_filter()

        try:
            tool_usage = {
                "tools": self.db.get_tool_success_rate(),
                "failing_tools": self.analytics.find_failing_tools(threshold=30),
                "tool_trends": [],
                "tool_errors": [],
            }

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                    SELECT
                        DATE(created_at) as date,
                        tool_name,
                        COUNT(*) as call_count,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as success_count
                    FROM tool_calls
                    WHERE created_at >= {time_filter}
                    GROUP BY date, tool_name
                    ORDER BY date DESC, call_count DESC
                    """
                )
                tool_usage["tool_trends"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    f"""
                    SELECT
                        tool_name,
                        error_type,
                        COUNT(*) as error_count
                    FROM tool_calls
                    WHERE success = 0 AND created_at >= {time_filter}
                    GROUP BY tool_name, error_type
                    ORDER BY error_count DESC
                    LIMIT 20
                    """
                )
                tool_usage["tool_errors"] = [dict(row) for row in cursor.fetchall()]

            self._set_cache(cache_key, tool_usage)
            return tool_usage
        except Exception as e:
            logger.error(f"Error getting tool usage panel: {e}")
            return {"error": str(e)}

    def get_model_comparison_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("model_comparison_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            model_comparison = {
                "models": self.analytics.get_model_comparison(),
                "recommendations": self.analytics.generate_model_recommendations(),
                "token_usage": [],
                "response_times": [],
            }

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        model,
                        DATE(created_at) as date,
                        SUM(total_tokens) as total_tokens,
                        AVG(response_time_ms) as avg_response_time_ms,
                        COUNT(*) as request_count
                    FROM llm_responses
                    WHERE created_at >= datetime('now', '-30 days')
                    GROUP BY model, date
                    ORDER BY date DESC, model
                    """
                )
                model_comparison["token_usage"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT
                        model,
                        ROUND(AVG(response_time_ms), 2) as avg_response_time_ms,
                        ROUND(MIN(response_time_ms), 2) as min_response_time_ms,
                        ROUND(MAX(response_time_ms), 2) as max_response_time_ms,
                        COUNT(*) as request_count
                    FROM llm_responses
                    WHERE created_at >= datetime('now', '-30 days')
                    GROUP BY model
                    ORDER BY avg_response_time_ms ASC
                    """
                )
                model_comparison["response_times"] = [dict(row) for row in cursor.fetchall()]

            self._set_cache(cache_key, model_comparison)
            return model_comparison
        except Exception as e:
            logger.error(f"Error getting model comparison panel: {e}")
            return {"error": str(e)}

    def get_error_analysis_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("error_analysis_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        time_filter, _ = self._get_time_filter()

        try:
            error_analysis = {
                "patterns": self.analytics.find_error_patterns(),
                "recovery_rates": [],
                "error_timeline": [],
                "top_errors": [],
            }

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        error_type,
                        recovery_strategy,
                        COUNT(*) as total_attempts,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_recoveries,
                        ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT)
                              / NULLIF(COUNT(*), 0) * 100, 2) as recovery_rate
                    FROM error_recovery
                    GROUP BY error_type, recovery_strategy
                    ORDER BY recovery_rate DESC
                    """
                )
                error_analysis["recovery_rates"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    f"""
                    SELECT
                        DATE(created_at) as date,
                        error_type,
                        COUNT(*) as error_count
                    FROM tool_calls
                    WHERE success = 0 AND created_at >= {time_filter}
                    GROUP BY date, error_type
                    ORDER BY date DESC, error_count DESC
                    """
                )
                error_analysis["error_timeline"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    f"""
                    SELECT
                        error_type,
                        COUNT(*) as total_errors,
                        ROUND(CAST(COUNT(*) AS FLOAT) * 100.0 /
                              (SELECT COUNT(*) FROM tool_calls WHERE success = 0 AND created_at >= {time_filter}), 2) as percentage
                    FROM tool_calls
                    WHERE success = 0 AND created_at >= {time_filter}
                    GROUP BY error_type
                    ORDER BY total_errors DESC
                    LIMIT 10
                    """
                )
                error_analysis["top_errors"] = [dict(row) for row in cursor.fetchall()]

            self._set_cache(cache_key, error_analysis)
            return error_analysis
        except Exception as e:
            logger.error(f"Error getting error analysis panel: {e}")
            return {"error": str(e)}

    def get_trends_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("trends_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            trends = {
                "success_rate": self.analytics.get_success_rate_trend(days=30),
                "degradation": self.analytics.detect_performance_degradation(days=7),
                "success_rate_drop": self.analytics.detect_success_rate_drop(days=7),
                "token_usage": self.analytics.analyze_token_usage_trend(days=30),
                "period_comparison": self.analytics.compare_periods(7, 7),
            }

            self._set_cache(cache_key, trends)
            return trends
        except Exception as e:
            logger.error(f"Error getting trends panel: {e}")
            return {"error": str(e)}

    def get_insights_panel(self) -> dict[str, Any]:
        cache_key = self._get_cache_key("insights_panel")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            insights = {
                "daily_insights": self.analytics.generate_daily_insights(days=1),
                "weekly_report": self.analytics.generate_weekly_report(),
                "optimization_suggestions": self.analytics.generate_optimization_suggestions(),
                "successful_patterns": self.analytics.find_successful_patterns(),
                "alerts": [asdict(alert) for alert in self.alerts if not alert.resolved],
            }

            self._set_cache(cache_key, insights)
            return insights
        except Exception as e:
            logger.error(f"Error getting insights panel: {e}")
            return {"error": str(e)}

    def get_full_dashboard(self) -> dict[str, Any]:
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "config": asdict(self.config),
            "panels": {
                "overview": self.get_overview_panel(),
                "performance": self.get_performance_panel(),
                "test_results": self.get_test_results_panel(),
                "tool_usage": self.get_tool_usage_panel(),
                "model_comparison": self.get_model_comparison_panel(),
                "error_analysis": self.get_error_analysis_panel(),
                "trends": self.get_trends_panel(),
                "insights": self.get_insights_panel(),
            },
        }
        return dashboard

    def generate_report(self, format: str = "json") -> str:
        dashboard = self.get_full_dashboard()

        if format.lower() == "json":
            return json.dumps(dashboard, indent=2, default=str)
        if format.lower() == "dict":
            return dashboard
        if format.lower() == "html":
            return self._generate_html_report(dashboard)
        if format.lower() == "markdown":
            return self._generate_markdown_report(dashboard)
        if format.lower() == "csv":
            return self._generate_csv_report(dashboard)
        raise ValueError(f"Unsupported format: {format}")

    def generate_summary(self) -> dict[str, Any]:
        overview = self.get_overview_panel()
        insights = self.get_insights_panel()
        trends = self.get_trends_panel()

        summary = {
            "generated_at": datetime.now().isoformat(),
            "time_range": self.config.time_range,
            "key_metrics": {
                "total_sessions": overview.get("total_sessions", 0),
                "success_rate": overview.get("success_rate", 0),
                "avg_duration_ms": overview.get("avg_duration_ms", 0),
                "total_tokens": overview.get("total_tokens", 0),
            },
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "degradation_detected": trends.get("degradation", {}).get(
                "degradation_detected", False
            ),
            "success_rate_drop": trends.get("success_rate_drop", {}).get("drop_detected", False),
            "top_recommendations": insights.get("optimization_suggestions", {}).get(
                "suggestions", []
            )[:5],
        }
        return summary

    def generate_detailed_report(self) -> dict[str, Any]:
        return self.get_full_dashboard()

    def generate_comparison_report(self, metric: str = "success_rate") -> dict[str, Any]:
        comparison = {
            "generated_at": datetime.now().isoformat(),
            "metric": metric,
            "comparison": self.analytics.compare_periods(7, 7),
            "model_comparison": self.analytics.get_model_comparison(),
            "tool_comparison": self.db.get_tool_success_rate(),
        }
        return comparison

    def _generate_html_report(self, dashboard: dict[str, Any]) -> str:
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>VibeAgent Analytics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .panel {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .panel h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: #ecf0f1; padding: 15px; border-radius: 4px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .alert {{ padding: 10px; margin: 10px 0; border-radius: 4px; }}
        .alert-high {{ background: #e74c3c; color: white; }}
        .alert-medium {{ background: #f39c12; color: white; }}
        .alert-low {{ background: #27ae60; color: white; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>VibeAgent Analytics Dashboard</h1>
            <p class="timestamp">Generated: {dashboard["generated_at"]}</p>
            <p>Time Range: {dashboard["config"]["time_range"]}</p>
        </div>
"""

        overview = dashboard["panels"]["overview"]
        if "error" not in overview:
            html += """
        <div class="panel">
            <h2>Overview</h2>
            <div class="metrics">
"""
            metrics = [
                ("Total Sessions", overview.get("total_sessions", 0)),
                ("Success Rate", f"{overview.get('success_rate', 0)}%"),
                ("Avg Duration", f"{overview.get('avg_duration_ms', 0)}ms"),
                ("Total Tool Calls", overview.get("total_tool_calls", 0)),
                ("Avg Iterations", f"{overview.get('avg_iterations', 0):.1f}"),
                ("Total Tokens", overview.get("total_tokens", 0)),
            ]
            for label, value in metrics:
                html += f"""
                <div class="metric">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
"""
            html += """
            </div>
        </div>
"""

        insights = dashboard["panels"]["insights"]
        if "error" not in insights:
            suggestions = insights.get("optimization_suggestions", {}).get("suggestions", [])
            if suggestions:
                html += """
        <div class="panel">
            <h2>Recommendations</h2>
"""
                for suggestion in suggestions[:10]:
                    priority = suggestion.get("priority", "low")
                    html += f"""
            <div class="alert alert-{priority}">
                <strong>{suggestion.get("category", "General")}:</strong> {suggestion.get("suggestion", "")}
            </div>
"""
                html += """
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def _generate_markdown_report(self, dashboard: dict[str, Any]) -> str:
        md = "# VibeAgent Analytics Dashboard\n\n"
        md += f"**Generated:** {dashboard['generated_at']}\n"
        md += f"**Time Range:** {dashboard['config']['time_range']}\n\n"

        overview = dashboard["panels"]["overview"]
        if "error" not in overview:
            md += "## Overview\n\n"
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            md += f"| Total Sessions | {overview.get('total_sessions', 0)} |\n"
            md += f"| Success Rate | {overview.get('success_rate', 0)}% |\n"
            md += f"| Avg Duration | {overview.get('avg_duration_ms', 0)}ms |\n"
            md += f"| Total Tool Calls | {overview.get('total_tool_calls', 0)} |\n"
            md += f"| Avg Iterations | {overview.get('avg_iterations', 0):.1f} |\n"
            md += f"| Total Tokens | {overview.get('total_tokens', 0)} |\n\n"

        insights = dashboard["panels"]["insights"]
        if "error" not in insights:
            suggestions = insights.get("optimization_suggestions", {}).get("suggestions", [])
            if suggestions:
                md += "## Recommendations\n\n"
                for i, suggestion in enumerate(suggestions[:10], 1):
                    md += f"{i}. **[{suggestion.get('priority', 'low').upper()}] {suggestion.get('category', 'General')}**: {suggestion.get('suggestion', '')}\n"
                md += "\n"

        return md

    def _generate_csv_report(self, dashboard: dict[str, Any]) -> str:
        csv_lines = []

        overview = dashboard["panels"]["overview"]
        if "error" not in overview:
            csv_lines.append("Overview Metrics")
            csv_lines.append("Metric,Value")
            csv_lines.append(f"Total Sessions,{overview.get('total_sessions', 0)}")
            csv_lines.append(f"Success Rate,{overview.get('success_rate', 0)}")
            csv_lines.append(f"Avg Duration (ms),{overview.get('avg_duration_ms', 0)}")
            csv_lines.append(f"Total Tool Calls,{overview.get('total_tool_calls', 0)}")
            csv_lines.append(f"Avg Iterations,{overview.get('avg_iterations', 0)}")
            csv_lines.append(f"Total Tokens,{overview.get('total_tokens', 0)}")
            csv_lines.append("")

        tool_usage = dashboard["panels"]["tool_usage"]
        if "error" not in tool_usage and tool_usage.get("tools"):
            csv_lines.append("Tool Performance")
            csv_lines.append("Tool Name,Success Rate,Total Calls")
            for tool in tool_usage["tools"][:20]:
                csv_lines.append(
                    f"{tool.get('tool_name', '')},{tool.get('success_rate', 0)},{tool.get('total_calls', 0)}"
                )
            csv_lines.append("")

        return "\n".join(csv_lines)

    def generate_pdf_report(self, output_path: str) -> str:
        try:
            import weasyprint

            html_report = self.generate_report(format="html")
            weasyprint.HTML(string=html_report).write_pdf(output_path)
            logger.info(f"PDF report generated: {output_path}")
            return output_path
        except ImportError:
            logger.warning("weasyprint not installed, generating HTML instead")
            html_path = output_path.replace(".pdf", ".html")
            with open(html_path, "w") as f:
                f.write(html_report)
            return html_path
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise

    def configure_alert(self, metric: str, threshold: float, operator: str = "greater_than"):
        alert_id = f"{metric}_{threshold}_{operator}"
        self.alert_configs[alert_id] = AlertConfig(
            metric=metric, threshold=threshold, operator=operator
        )
        logger.info(f"Alert configured: {alert_id}")

    def check_alerts(self):
        overview = self.get_overview_panel()
        alerts_triggered = []

        for alert_id, config in self.alert_configs.items():
            if not config.enabled:
                continue

            metric_value = overview.get(config.metric)
            if metric_value is None:
                continue

            triggered = False
            if (
                config.operator == "greater_than"
                and metric_value > config.threshold
                or config.operator == "less_than"
                and metric_value < config.threshold
                or config.operator == "equals"
                and metric_value == config.threshold
            ):
                triggered = True

            if triggered:
                severity = (
                    "high"
                    if abs(metric_value - config.threshold) > config.threshold * 0.2
                    else "medium"
                )
                alert = Alert(
                    id=alert_id,
                    metric=config.metric,
                    threshold=config.threshold,
                    actual_value=metric_value,
                    message=f"{config.metric} is {metric_value} (threshold: {config.threshold})",
                    severity=severity,
                    triggered_at=datetime.now(),
                )
                self.alerts.append(alert)
                alerts_triggered.append(alert)
                logger.warning(f"Alert triggered: {alert.message}")

        return alerts_triggered

    def resolve_alert(self, alert_id: str):
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self) -> list[Alert]:
        return [a for a in self.alerts if not a.resolved]

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        return sorted(self.alerts, key=lambda a: a.triggered_at, reverse=True)[:limit]

    def export_dashboard(self, file_path: str, format: str = "json"):
        report = self.generate_report(format=format)

        if format == "json" or format == "html" or format == "markdown" or format == "csv":
            with open(file_path, "w") as f:
                f.write(report)
        elif format == "pdf":
            return self.generate_pdf_report(file_path)

        logger.info(f"Dashboard exported to {file_path}")
        return file_path

    def get_metrics_by_filter(self, filter_type: str, filter_value: str) -> dict[str, Any]:
        time_filter, _ = self._get_time_filter()

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if filter_type == "model":
                    cursor.execute(
                        f"""
                        SELECT
                            COUNT(*) as total_sessions,
                            COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                            ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT)
                                  / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                            AVG(total_duration_ms) as avg_duration_ms,
                            AVG(total_iterations) as avg_iterations
                        FROM sessions
                        WHERE model = ? AND created_at >= {time_filter}
                        """,
                        (filter_value,),
                    )
                    return dict(cursor.fetchone())

                if filter_type == "tool":
                    cursor.execute(
                        f"""
                        SELECT
                            COUNT(*) as total_calls,
                            COUNT(CASE WHEN success = 1 THEN 1 END) as successful_calls,
                            ROUND(CAST(COUNT(CASE WHEN success = 1 THEN 1 END) AS FLOAT)
                                  / NULLIF(COUNT(*), 0) * 100, 2) as success_rate,
                            AVG(execution_time_ms) as avg_execution_time_ms
                        FROM tool_calls
                        WHERE tool_name = ? AND created_at >= {time_filter}
                        """,
                        (filter_value,),
                    )
                    return dict(cursor.fetchone())

                if filter_type == "status":
                    cursor.execute(
                        f"""
                        SELECT
                            COUNT(*) as total_sessions,
                            AVG(total_duration_ms) as avg_duration_ms,
                            AVG(total_iterations) as avg_iterations,
                            AVG(total_tool_calls) as avg_tool_calls
                        FROM sessions
                        WHERE final_status = ? AND created_at >= {time_filter}
                        """,
                        (filter_value,),
                    )
                    return dict(cursor.fetchone())

                return {"error": f"Unknown filter type: {filter_type}"}
        except Exception as e:
            logger.error(f"Error getting metrics by filter: {e}")
            return {"error": str(e)}

    def get_parallel_execution_stats(self) -> dict[str, Any]:
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_parallel_sessions,
                        AVG(total_duration_ms) as avg_duration_ms,
                        AVG(total_tool_calls) as avg_tool_calls,
                        COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT)
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions
                    WHERE orchestrator_type = 'parallel_executor'
                    """
                )
                parallel_stats = dict(cursor.fetchone())

                cursor.execute(
                    """
                    SELECT
                        AVG(total_duration_ms) as avg_duration_ms,
                        COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_sessions,
                        ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT)
                              / NULLIF(COUNT(*), 0) * 100, 2) as success_rate
                    FROM sessions
                    WHERE orchestrator_type != 'parallel_executor' OR orchestrator_type IS NULL
                    """
                )
                sequential_stats = dict(cursor.fetchone())

                speedup = 0
                if parallel_stats.get("avg_duration_ms") and sequential_stats.get(
                    "avg_duration_ms"
                ):
                    speedup = (
                        sequential_stats["avg_duration_ms"] / parallel_stats["avg_duration_ms"]
                    )

                return {
                    "parallel": parallel_stats,
                    "sequential": sequential_stats,
                    "speedup_factor": round(speedup, 2) if speedup > 0 else 0,
                }
        except Exception as e:
            logger.error(f"Error getting parallel execution stats: {e}")
            return {"error": str(e)}

    def get_forecast(self, metric: str, days: int = 7) -> dict[str, Any]:
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if metric == "success_rate":
                    cursor.execute(
                        """
                        SELECT
                            DATE(created_at) as date,
                            ROUND(CAST(COUNT(CASE WHEN final_status = 'success' THEN 1 END) AS FLOAT)
                                  / NULLIF(COUNT(*), 0) * 100, 2) as value
                        FROM sessions
                        WHERE created_at >= datetime('now', '-30 days')
                        GROUP BY DATE(created_at)
                        ORDER BY date ASC
                        """
                    )
                elif metric == "duration":
                    cursor.execute(
                        """
                        SELECT
                            DATE(created_at) as date,
                            AVG(total_duration_ms) as value
                        FROM sessions
                        WHERE created_at >= datetime('now', '-30 days')
                        GROUP BY DATE(created_at)
                        ORDER BY date ASC
                        """
                    )
                else:
                    return {"error": f"Unknown metric for forecast: {metric}"}

                historical_data = [dict(row) for row in cursor.fetchall()]

                if len(historical_data) < 3:
                    return {"error": "Insufficient data for forecast"}

                values = [d["value"] for d in historical_data if d["value"] is not None]
                if not values:
                    return {"error": "No valid data for forecast"}

                avg_value = sum(values) / len(values)
                trend = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0

                forecast = []
                last_date = datetime.strptime(historical_data[-1]["date"], "%Y-%m-%d")
                for i in range(1, days + 1):
                    next_date = last_date + timedelta(days=i)
                    forecast_value = avg_value + (trend * (len(values) + i))
                    forecast.append(
                        {
                            "date": next_date.strftime("%Y-%m-%d"),
                            "forecast_value": round(forecast_value, 2),
                        }
                    )

                return {
                    "metric": metric,
                    "historical_data": historical_data,
                    "forecast": forecast,
                    "trend": round(trend, 2),
                    "confidence": "low"
                    if len(values) < 7
                    else "medium"
                    if len(values) < 14
                    else "high",
                }
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {"error": str(e)}
