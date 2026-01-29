#!/usr/bin/env python3
"""
Example usage of the Analytics Dashboard.

This script demonstrates how to use the AnalyticsDashboard class
to generate comprehensive analytics reports with visualizations.
"""

import logging
from pathlib import Path

from core.analytics_dashboard import AnalyticsDashboard
from core.analytics_engine import AnalyticsEngine
from core.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    db_path = Path("data/vibeagent.db")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        logger.info("Please run some tests first to generate data")
        return

    db_manager = DatabaseManager(str(db_path))
    analytics_engine = AnalyticsEngine(db_manager)
    dashboard = AnalyticsDashboard(db_manager, analytics_engine)

    logger.info("Generating Analytics Dashboard...")
    print("\n" + "=" * 60)
    print("VIBEAGENT ANALYTICS DASHBOARD")
    print("=" * 60 + "\n")

    dashboard.set_time_range("7d")

    overview = dashboard.get_overview_panel()
    if "error" not in overview:
        print("📊 OVERVIEW PANEL")
        print("-" * 40)
        print(f"  Total Sessions:       {overview.get('total_sessions', 0):,}")
        print(f"  Success Rate:         {overview.get('success_rate', 0):.2f}%")
        print(f"  Avg Duration:         {overview.get('avg_duration_ms', 0):.2f}ms")
        print(f"  Total Tool Calls:     {overview.get('total_tool_calls', 0):,}")
        print(f"  Avg Iterations:       {overview.get('avg_iterations', 0):.2f}")
        print(f"  Total Tokens:         {overview.get('total_tokens', 0):,}")
        print(f"  Unique Models:        {overview.get('unique_models', 0)}")
        print()

    performance = dashboard.get_performance_panel()
    if "error" not in performance:
        print("⚡ PERFORMANCE PANEL")
        print("-" * 40)
        exec_times = performance.get("execution_times", {})
        if "error" not in exec_times:
            print(f"  Mean Execution Time:  {exec_times.get('mean_ms', 0):.2f}ms")
            print(f"  Median Execution Time: {exec_times.get('median_ms', 0):.2f}ms")
            print(f"  P95 Execution Time:    {exec_times.get('p95_ms', 0):.2f}ms")
            print(f"  P99 Execution Time:    {exec_times.get('p99_ms', 0):.2f}ms")

        iter_stats = performance.get("iteration_statistics", {})
        if "error" not in iter_stats:
            print(f"  Avg Iterations:       {iter_stats.get('mean_iterations', 0):.2f}")
            print(f"  Max Iterations:       {iter_stats.get('max_iterations', 0)}")
        print()

    test_results = dashboard.get_test_results_panel()
    if "error" not in test_results:
        print("🧪 TEST RESULTS PANEL")
        print("-" * 40)
        summary = test_results.get("summary", {})
        if summary:
            print(f"  Total Test Cases:     {summary.get('total_test_cases', 0)}")
            print(f"  Executed Tests:       {summary.get('executed_test_cases', 0)}")
            print(f"  Total Runs:           {summary.get('total_runs', 0)}")
            print(f"  Avg Success Rate:     {summary.get('avg_success_rate', 0):.2f}%")

        failing = test_results.get("failing_tests", [])
        if failing:
            print(f"  Failing Tests:        {len(failing)}")
        print()

    tool_usage = dashboard.get_tool_usage_panel()
    if "error" not in tool_usage:
        print("🔧 TOOL USAGE PANEL")
        print("-" * 40)
        tools = tool_usage.get("tools", [])
        if tools:
            print(f"  Total Tools:          {len(tools)}")
            print("\n  Top 5 Tools:")
            for tool in tools[:5]:
                print(
                    f"    - {tool.get('tool_name'):20s} {tool.get('success_rate', 0):6.2f}% ({tool.get('total_calls', 0)} calls)"
                )

        failing_tools = tool_usage.get("failing_tools", [])
        if failing_tools:
            print(f"\n  ⚠️  Failing Tools: {len(failing_tools)}")
            for tool in failing_tools[:3]:
                print(f"    - {tool.get('tool_name'):20s} {tool.get('failure_rate', 0):6.2f}%")
        print()

    model_comparison = dashboard.get_model_comparison_panel()
    if "error" not in model_comparison:
        print("🤖 MODEL COMPARISON PANEL")
        print("-" * 40)
        models = model_comparison.get("models", [])
        if models:
            print(f"  Total Models:         {len(models)}")
            print("\n  Model Performance:")
            for model in models:
                print(
                    f"    - {model.get('model'):20s} {model.get('success_rate', 0):6.2f}% ({model.get('total_requests', 0)} requests)"
                )

        recommendations = model_comparison.get("recommendations", {})
        if recommendations:
            print(
                f"\n  Best for Speed:        {recommendations.get('best_for_speed', {}).get('model', 'N/A')}"
            )
            print(
                f"  Best for Quality:      {recommendations.get('best_for_quality', {}).get('model', 'N/A')}"
            )
            print(
                f"  Best for Efficiency:   {recommendations.get('best_for_efficiency', {}).get('model', 'N/A')}"
            )
        print()

    error_analysis = dashboard.get_error_analysis_panel()
    if "error" not in error_analysis:
        print("❌ ERROR ANALYSIS PANEL")
        print("-" * 40)
        patterns = error_analysis.get("patterns", {})
        error_types = patterns.get("by_error_type", [])
        if error_types:
            print("  Top Error Types:")
            for error in error_types[:5]:
                print(f"    - {error.get('error_type'):30s} {error.get('count', 0):,}")
        print()

    trends = dashboard.get_trends_panel()
    if "error" not in trends:
        print("📈 TRENDS PANEL")
        print("-" * 40)
        degradation = trends.get("degradation", {})
        if degradation.get("degradation_detected"):
            print("  ⚠️  Performance degradation detected!")
            for insight in degradation.get("insights", []):
                print(f"    - {insight.get('metric')}: {insight.get('trend_percent', 0):.2f}%")
        else:
            print("  ✅ No performance degradation detected")

        success_drop = trends.get("success_rate_drop", {})
        if success_drop.get("drop_detected"):
            print(f"  ⚠️  Success rate drop: {success_drop.get('drop_percent', 0):.2f}%")
        else:
            print("  ✅ Success rate stable")
        print()

    insights = dashboard.get_insights_panel()
    if "error" not in insights:
        print("💡 INSIGHTS PANEL")
        print("-" * 40)
        suggestions = insights.get("optimization_suggestions", {}).get("suggestions", [])
        if suggestions:
            print("  Top Recommendations:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                priority = suggestion.get("priority", "low")
                emoji = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                print(f"    {emoji} {i}. {suggestion.get('suggestion', '')}")

        active_alerts = insights.get("alerts", [])
        if active_alerts:
            print(f"\n  🚨 Active Alerts: {len(active_alerts)}")
        else:
            print("\n  ✅ No active alerts")
        print()

    parallel_stats = dashboard.get_parallel_execution_stats()
    if "error" not in parallel_stats:
        print("🚀 PARALLEL EXECUTION STATS")
        print("-" * 40)
        print(f"  Speedup Factor:       {parallel_stats.get('speedup_factor', 0):.2f}x")
        print()

    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60 + "\n")

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    logger.info("Generating JSON report...")
    dashboard.export_dashboard(str(output_dir / "dashboard.json"), format="json")
    print(f"  ✅ JSON report: {output_dir / 'dashboard.json'}")

    logger.info("Generating HTML report...")
    dashboard.export_dashboard(str(output_dir / "dashboard.html"), format="html")
    print(f"  ✅ HTML report: {output_dir / 'dashboard.html'}")

    logger.info("Generating Markdown report...")
    dashboard.export_dashboard(str(output_dir / "dashboard.md"), format="markdown")
    print(f"  ✅ Markdown report: {output_dir / 'dashboard.md'}")

    logger.info("Generating CSV report...")
    dashboard.export_dashboard(str(output_dir / "dashboard.csv"), format="csv")
    print(f"  ✅ CSV report: {output_dir / 'dashboard.csv'}")

    logger.info("Generating summary...")
    summary = dashboard.generate_summary()
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        import json

        json.dump(summary, f, indent=2, default=str)
    print(f"  ✅ Summary: {summary_path}")

    logger.info("Generating detailed report...")
    detailed = dashboard.generate_detailed_report()
    detailed_path = output_dir / "detailed.json"
    with open(detailed_path, "w") as f:
        import json

        json.dump(detailed, f, indent=2, default=str)
    print(f"  ✅ Detailed report: {detailed_path}")

    logger.info("Generating comparison report...")
    comparison = dashboard.generate_comparison_report()
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        import json

        json.dump(comparison, f, indent=2, default=str)
    print(f"  ✅ Comparison report: {comparison_path}")

    print("\n" + "=" * 60)
    print("DEMONSTRATING ALERT SYSTEM")
    print("=" * 60 + "\n")

    dashboard.configure_alert("success_rate", threshold=80.0, operator="less_than")
    dashboard.configure_alert("avg_duration_ms", threshold=10000.0, operator="greater_than")
    print("  ✅ Alerts configured:")
    print("    - Success rate < 80%")
    print("    - Avg duration > 10000ms")

    logger.info("Checking alerts...")
    triggered_alerts = dashboard.check_alerts()
    if triggered_alerts:
        print(f"\n  🚨 Triggered Alerts: {len(triggered_alerts)}")
        for alert in triggered_alerts:
            print(f"    - [{alert.severity.upper()}] {alert.message}")
    else:
        print("\n  ✅ No alerts triggered")

    print("\n" + "=" * 60)
    print("GENERATING FORECAST")
    print("=" * 60 + "\n")

    logger.info("Generating success rate forecast...")
    forecast = dashboard.get_forecast("success_rate", days=7)
    if "error" not in forecast:
        print(f"  Metric:               {forecast.get('metric')}")
        print(f"  Trend:                {forecast.get('trend', 0):.2f}")
        print(f"  Confidence:           {forecast.get('confidence')}")
        print("\n  7-Day Forecast:")
        for day in forecast.get("forecast", []):
            print(f"    - {day.get('date')}: {day.get('forecast_value', 0):.2f}%")

    print("\n" + "=" * 60)
    print("DASHBOARD GENERATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
