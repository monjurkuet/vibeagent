#!/usr/bin/env python3
"""
Analytics Dashboard - View and analyze LLM capability test results over time.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """Dashboard for analyzing LLM capability test results."""

    def __init__(self, db_path: str = "data/llm_capability_test.db"):
        from core.database_manager import DatabaseManager

        self.db_manager = DatabaseManager(db_path=db_path)
        logger.info(f"✅ Connected to database: {db_path}")

    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall summary of all tests."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Total test runs
            cursor.execute("SELECT COUNT(*) FROM test_runs")
            total_runs = cursor.fetchone()[0]

            # Success rate
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN final_status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM test_runs
            """)
            success_rate = cursor.fetchone()[0] or 0

            # Total test cases
            cursor.execute("SELECT COUNT(*) FROM test_cases")
            total_test_cases = cursor.fetchone()[0]

            # Date range
            cursor.execute("SELECT MIN(started_at), MAX(completed_at) FROM test_runs")
            date_range = cursor.fetchone()

            return {
                "total_runs": total_runs,
                "success_rate": round(success_rate, 2),
                "total_test_cases": total_test_cases,
                "date_range": {
                    "first_run": date_range[0],
                    "last_run": date_range[1],
                },
            }

    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Daily success rate trend
            cursor.execute(
                """
                SELECT 
                    DATE(started_at) as date,
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_runs,
                    ROUND(COUNT(CASE WHEN final_status = 'success' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
                    AVG(total_iterations) as avg_iterations,
                    AVG(total_tool_calls) as avg_tool_calls
                FROM test_runs
                WHERE started_at >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(started_at)
                ORDER BY date DESC
            """,
                (days,),
            )

            daily_trends = [dict(row) for row in cursor.fetchall()]

            # Weekly trend
            cursor.execute(
                """
                SELECT 
                    strftime('%Y-W%W', started_at) as week,
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN final_status = 'success' THEN 1 END) as successful_runs,
                    ROUND(COUNT(CASE WHEN final_status = 'success' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate
                FROM test_runs
                WHERE started_at >= datetime('now', '-' || ? || ' days')
                GROUP BY week
                ORDER BY week DESC
            """,
                (days,),
            )

            weekly_trends = [dict(row) for row in cursor.fetchall()]

            return {
                "daily_trends": daily_trends,
                "weekly_trends": weekly_trends,
            }

    def get_test_case_performance(
        self, test_case_name: str = None
    ) -> List[Dict[str, Any]]:
        """Get performance by test case."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            if test_case_name:
                cursor.execute(
                    """
                    SELECT 
                        tc.name as test_case_name,
                        tc.category,
                        COUNT(tr.id) as total_runs,
                        COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) as successful_runs,
                        ROUND(COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
                        AVG(tr.total_iterations) as avg_iterations,
                        AVG(tr.total_tool_calls) as avg_tool_calls,
                        MAX(tr.completed_at) as last_run
                    FROM test_cases tc
                    LEFT JOIN test_runs tr ON tc.id = tr.test_case_id
                    WHERE tc.name = ?
                    GROUP BY tc.id
                """,
                    (test_case_name,),
                )
            else:
                cursor.execute("""
                    SELECT 
                        tc.name as test_case_name,
                        tc.category,
                        COUNT(tr.id) as total_runs,
                        COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) as successful_runs,
                        ROUND(COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
                        AVG(tr.total_iterations) as avg_iterations,
                        AVG(tr.total_tool_calls) as avg_tool_calls,
                        MAX(tr.completed_at) as last_run
                    FROM test_cases tc
                    LEFT JOIN test_runs tr ON tc.id = tr.test_case_id
                    GROUP BY tc.id
                    ORDER BY success_rate DESC
                """)

            return [dict(row) for row in cursor.fetchall()]

    def get_strategy_comparison(self) -> List[Dict[str, Any]]:
        """Compare performance across strategies (if metadata contains strategy)."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    json_extract(tr.metadata, '$.strategy') as strategy,
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) as successful_runs,
                    ROUND(COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
                    AVG(tr.total_iterations) as avg_iterations,
                    AVG(tr.total_tool_calls) as avg_tool_calls,
                    MAX(tr.completed_at) as last_run
                FROM test_runs tr
                WHERE tr.metadata LIKE '%"strategy"%' 
                GROUP BY strategy
                ORDER BY success_rate DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def get_metric_trends(
        self, metric_name: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get trends for a specific metric over time."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT 
                    DATE(pm.created_at) as date,
                    AVG(pm.metric_value) as avg_value,
                    MIN(pm.metric_value) as min_value,
                    MAX(pm.metric_value) as max_value,
                    COUNT(*) as count
                FROM performance_metrics pm
                JOIN test_runs tr ON tr.session_id = pm.session_id
                WHERE pm.metric_name = ?
                    AND pm.created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(pm.created_at)
                ORDER BY date DESC
            """,
                (metric_name, days),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent test failures."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT 
                    tc.name as test_case_name,
                    tr.id as run_id,
                    tr.run_number,
                    tr.error_message,
                    tr.completed_at,
                    tr.total_iterations,
                    tr.total_tool_calls
                FROM test_runs tr
                JOIN test_cases tc ON tr.test_case_id = tc.id
                WHERE tr.final_status = 'failed' OR tr.status = 'error'
                ORDER BY tr.completed_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def generate_report(self, days: int = 30) -> str:
        """Generate a comprehensive analytics report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 LLM CAPABILITY TEST ANALYTICS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Time Period: Last {days} days")
        report_lines.append("")

        # Overall summary
        report_lines.append("📈 OVERALL SUMMARY")
        report_lines.append("-" * 80)
        summary = self.get_overall_summary()
        report_lines.append(f"Total Test Runs: {summary['total_runs']}")
        report_lines.append(f"Success Rate: {summary['success_rate']}%")
        report_lines.append(f"Total Test Cases: {summary['total_test_cases']}")
        if summary["date_range"]["first_run"]:
            report_lines.append(f"First Run: {summary['date_range']['first_run']}")
            report_lines.append(f"Last Run: {summary['date_range']['last_run']}")
        report_lines.append("")

        # Test case performance
        report_lines.append("📋 TEST CASE PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(
            f"{'Test Case':<30} | {'Runs':<5} | {'Success':<8} | {'Avg Iter':<10} | {'Avg Tools':<10}"
        )
        report_lines.append("-" * 80)

        test_cases = self.get_test_case_performance()
        for tc in test_cases:
            report_lines.append(
                f"{tc['test_case_name']:<30} | {tc['total_runs']:<5} | "
                f"{tc['success_rate']:<7.1f}% | {tc['avg_iterations']:<10.1f} | {tc['avg_tool_calls']:<10.1f}"
            )
        report_lines.append("")

        # Strategy comparison
        report_lines.append("🎯 STRATEGY COMPARISON")
        report_lines.append("-" * 80)
        strategies = self.get_strategy_comparison()
        if strategies:
            report_lines.append(
                f"{'Strategy':<15} | {'Runs':<5} | {'Success':<8} | {'Avg Iter':<10} | {'Avg Tools':<10}"
            )
            report_lines.append("-" * 80)
            for s in strategies:
                strategy = s["strategy"] if s["strategy"] else "unknown"
                report_lines.append(
                    f"{strategy:<15} | {s['total_runs']:<5} | "
                    f"{s['success_rate']:<7.1f}% | {s['avg_iterations']:<10.1f} | {s['avg_tool_calls']:<10.1f}"
                )
        else:
            report_lines.append("No strategy data available (metadata not stored)")
        report_lines.append("")

        # Performance trends
        report_lines.append(f"📅 DAILY PERFORMANCE TREND (Last {days} days)")
        report_lines.append("-" * 80)
        trends = self.get_performance_trends(days)
        daily = trends["daily_trends"]
        if daily:
            report_lines.append(
                f"{'Date':<12} | {'Runs':<5} | {'Success':<8} | {'Avg Iter':<10} | {'Avg Tools':<10}"
            )
            report_lines.append("-" * 80)
            for d in daily[:10]:  # Show last 10 days
                report_lines.append(
                    f"{str(d['date']):<12} | {d['total_runs']:<5} | "
                    f"{d['success_rate']:<7.1f}% | {d['avg_iterations']:<10.1f} | {d['avg_tool_calls']:<10.1f}"
                )
        else:
            report_lines.append("No trend data available")
        report_lines.append("")

        # Metric trends
        report_lines.append("📊 METRIC TRENDS")
        report_lines.append("-" * 80)

        # Response time trend
        response_time_trend = self.get_metric_trends("response_time", days)
        if response_time_trend:
            report_lines.append("Response Time (ms):")
            for m in response_time_trend[:5]:
                report_lines.append(
                    f"  {m['date']}: avg={m['avg_value']:.2f}ms, min={m['min_value']:.2f}ms, max={m['max_value']:.2f}ms"
                )
            report_lines.append("")

        # Reasoning steps trend
        reasoning_trend = self.get_metric_trends("reasoning_steps", days)
        if reasoning_trend:
            report_lines.append("Reasoning Steps:")
            for m in reasoning_trend[:5]:
                report_lines.append(f"  {m['date']}: avg={m['avg_value']:.2f} steps")
            report_lines.append("")

        # Recent failures
        report_lines.append("❌ RECENT FAILURES")
        report_lines.append("-" * 80)
        failures = self.get_recent_failures(5)
        if failures:
            for f in failures:
                report_lines.append(f"  {f['test_case_name']} (Run #{f['run_id']})")
                if f["error_message"]:
                    report_lines.append(f"    Error: {f['error_message'][:100]}...")
        else:
            report_lines.append("  No recent failures!")
        report_lines.append("")

        return "\n".join(report_lines)

    def export_to_json(self, output_file: str = None) -> str:
        """Export all analytics data to JSON."""
        if output_file is None:
            output_file = (
                f"llm_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        data = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_overall_summary(),
            "test_cases": self.get_test_case_performance(),
            "strategies": self.get_strategy_comparison(),
            "trends": self.get_performance_trends(30),
            "metrics": {
                "response_time": self.get_metric_trends("response_time", 30),
                "reasoning_steps": self.get_metric_trends("reasoning_steps", 30),
            },
            "recent_failures": self.get_recent_failures(10),
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"💾 Exported analytics to: {output_file}")
        return output_file


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="View LLM capability test analytics")
    parser.add_argument(
        "--db", default="data/llm_capability_test.db", help="Database path"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze"
    )
    parser.add_argument("--report", action="store_true", help="Generate text report")
    parser.add_argument("--export", help="Export to JSON file")
    parser.add_argument("--test-case", help="Show specific test case performance")

    args = parser.parse_args()

    dashboard = AnalyticsDashboard(db_path=args.db)

    if args.report:
        print(dashboard.generate_report(days=args.days))

    if args.export:
        dashboard.export_to_json(args.export)

    if args.test_case:
        print(f"\n📋 Test Case: {args.test_case}")
        print("-" * 80)
        results = dashboard.get_test_case_performance(args.test_case)
        for r in results:
            print(f"  Total Runs: {r['total_runs']}")
            print(f"  Success Rate: {r['success_rate']}%")
            print(f"  Avg Iterations: {r['avg_iterations']}")
            print(f"  Avg Tool Calls: {r['avg_tool_calls']}")
            print(f"  Last Run: {r['last_run']}")

    # Default: show summary
    if not args.report and not args.export and not args.test_case:
        summary = dashboard.get_overall_summary()
        print("\n📊 Quick Summary:")
        print(f"  Total Runs: {summary['total_runs']}")
        print(f"  Success Rate: {summary['success_rate']}%")
        print(f"  Test Cases: {summary['total_test_cases']}")
        print("\nUse --report for full analysis, --export to save data")


if __name__ == "__main__":
    main()
