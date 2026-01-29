#!/usr/bin/env python3
"""
Simple web interface for the Analytics Dashboard.

This provides a Flask-based web UI for viewing analytics data.
"""

from pathlib import Path

from core.analytics_dashboard import AnalyticsDashboard
from core.analytics_engine import AnalyticsEngine
from core.database_manager import DatabaseManager
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

db_path = Path("data/vibeagent.db")
db_manager = DatabaseManager(str(db_path))
analytics_engine = AnalyticsEngine(db_manager)
dashboard = AnalyticsDashboard(db_manager, analytics_engine)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VibeAgent Analytics Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }
        .header h1 { margin-bottom: 5px; }
        .header .meta { font-size: 14px; opacity: 0.9; }
        .container { max-width: 1400px; margin: 20px auto; padding: 0 20px; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 15px; align-items: center; }
        .controls select, .controls button { padding: 8px 15px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
        .controls button { background: #667eea; color: white; border: none; }
        .controls button:hover { background: #5568d3; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .panel h2 { color: #333; font-size: 18px; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 4px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #667eea; }
        .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 4px; font-size: 14px; }
        .alert-high { background: #fee; border-left: 4px solid #f44; }
        .alert-medium { background: #ffd; border-left: 4px solid #fb4; }
        .alert-low { background: #efe; border-left: 4px solid #4f4; }
        .loading { text-align: center; padding: 40px; color: #666; }
        .error { background: #fee; color: #c33; padding: 15px; border-radius: 4px; margin: 20px 0; }
        .chart-container { position: relative; height: 250px; }
        .full-width { grid-column: 1 / -1; }
        .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .status-success { background: #d4edda; color: #155724; }
        .status-failure { background: #f8d7da; color: #721c24; }
        .status-warning { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 VibeAgent Analytics Dashboard</h1>
        <div class="meta">Generated: {{ generated_at }} | Time Range: {{ time_range }}</div>
    </div>

    <div class="container">
        <div class="controls">
            <select id="timeRange" onchange="changeTimeRange()">
                <option value="1h" {% if time_range == '1h' %}selected{% endif %}>Last Hour</option>
                <option value="24h" {% if time_range == '24h' %}selected{% endif %}>Last 24 Hours</option>
                <option value="7d" {% if time_range == '7d' %}selected{% endif %}>Last 7 Days</option>
                <option value="30d" {% if time_range == '30d' %}selected{% endif %}>Last 30 Days</option>
                <option value="all" {% if time_range == 'all' %}selected{% endif %}>All Time</option>
            </select>
            <button onclick="refreshDashboard()">🔄 Refresh</button>
            <button onclick="exportReport('json')">📥 Export JSON</button>
            <button onclick="exportReport('html')">📄 Export HTML</button>
        </div>

        {% if error %}
        <div class="error">⚠️ {{ error }}</div>
        {% endif %}

        <div class="grid">
            <div class="panel">
                <h2>📊 Overview</h2>
                <div class="grid" style="grid-template-columns: repeat(2, 1fr); gap: 10px;">
                    <div class="metric">
                        <div class="metric-value">{{ overview.total_sessions or 0 }}</div>
                        <div class="metric-label">Total Sessions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ "%.2f"|format(overview.success_rate or 0) }}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ "%.0f"|format(overview.avg_duration_ms or 0) }}ms</div>
                        <div class="metric-label">Avg Duration</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ overview.total_tool_calls or 0 }}</div>
                        <div class="metric-label">Tool Calls</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <h2>⚡ Performance</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Mean Time</td><td>{{ "%.2f"|format(performance.execution_times.mean_ms or 0) }}ms</td></tr>
                    <tr><td>Median Time</td><td>{{ "%.2f"|format(performance.execution_times.median_ms or 0) }}ms</td></tr>
                    <tr><td>P95 Time</td><td>{{ "%.2f"|format(performance.execution_times.p95_ms or 0) }}ms</td></tr>
                    <tr><td>P99 Time</td><td>{{ "%.2f"|format(performance.execution_times.p99_ms or 0) }}ms</td></tr>
                    <tr><td>Avg Iterations</td><td>{{ "%.2f"|format(performance.iteration_stats.mean_iterations or 0) }}</td></tr>
                </table>
            </div>

            <div class="panel">
                <h2>🧪 Test Results</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Tests</td><td>{{ test_results.summary.total_test_cases or 0 }}</td></tr>
                    <tr><td>Executed</td><td>{{ test_results.summary.executed_test_cases or 0 }}</td></tr>
                    <tr><td>Total Runs</td><td>{{ test_results.summary.total_runs or 0 }}</td></tr>
                    <tr><td>Avg Pass Rate</td><td>{{ "%.2f"|format(test_results.summary.avg_success_rate or 0) }}%</td></tr>
                    <tr><td>Failing Tests</td><td>{{ test_results.failing_tests|length }}</td></tr>
                </table>
            </div>

            <div class="panel full-width">
                <h2>🔧 Tool Usage</h2>
                <div class="chart-container">
                    <canvas id="toolChart"></canvas>
                </div>
            </div>

            <div class="panel full-width">
                <h2>🤖 Model Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Success Rate</th>
                            <th>Total Requests</th>
                            <th>Avg Response Time</th>
                            <th>Token Efficiency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for model in model_comparison.models %}
                        <tr>
                            <td>{{ model.model }}</td>
                            <td>{{ "%.2f"|format(model.success_rate or 0) }}%</td>
                            <td>{{ model.total_requests or 0 }}</td>
                            <td>{{ "%.0f"|format(model.avg_response_time_ms or 0) }}ms</td>
                            <td>{{ "%.2f"|format(model.avg_token_efficiency or 0) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="panel full-width">
                <h2>📈 Success Rate Trend</h2>
                <div class="chart-container">
                    <canvas id="successTrendChart"></canvas>
                </div>
            </div>

            <div class="panel">
                <h2>❌ Top Errors</h2>
                <table>
                    <thead>
                        <tr><th>Error Type</th><th>Count</th></tr>
                    </thead>
                    <tbody>
                        {% for error in error_analysis.patterns.by_error_type[:10] %}
                        <tr>
                            <td>{{ error.error_type }}</td>
                            <td>{{ error.count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="panel">
                <h2>🚀 Parallel Execution</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Speedup Factor</td><td>{{ "%.2fx"|format(parallel_stats.speedup_factor or 0) }}</td></tr>
                    <tr><td>Parallel Sessions</td><td>{{ parallel_stats.parallel.total_parallel_sessions or 0 }}</td></tr>
                    <tr><td>Parallel Success Rate</td><td>{{ "%.2f"|format(parallel_stats.parallel.success_rate or 0) }}%</td></tr>
                </table>
            </div>

            <div class="panel full-width">
                <h2>💡 Insights & Recommendations</h2>
                {% for suggestion in insights.optimization_suggestions.suggestions[:10] %}
                <div class="alert alert-{{ suggestion.priority }}">
                    <strong>[{{ suggestion.priority|upper }}] {{ suggestion.category }}:</strong> {{ suggestion.suggestion }}
                </div>
                {% endfor %}
            </div>

            <div class="panel full-width">
                <h2>🚨 Active Alerts</h2>
                {% if insights.alerts %}
                    {% for alert in insights.alerts %}
                    <div class="alert alert-{{ alert.severity }}">
                        <strong>{{ alert.metric }}</strong>: {{ alert.message }} (Triggered: {{ alert.triggered_at }})
                    </div>
                    {% endfor %}
                {% else %}
                    <p style="color: #4a4; text-align: center; padding: 20px;">✅ No active alerts</p>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        function changeTimeRange() {
            const range = document.getElementById('timeRange').value;
            window.location.href = '/?time_range=' + range;
        }

        function refreshDashboard() {
            window.location.reload();
        }

        function exportReport(format) {
            window.location.href = '/export/' + format;
        }

        // Tool Usage Chart
        const toolCtx = document.getElementById('toolChart').getContext('2d');
        new Chart(toolCtx, {
            type: 'bar',
            data: {
                labels: {{ tool_usage.tools|map(attribute='tool_name')|list|tojson|safe }},
                datasets: [{
                    label: 'Success Rate (%)',
                    data: {{ tool_usage.tools|map(attribute='success_rate')|list|tojson|safe }},
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });

        // Success Rate Trend Chart
        const trendCtx = document.getElementById('successTrendChart').getContext('2d');
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: {{ trends.success_rate_trend.data|map(attribute='date')|reverse|list|tojson|safe }},
                datasets: [{
                    label: 'Success Rate (%)',
                    data: {{ trends.success_rate_trend.data|map(attribute='success_rate')|reverse|list|tojson|safe }},
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    time_range = request.args.get("time_range", "7d")
    dashboard.set_time_range(time_range)

    try:
        data = {
            "generated_at": dashboard.get_full_dashboard()["generated_at"],
            "time_range": dashboard.config.time_range,
            "overview": dashboard.get_overview_panel(),
            "performance": dashboard.get_performance_panel(),
            "test_results": dashboard.get_test_results_panel(),
            "tool_usage": dashboard.get_tool_usage_panel(),
            "model_comparison": dashboard.get_model_comparison_panel(),
            "error_analysis": dashboard.get_error_analysis_panel(),
            "trends": dashboard.get_trends_panel(),
            "insights": dashboard.get_insights_panel(),
            "parallel_stats": dashboard.get_parallel_execution_stats(),
            "error": None,
        }
    except Exception as e:
        data = {"error": str(e)}

    return render_template_string(HTML_TEMPLATE, **data)


@app.route("/api/dashboard")
def api_dashboard():
    time_range = request.args.get("time_range", "7d")
    dashboard.set_time_range(time_range)
    return jsonify(dashboard.get_full_dashboard())


@app.route("/api/overview")
def api_overview():
    return jsonify(dashboard.get_overview_panel())


@app.route("/api/performance")
def api_performance():
    return jsonify(dashboard.get_performance_panel())


@app.route("/api/test_results")
def api_test_results():
    return jsonify(dashboard.get_test_results_panel())


@app.route("/api/tool_usage")
def api_tool_usage():
    return jsonify(dashboard.get_tool_usage_panel())


@app.route("/api/model_comparison")
def api_model_comparison():
    return jsonify(dashboard.get_model_comparison_panel())


@app.route("/api/error_analysis")
def api_error_analysis():
    return jsonify(dashboard.get_error_analysis_panel())


@app.route("/api/trends")
def api_trends():
    return jsonify(dashboard.get_trends_panel())


@app.route("/api/insights")
def api_insights():
    return jsonify(dashboard.get_insights_panel())


@app.route("/export/<format>")
def export_report(format):
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    filename = f"dashboard.{format}"
    if format == "markdown":
        filename = "dashboard.md"

    file_path = output_dir / filename
    dashboard.export_dashboard(str(file_path), format=format)

    return jsonify({"status": "success", "file": str(file_path)})


@app.route("/api/forecast/<metric>")
def api_forecast(metric):
    days = request.args.get("days", 7, type=int)
    return jsonify(dashboard.get_forecast(metric, days))


@app.route("/api/alerts", methods=["POST"])
def api_configure_alert():
    data = request.json
    dashboard.configure_alert(
        metric=data.get("metric"),
        threshold=data.get("threshold"),
        operator=data.get("operator", "greater_than"),
    )
    return jsonify({"status": "success"})


@app.route("/api/alerts/check")
def api_check_alerts():
    alerts = dashboard.check_alerts()
    return jsonify({"alerts": [a.__dict__ for a in alerts]})


if __name__ == "__main__":
    print("🚀 Starting VibeAgent Analytics Dashboard...")
    print("📊 Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
