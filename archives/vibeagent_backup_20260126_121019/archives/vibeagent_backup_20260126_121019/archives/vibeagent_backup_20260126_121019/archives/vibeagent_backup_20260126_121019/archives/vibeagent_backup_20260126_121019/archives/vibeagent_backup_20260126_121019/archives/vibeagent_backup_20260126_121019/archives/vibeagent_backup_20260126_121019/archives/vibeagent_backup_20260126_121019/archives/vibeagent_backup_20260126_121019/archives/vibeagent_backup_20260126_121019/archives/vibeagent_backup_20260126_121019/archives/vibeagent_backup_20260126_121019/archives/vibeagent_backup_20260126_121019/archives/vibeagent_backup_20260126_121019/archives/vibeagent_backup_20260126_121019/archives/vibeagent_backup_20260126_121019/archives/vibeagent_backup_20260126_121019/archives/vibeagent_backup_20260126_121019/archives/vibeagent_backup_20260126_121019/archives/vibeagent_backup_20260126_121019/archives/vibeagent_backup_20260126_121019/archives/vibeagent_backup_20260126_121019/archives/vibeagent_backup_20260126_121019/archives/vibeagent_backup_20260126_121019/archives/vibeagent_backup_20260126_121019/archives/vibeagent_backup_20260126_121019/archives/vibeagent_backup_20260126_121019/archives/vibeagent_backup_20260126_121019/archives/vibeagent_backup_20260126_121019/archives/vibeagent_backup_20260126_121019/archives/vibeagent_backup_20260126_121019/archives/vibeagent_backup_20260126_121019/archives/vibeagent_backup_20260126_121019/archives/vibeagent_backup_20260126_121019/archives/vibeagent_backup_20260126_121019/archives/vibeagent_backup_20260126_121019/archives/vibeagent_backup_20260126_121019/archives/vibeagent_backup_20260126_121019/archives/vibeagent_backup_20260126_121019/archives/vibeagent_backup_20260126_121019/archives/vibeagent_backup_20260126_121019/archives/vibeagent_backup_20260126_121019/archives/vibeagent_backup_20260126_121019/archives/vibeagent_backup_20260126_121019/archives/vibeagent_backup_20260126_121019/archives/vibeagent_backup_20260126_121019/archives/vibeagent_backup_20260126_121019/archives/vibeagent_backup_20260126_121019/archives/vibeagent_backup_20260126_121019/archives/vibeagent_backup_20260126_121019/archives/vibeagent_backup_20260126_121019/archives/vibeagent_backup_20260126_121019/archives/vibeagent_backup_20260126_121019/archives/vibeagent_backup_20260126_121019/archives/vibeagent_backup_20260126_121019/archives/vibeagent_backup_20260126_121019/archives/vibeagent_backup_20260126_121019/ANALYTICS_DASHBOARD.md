# Analytics Dashboard

A comprehensive analytics dashboard system for VibeAgent that provides real-time insights, visualizations, and exportable reports.

## Features

### Dashboard Panels

1. **Overview Panel**: Key metrics at a glance
   - Total sessions
   - Success rate
   - Average duration
   - Tool calls
   - Iterations
   - Token usage
   - Unique models and session types

2. **Performance Panel**: Performance trends and analysis
   - Execution time distribution (mean, median, p95, p99)
   - Iteration statistics
   - Daily trends
   - Hourly patterns

3. **Test Results Panel**: Test execution statistics
   - Test summary
   - Failing tests
   - Regression issues
   - Recent test runs

4. **Tool Usage Panel**: Tool performance and usage
   - Tool success rates
   - Failing tools
   - Tool trends over time
   - Tool error breakdown

5. **Model Comparison Panel**: Model performance comparison
   - Model performance metrics
   - Recommendations (speed, quality, efficiency)
   - Token usage trends
   - Response time comparisons

6. **Error Analysis Panel**: Error patterns and recovery
   - Error patterns by type
   - Recovery rates by strategy
   - Error timeline
   - Top errors

7. **Trends Panel**: Historical trends and forecasts
   - Success rate trends
   - Performance degradation detection
   - Success rate drop detection
   - Token usage trends
   - Period comparisons

8. **Insights Panel**: Auto-generated insights
   - Daily insights
   - Weekly reports
   - Optimization suggestions
   - Successful patterns
   - Active alerts

### Metrics Displayed

- **Success Rate**: Overall, by model, by tool
- **Execution Time**: Average, p50, p95, p99
- **Iteration Count**: Average, max, min
- **Token Usage**: Total, average, trend
- **Error Rate**: Overall, by type
- **Recovery Rate**: Overall, by strategy
- **Test Pass Rate**: Overall, by test
- **Parallel Execution Speedup**: Speedup factor

### Time Range Selection

- Last hour
- Last 24 hours
- Last 7 days
- Last 30 days
- All time

### Filtering Options

- Filter by model
- Filter by tool
- Filter by test case
- Filter by status
- Filter by error type

### Report Generation

- `generate_report(format)` - Generate full report
- `generate_summary()` - Generate executive summary
- `generate_detailed_report()` - Generate detailed report
- `generate_comparison_report()` - Generate comparison report

### Export Formats

- **JSON** - Structured data export
- **HTML** - Interactive web report
- **Markdown** - Documentation format
- **CSV** - Spreadsheet format
- **PDF** - Printable report (requires weasyprint)

### Insight Generation

- Auto-generate insights from data
- Identify trends and patterns
- Detect anomalies
- Provide recommendations
- Highlight issues

### Alert System

- Configure alerts for metrics
- Alert on threshold violations
- Alert on trend changes
- Alert on anomalies
- Track alert history

### Database Integration

- Query data from database
- Cache frequent queries (5-minute TTL)
- Optimize query performance
- Handle large datasets

### Web Interface

- Simple web dashboard (Flask-based)
- Interactive charts (Chart.js)
- Real-time updates
- Export buttons
- REST API endpoints

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies for PDF export:
```bash
pip install weasyprint
```

## Usage

### Command Line Example

```python
from core.database_manager import DatabaseManager
from core.analytics_engine import AnalyticsEngine
from core.analytics_dashboard import AnalyticsDashboard

# Initialize
db_manager = DatabaseManager("data/vibeagent.db")
analytics_engine = AnalyticsEngine(db_manager)
dashboard = AnalyticsDashboard(db_manager, analytics_engine)

# Set time range
dashboard.set_time_range("7d")

# Get panels
overview = dashboard.get_overview_panel()
performance = dashboard.get_performance_panel()
test_results = dashboard.get_test_results_panel()
tool_usage = dashboard.get_tool_usage_panel()
model_comparison = dashboard.get_model_comparison_panel()
error_analysis = dashboard.get_error_analysis_panel()
trends = dashboard.get_trends_panel()
insights = dashboard.get_insights_panel()

# Get full dashboard
full_dashboard = dashboard.get_full_dashboard()

# Generate reports
json_report = dashboard.generate_report("json")
html_report = dashboard.generate_report("html")
markdown_report = dashboard.generate_report("markdown")
csv_report = dashboard.generate_report("csv")

# Export to files
dashboard.export_dashboard("reports/dashboard.json", format="json")
dashboard.export_dashboard("reports/dashboard.html", format="html")
dashboard.export_dashboard("reports/dashboard.md", format="markdown")
dashboard.export_dashboard("reports/dashboard.csv", format="csv")
dashboard.export_dashboard("reports/dashboard.pdf", format="pdf")

# Generate summary
summary = dashboard.generate_summary()

# Configure alerts
dashboard.configure_alert("success_rate", threshold=80.0, operator="less_than")
dashboard.configure_alert("avg_duration_ms", threshold=10000.0, operator="greater_than")

# Check alerts
triggered_alerts = dashboard.check_alerts()

# Get forecasts
forecast = dashboard.get_forecast("success_rate", days=7)

# Get parallel execution stats
parallel_stats = dashboard.get_parallel_execution_stats()
```

### Run Example Script

```bash
python examples/analytics_dashboard_example.py
```

### Web Interface

```bash
python examples/dashboard_web.py
```

Then open http://localhost:5000 in your browser.

### API Endpoints

- `GET /` - Main dashboard
- `GET /api/dashboard` - Full dashboard JSON
- `GET /api/overview` - Overview panel
- `GET /api/performance` - Performance panel
- `GET /api/test_results` - Test results panel
- `GET /api/tool_usage` - Tool usage panel
- `GET /api/model_comparison` - Model comparison panel
- `GET /api/error_analysis` - Error analysis panel
- `GET /api/trends` - Trends panel
- `GET /api/insights` - Insights panel
- `GET /export/<format>` - Export report
- `GET /api/forecast/<metric>` - Get forecast
- `POST /api/alerts` - Configure alert
- `GET /api/alerts/check` - Check alerts

## Class Reference

### AnalyticsDashboard

#### Methods

- `set_time_range(time_range)` - Set time range ("1h", "24h", "7d", "30d", "all")
- `set_filters(**filters)` - Set filters
- `clear_filters()` - Clear all filters
- `get_overview_panel()` - Get overview metrics
- `get_performance_panel()` - Get performance metrics
- `get_test_results_panel()` - Get test results
- `get_tool_usage_panel()` - Get tool usage
- `get_model_comparison_panel()` - Get model comparison
- `get_error_analysis_panel()` - Get error analysis
- `get_trends_panel()` - Get trends
- `get_insights_panel()` - Get insights
- `get_full_dashboard()` - Get all panels
- `generate_report(format)` - Generate report
- `generate_summary()` - Generate summary
- `generate_detailed_report()` - Generate detailed report
- `generate_comparison_report()` - Generate comparison report
- `export_dashboard(file_path, format)` - Export to file
- `configure_alert(metric, threshold, operator)` - Configure alert
- `check_alerts()` - Check and trigger alerts
- `resolve_alert(alert_id)` - Resolve alert
- `get_active_alerts()` - Get active alerts
- `get_alert_history(limit)` - Get alert history
- `get_metrics_by_filter(filter_type, filter_value)` - Get filtered metrics
- `get_parallel_execution_stats()` - Get parallel execution stats
- `get_forecast(metric, days)` - Get forecast

## Examples

### Basic Usage

```python
# Initialize dashboard
dashboard = AnalyticsDashboard(db_manager, analytics_engine)

# Get overview metrics
overview = dashboard.get_overview_panel()
print(f"Success Rate: {overview['success_rate']}%")
print(f"Total Sessions: {overview['total_sessions']}")
```

### Generating Reports

```python
# Generate and export reports
dashboard.export_dashboard("reports/dashboard.json", "json")
dashboard.export_dashboard("reports/dashboard.html", "html")
dashboard.export_dashboard("reports/dashboard.md", "markdown")
dashboard.export_dashboard("reports/dashboard.csv", "csv")
```

### Setting Up Alerts

```python
# Configure alerts
dashboard.configure_alert("success_rate", 80.0, "less_than")
dashboard.configure_alert("avg_duration_ms", 10000.0, "greater_than")

# Check alerts
alerts = dashboard.check_alerts()
for alert in alerts:
    print(f"ALERT: {alert.message}")
```

### Filtering Data

```python
# Set time range
dashboard.set_time_range("7d")

# Set filters
dashboard.set_filters(model="gpt-4", status="success")

# Get filtered metrics
metrics = dashboard.get_metrics_by_filter("model", "gpt-4")
```

### Forecasting

```python
# Get success rate forecast
forecast = dashboard.get_forecast("success_rate", days=7)
print(f"Trend: {forecast['trend']}")
print(f"Confidence: {forecast['confidence']}")
for day in forecast['forecast']:
    print(f"{day['date']}: {day['forecast_value']}%")
```

## Architecture

The AnalyticsDashboard integrates with:

- **DatabaseManager**: Queries the SQLite database
- **AnalyticsEngine**: Provides advanced analytics methods
- **Caching**: Implements 5-minute TTL cache for performance
- **Alert System**: Tracks and manages alerts
- **Export System**: Supports multiple export formats
- **Web Interface**: Provides Flask-based web UI

## Performance

- Cached queries (5-minute TTL)
- Optimized SQL queries
- Efficient data aggregation
- Lazy loading of panels
- Batch operations for exports

## License

MIT License