# 📊 Complete LLM Capability Testing & Analytics System

## Overview

You now have a **complete system** to:
1. ✅ Test LLM capabilities across 4 orchestration strategies
2. ✅ Store all results in a database with full metadata
3. ✅ Analyze performance trends over time
4. ✅ Compare strategies and test cases
5. ✅ Export data for further analysis

---

## 🧪 Test Suite: `test_llm_capabilities.py`

### What It Tests

**4 Orchestration Strategies:**
- **Basic** - Simple tool orchestration
- **ReAct** - Reasoning + Acting with explicit thoughts
- **Plan-and-Execute** - Plan first, then execute
- **Tree-of-Thought (ToT)** - Explore multiple reasoning paths

**8 Test Cases:**
| Test | Complexity | Tools | Parallel | Description |
|------|------------|-------|----------|-------------|
| Simple Single Tool | Low | 1 | No | Basic tool calling |
| Sequential Chaining | Medium | 2 | No | Multi-step task chaining |
| Parallel Independent | Medium | 2 | Yes | Simultaneous independent tasks |
| Complex Multi-Step | High | 3 | No | Multiple tools and steps |
| Tool Selection | Medium | 1 | No | Choose right tool |
| Error Recovery | High | 2 | No | Handle failures |
| Context-Aware | High | 3 | No | Maintain context |
| Decision Making | High | 2 | No | Decide based on results |

### Metrics Tracked Per Test

✅ **Execution Metrics:**
- Success/Failure status
- Number of iterations
- Number of tool calls
- Response time (ms)
- Parallel execution used (yes/no)

✅ **Strategy Metrics:**
- Orchestrator type used
- Strategy name stored in metadata

✅ **Test Metadata:**
- Expected tools count
- Expected parallel usage
- Test complexity level
- Individual skill call counts

### Database Schema Used

The test uses **20 database tables** to store:

1. **test_cases** - Test case definitions
2. **test_runs** - Individual test executions
3. **performance_metrics** - All metrics (response_time, reasoning_steps, iterations, tool_calls, parallel_used)
4. **sessions** - Orchestrator sessions
5. **messages** - LLM conversation history
6. **llm_responses** - LLM API responses
7. **tool_calls** - Tool execution records
8. **tool_results** - Tool execution results
9. **reasoning_steps** - ReAct/ToT reasoning
10. **error_recovery** - Error handling attempts
11. **self_corrections** - Self-correction events
12. **judge_evaluations** - LLM judge evaluations
13. Plus more...

### How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests with all strategies
python test_llm_capabilities.py

# Custom LLM URL
python test_llm_capabilities.py --llm-url http://localhost:8087/v1

# Specific model
python test_llm_capabilities.py --llm-model glm-4.7

# Only test specific strategies
python test_llm_capabilities.py --strategies basic react
```

### Output Files

1. **`llm_capability_test.log`** - Detailed execution log
2. **`llm_capability_report.json`** - Machine-readable results
3. **`data/llm_capability_test.db`** - Database with all test runs

---

## 📈 Analytics Dashboard: `analytics_dashboard.py`

### What It Does

**View & Analyze:**
- Overall test summary
- Performance trends (daily/weekly)
- Test case performance comparison
- Strategy comparison
- Metric trends over time
- Recent failures
- Export to JSON

### How to Use

```bash
# Quick summary
python analytics_dashboard.py

# Full report (last 30 days)
python analytics_dashboard.py --report

# Custom time period
python analytics_dashboard.py --report --days 60

# Export to JSON
python analytics_dashboard.py --export analytics_export.json

# Specific test case
python analytics_dashboard.py --test-case "Simple Single Tool"

# Custom database
python analytics_dashboard.py --db data/llm_capability_test.db
```

### Sample Report Output

```
================================================================================
📊 LLM CAPABILITY TEST ANALYTICS REPORT
================================================================================
Generated: 2026-01-24 06:30:00
Time Period: Last 30 days

📈 OVERALL SUMMARY
--------------------------------------------------------------------------------
Total Test Runs: 128
Success Rate: 87.5%
Total Test Cases: 8
First Run: 2026-01-01 10:00:00
Last Run: 2026-01-24 15:30:00

📋 TEST CASE PERFORMANCE
--------------------------------------------------------------------------------
Test Case                      | Runs  | Success | Avg Iter | Avg Tools
--------------------------------------------------------------------------------
Simple Single Tool             | 16    | 100.0%  | 2.0      | 1.0
Sequential Chaining            | 16    | 93.8%   | 2.5      | 2.0
Parallel Independent           | 16    | 87.5%   | 2.0      | 2.0
Complex Multi-Step             | 16    | 81.2%   | 3.8      | 2.8
...

🎯 STRATEGY COMPARISON
--------------------------------------------------------------------------------
Strategy        | Runs  | Success | Avg Iter | Avg Tools
--------------------------------------------------------------------------------
basic           | 32    | 87.5%   | 2.3      | 1.5
react           | 32    | 93.8%   | 2.1      | 1.8
plan_execute    | 32    | 81.2%   | 3.5      | 2.2
tot             | 32    | 87.5%   | 4.2      | 2.5

📅 DAILY PERFORMANCE TREND (Last 30 days)
--------------------------------------------------------------------------------
Date         | Runs  | Success | Avg Iter | Avg Tools
--------------------------------------------------------------------------------
2026-01-24   | 8     | 100.0%  | 2.1      | 1.8
2026-01-23   | 8     | 87.5%   | 2.5      | 2.0
...

📊 METRIC TRENDS
--------------------------------------------------------------------------------
Response Time (ms):
  2026-01-24: avg=1456.78ms, min=1234.56ms, max=2345.67ms
  2026-01-23: avg=1523.45ms, min=1345.67ms, max=2456.78ms
  ...

Reasoning Steps:
  2026-01-24: avg=1.5 steps
  2026-01-23: avg=1.3 steps
  ...

❌ RECENT FAILURES
--------------------------------------------------------------------------------
  Complex Multi-Step (Run #156)
    Error: Timeout waiting for LLM response
  Decision Making (Run #155)
    Error: Tool execution failed: database save error
```

---

## 🗄️ Database Queries Available

The database has **pre-built views** for common queries:

### 1. Test Performance View
```sql
SELECT * FROM test_performance;
```
Shows:
- Total runs per test case
- Success rate
- Last run date

### 2. Tool Success Rate View
```sql
SELECT * FROM tool_success_rate;
```
Shows:
- Total calls per tool
- Success rate per tool
- Average execution time

### 3. Model Comparison View
```sql
SELECT * FROM model_comparison;
```
Shows:
- Total requests per model
- Token usage
- Average response time
- Token efficiency

### Custom Queries You Can Run

```sql
-- Strategy comparison over time
SELECT 
    json_extract(tr.metadata, '$.strategy') as strategy,
    DATE(tr.started_at) as date,
    COUNT(*) as runs,
    AVG(tr.total_iterations) as avg_iterations,
    AVG(tr.total_tool_calls) as avg_tools
FROM test_runs tr
WHERE tr.metadata LIKE '%"strategy"%'
GROUP BY strategy, date
ORDER BY date DESC;

-- Performance trend by complexity
SELECT 
    json_extract(tc.metadata, '$.complexity') as complexity,
    COUNT(tr.id) as runs,
    COUNT(CASE WHEN tr.final_status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
FROM test_runs tr
JOIN test_cases tc ON tr.test_case_id = tc.id
GROUP BY complexity;

-- Metric trends
SELECT 
    DATE(pm.created_at) as date,
    pm.metric_name,
    AVG(pm.metric_value) as avg_value,
    MIN(pm.metric_value) as min_value,
    MAX(pm.metric_value) as max_value
FROM performance_metrics pm
GROUP BY date, pm.metric_name
ORDER BY date DESC;
```

---

## 📊 What You Can Track Over Time

### 1. **LLM Model Performance**
- Success rate trends
- Response time trends
- Tool calling accuracy
- Reasoning quality

### 2. **Strategy Comparison**
- Which strategy works best for your LLM
- Performance by complexity level
- Resource usage (iterations, tool calls)

### 3. **Test Case Performance**
- Which test cases are most challenging
- Failure patterns
- Improvement over time

### 4. **Custom Metrics**
- Any metric you add to the database
- Trends over days/weeks/months
- Compare across models/strategies

---

## 🔍 Example Use Cases

### Use Case 1: Compare LLM Models

```bash
# Test with model A
python test_llm_capabilities.py --llm-model gpt-4

# Test with model B
python test_llm_capabilities.py --llm-model glm-4.7

# Compare results
python analytics_dashboard.py --report
```

### Use Case 2: Track Improvement Over Time

```bash
# Run tests weekly
python test_llm_capabilities.py

# View trends
python analytics_dashboard.py --report --days 90

# Export for plotting
python analytics_dashboard.py --export trends.json
```

### Use Case 3: Find Optimal Strategy

```bash
# Test all strategies
python test_llm_capabilities.py --strategies basic react plan_execute tot

# Compare strategy performance
python analytics_dashboard.py --report | grep "STRATEGY COMPARISON"
```

### Use Case 4: Debug Failures

```bash
# View recent failures
python analytics_dashboard.py --report | grep -A 10 "RECENT FAILURES"

# Query database directly
sqlite3 data/llm_capability_test.db "SELECT * FROM test_runs WHERE final_status = 'failed' ORDER BY completed_at DESC LIMIT 10;"
```

---

## 📦 Summary

You now have a **complete testing and analytics system**:

✅ **4 orchestration strategies** tested with **8 test cases**
✅ **All results stored** in 20 database tables
✅ **Full metadata tracking** (strategy, complexity, metrics)
✅ **Analytics dashboard** for visualization
✅ **Trend analysis** over time
✅ **Strategy comparison** capabilities
✅ **Export functionality** for further analysis

**Run tests regularly to track your LLM's performance over time!**