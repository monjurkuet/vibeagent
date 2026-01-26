# 🧪 LLM Capability Test

## What This Tests

This comprehensive test measures your LLM's capabilities across **4 orchestration strategies** and **8 test cases**:

### Strategies Tested
1. **Basic** - Simple tool orchestration
2. **ReAct** - Reasoning + Acting with explicit thoughts
3. **Plan-and-Execute** - Plan first, then execute
4. **Tree-of-Thought (ToT)** - Explore multiple reasoning paths

### Test Cases
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

### Metrics Measured
- ✅ Success rate per strategy
- ✅ Average iterations
- ✅ Average tool calls
- ✅ Response time
- ✅ Parallel execution usage
- ✅ Reasoning steps (ReAct/ToT)
- ✅ Error handling

## How to Run

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests with all strategies (single model)
python test_llm_capabilities.py

# Run with custom LLM URL
python test_llm_capabilities.py --llm-url http://localhost:8087/v1

# Run with specific model
python test_llm_capabilities.py --llm-model glm-4.7

# Test ALL available models from the API
python test_llm_capabilities.py --all-models

# Run only specific strategies
python test_llm_capabilities.py --strategies basic react

# Test all models with specific strategies
python test_llm_capabilities.py --all-models --strategies basic react
```

### Full Options

```bash
python test_llm_capabilities.py --help

Options:
  --llm-url URL       LLM API base URL (default: http://localhost:8087/v1)
  --llm-model MODEL   LLM model name (default: glm-4.7)
  --strategies LIST   Strategies to test: basic, react, plan_execute, tot
  --all-models        Test all available models from the API (overrides --llm-model)
```

## Expected Output

### Console Output

#### Single Model Test

```
🧪 Starting LLM Capability Test
   LLM URL: http://localhost:8087/v1
   Models to test: glm-4.7
   Strategies: basic, react, plan_execute, tot
   Test cases: 8
   Total tests: 32

################################################################################
# STRATEGY: BASIC
################################################################################

================================================================================
TEST: Simple Single Tool
Strategy: basic
Prompt: Search for papers about machine learning on arXiv...
================================================================================
   📚 arxiv_search: query='machine learning', max_results=10

📊 Results:
   Success: ✅
   Iterations: 2
   Tool calls: 1
   Parallel: ❌
   Reasoning steps: 0
   Response time: 1234.56ms
   Final response: I found several papers about machine learning...

... (more tests)

================================================================================
📊 FINAL REPORT
================================================================================

STRATEGY COMPARISON:
Strategy        | Pass  | Fail  | Success % | Avg Iter   | Avg Tools  | Avg Time (ms)
----------------------------------------------------------------------------------------------------
basic           | 7     | 1     | 87.5%     | 2.3        | 1.5        | 1234.56
react           | 8     | 0     | 100.0%    | 2.1        | 1.8        | 1456.78
plan_execute    | 6     | 2     | 75.0%     | 3.5        | 2.2        | 2345.67
tot             | 7     | 1     | 87.5%     | 4.2        | 2.5        | 3123.45

💾 Report saved to: /home/muham/development/vibeagent/llm_capability_report.json

================================================================================
SUMMARY
================================================================================
   Best overall strategy: react (8/8 passed)
   Fastest strategy: basic (1234.56ms avg)
   Most reasoning: tot (2.3 avg steps)
```

#### Multi-Model Test (--all-models)

```
🧪 Starting LLM Capability Test
   LLM URL: http://localhost:8087/v1
📋 Found 3 available models: glm-4.7, llama-3-8b, mistral-7b
   Models to test: glm-4.7, llama-3-8b, mistral-7b
   Strategies: basic, react, plan_execute, tot
   Test cases: 8
   Total tests: 96

================================================================================
🤖 TESTING MODEL: glm-4.7
================================================================================

... (test output for glm-4.7) ...

================================================================================
🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
MODEL: glm-4.7
🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖

... (strategy comparison for glm-4.7) ...

================================================================================
SUMMARY FOR glm-4.7
================================================================================
   Best overall strategy: react (8/8 passed)
   Fastest strategy: basic (1234.56ms avg)
   Most reasoning: tot (2.3 avg steps)

================================================================================
🤖 TESTING MODEL: llama-3-8b
================================================================================

... (test output for other models) ...

================================================================================
🏆 CROSS-MODEL COMPARISON
================================================================================

OVERALL SUCCESS RATE:
Model              | Passed   | Total    | Success %
------------------------------------------------------------
glm-4.7            | 30       | 32       | 93.8%
llama-3-8b         | 26       | 32       | 81.3%
mistral-7b         | 28       | 32       | 87.5%

🥇 Best model: glm-4.7 (93.8% success rate)

💾 Report saved to: /home/muham/development/vibeagent/llm_capability_report.json
```

### JSON Report

#### Single Model Report

```json
{
  "timestamp": "2026-01-24T06:30:00",
  "llm_url": "http://localhost:8087/v1",
  "models_tested": ["glm-4.7"],
  "results": {
    "glm-4.7": {
      "strategies": {
        "basic": {
          "total_tests": 8,
          "passed": 7,
          "failed": 1,
          "success_rate": 0.875,
          "avg_iterations": 2.3,
          "avg_tool_calls": 1.5,
          "avg_response_time_ms": 1234.56,
          "parallel_usage_count": 1,
          "avg_reasoning_steps": 0.0
        },
        ...
      }
    }
  },
  "all_results": [...]
}
```

#### Multi-Model Report (--all-models)

```json
{
  "timestamp": "2026-01-24T06:30:00",
  "llm_url": "http://localhost:8087/v1",
  "models_tested": ["glm-4.7", "llama-3-8b", "mistral-7b"],
  "results": {
    "glm-4.7": {
      "strategies": {
        "basic": {
          "total_tests": 8,
          "passed": 7,
          "failed": 1,
          "success_rate": 0.875,
          "avg_iterations": 2.3,
          "avg_tool_calls": 1.5,
          "avg_response_time_ms": 1234.56,
          "parallel_usage_count": 1,
          "avg_reasoning_steps": 0.0
        },
        ...
      }
    },
    "llama-3-8b": {
      "strategies": { ... }
    },
    "mistral-7b": {
      "strategies": { ... }
    }
  },
  "all_results": [...]
}
```

## Files Generated

1. **`llm_capability_test.log`** - Detailed execution log
2. **`llm_capability_report.json`** - Machine-readable results
3. **`data/llm_capability_test.db`** - Database with all test runs

## Interpreting Results

### Success Rate
- **>90%**: Excellent tool calling capability
- **70-90%**: Good, some edge cases fail
- **<70%**: Needs improvement

### Iterations
- **<2**: Efficient, makes correct decisions quickly
- **2-4**: Normal, reasonable exploration
- **>4**: May be over-thinking or confused

### Parallel Usage
- **High**: LLM recognizes independent tasks
- **Low**: LLM prefers sequential thinking

### Reasoning Steps (ReAct/ToT)
- **High**: Explicit reasoning, better explainability
- **Low**: Intuitive, less transparent

## Requirements

- ✅ LLM API running (OpenAI-compatible)
- ✅ Python 3.10+
- ✅ Virtual environment activated

## Troubleshooting

**LLM connection failed?**
- Check your LLM API is running
- Verify URL with `--llm-url`
- Check model name with `--llm-model`

**--all-models finds no models?**
- Verify your API supports the `/v1/models` endpoint
- Check that models are properly configured in your LLM server
- Ensure the API is accessible from the test script

**Tests timing out?**
- LLM may be slow
- Reduce test cases or strategies
- Check API response time

**Database errors?**
- Delete `data/llm_capability_test.db` and retry
- Check file permissions

## Customization

### Add New Test Cases

Edit `test_llm_capabilities.py` and add to `TEST_CASES`:

```python
{
    "name": "My Custom Test",
    "description": "Test something specific",
    "complexity": "medium",
    "expected_tools": 1,
    "expected_parallel": False,
    "prompt": "Your custom test prompt here",
}
```

### Add Custom Skills

Edit the `setup_skills()` method to add new mock skills.

### Change Strategies

Modify the `run_test_with_strategy()` method to add new orchestration approaches.