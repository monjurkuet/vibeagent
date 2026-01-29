# LLM Tool Calling Test Suite

## Overview

This test suite evaluates which LLMs support OpenAI-compatible tool/function calling and how well they can use the available tools in the VibeAgent system.

## Features

- **Extensive Logging**: Every request, response, and tool call is logged in detail
- **Comprehensive Tests**: 10 test cases covering various scenarios
- **28 Models**: Tests all available models from the LLM endpoint
- **Detailed Metrics**: Response times, success rates, error tracking
- **JSON Reports**: Saves detailed results for analysis

## What Gets Tested

### Test Cases

1. **Simple Tool Call** - Basic arXiv search
2. **Multiple Tool Calls** - Search and save papers
3. **No Tools Needed** - General question (should not call tools)
4. **Complex Multi-Step** - Research, save, and summarize
5. **Parallel Tool Calls** - Multiple searches at once
6. **Error Handling** - Invalid parameters
7. **Conditional Tool Use** - Tool only used when needed
8. **Nested Tool Calls** - Tool results used in subsequent calls
9. **Tool with Many Parameters** - Complex parameter handling
10. **Tool Choice Override** - Force specific tool usage

### Models Tested

All 28 models from `http://localhost:8087/v1/models` including:
- glm-4.7, glm-4.6
- deepseek-v3, deepseek-v3.1, deepseek-v3.2, deepseek-r1
- qwen3-max, qwen3-32b, qwen3-235b, qwen3-coder-plus
- kimi-k2, kimi-k2-thinking
- gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview
- And more...

## How to Run Tests

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests on all models (this will take 30-60 minutes)
python tests/run_llm_tests.py
```

### Test Specific Models

```bash
# Test only specific models
python tests/run_llm_tests.py --models glm-4.7 deepseek-v3.2 qwen3-max

# Test a single model
python tests/run_llm_tests.py --models glm-4.7
```

### Custom Output Location

```bash
# Save report to custom location
python tests/run_llm_tests.py --output /path/to/custom_report.json
```

### Parallel Testing (Faster)

```bash
# Run tests in parallel (faster but harder to read logs)
python tests/run_llm_tests.py --parallel
```

### Custom LLM Endpoint

```bash
# Test against different LLM endpoint
python tests/run_llm_tests.py --base-url http://localhost:8080/v1
```

## What You'll See

### Console Output

The test suite provides extensive console logging showing:

```
2024-01-24 10:00:00 | INFO | ================================================================================
2024-01-24 10:00:00 | INFO | FETCHING AVAILABLE MODELS
2024-01-24 10:00:00 | INFO | ================================================================================
2024-01-24 10:00:00 | INFO | Request URL: http://localhost:8087/v1/models
2024-01-24 10:00:00 | INFO | ✓ Successfully fetched 28 models
2024-01-24 10:00:00 | INFO | --------------------------------------------------------------------------------
2024-01-24 10:00:00 | INFO |   1. gemini-2.5-flash                    (by google)
2024-01-24 10:00:00 | INFO |   2. gemini-3-pro-preview                (by google)
...
2024-01-24 10:00:00 | INFO | ╔══════════════════════════════════════════════════════════════════════════════════════════╗
2024-01-24 10:00:01 | INFO | ║                      TESTING MODEL: glm-4.7                                     ║
2024-01-24 10:00:01 | INFO | ╚══════════════════════════════════════════════════════════════════════════════════════════╝
2024-01-24 10:00:01 | INFO | Total test cases: 10
2024-01-24 10:00:01 | INFO |
2024-01-24 10:00:01 | INFO | ┌─ TEST 1/10: Simple Tool Call - ArXiv Search
2024-01-24 10:00:01 | INFO | │
2024-01-24 10:00:01 | INFO | │  📤 SENDING REQUEST TO LLM
2024-01-24 10:00:01 | INFO | │  Model: glm-4.7
2024-01-24 10:00:01 | INFO | │  Endpoint: http://localhost:8087/v1/chat/completions
2024-01-24 10:00:01 | INFO | │  Tool Choice: auto
2024-01-24 10:00:01 | INFO | │  Temperature: 0.7
2024-01-24 10:00:01 | INFO | │  Max Tokens: 1000
2024-01-24 10:00:01 | INFO | │  Messages (1):
2024-01-24 10:00:01 | INFO | │    [1] Role: user
2024-01-24 10:00:01 | INFO | │    Content: Search for papers about machine learning
2024-01-24 10:00:01 | INFO | │  Tools Available (1):
2024-01-24 10:00:01 | INFO | │    [1] arxiv_search: Search arXiv for academic papers
2024-01-24 10:00:01 | INFO | │
2024-01-24 10:00:03 | INFO | │  📥 RECEIVED RESPONSE (Status: 200)
2024-01-24 10:00:03 | INFO | │  Response Time: 1.234s
2024-01-24 10:00:03 | INFO | │
2024-01-24 10:00:03 | INFO | │  RESPONSE DETAILS:
2024-01-24 10:00:03 | INFO | │    ID: chat-1234567890
2024-01-24 10:00:03 | INFO | │    Model: glm-4.7
2024-01-24 10:00:03 | INFO | │    Object: chat.completion
2024-01-24 10:00:03 | INFO | │    Usage:
2024-01-24 10:00:03 | INFO | │      Prompt Tokens: 245
2024-01-24 10:00:03 | INFO | │      Completion Tokens: 87
2024-01-24 10:00:03 | INFO | │      Total Tokens: 332
2024-01-24 10:00:03 | INFO | │
2024-01-24 10:00:03 | INFO | │    Finish Reason: tool_calls
2024-01-24 10:00:03 | INFO | │    Content:
2024-01-24 10:00:03 | INFO | │      I'll search for machine learning papers on arXiv.
2024-01-24 10:00:03 | INFO | │
2024-01-24 10:00:03 | INFO | │    Tool Calls: 1
2024-01-24 10:00:03 | INFO | │      Tool Call #1:
2024-01-24 10:00:03 | INFO | │        ID: call_abc123
2024-01-24 10:00:03 | INFO | │        Type: function
2024-01-24 10:00:03 | INFO | │        Function: arxiv_search
2024-01-24 10:00:03 | INFO | │        Arguments:
2024-01-24 10:00:03 | INFO | │          query: machine learning
2024-01-24 10:00:03 | INFO | │          max_results: 10
2024-01-24 10:00:03 | INFO | │
2024-01-24 10:00:03 | INFO | │  VERIFICATION: ✓ PASSED
2024-01-24 10:00:03 | INFO | │  Expected Tools: ['arxiv_search']
2024-01-24 10:00:03 | INFO | │  Actual Tools: ['arxiv_search']
2024-01-24 10:00:03 | INFO | │ ✓ PASSED (response time: 1.234s)
2024-01-24 10:00:03 | INFO | │ └──────────────────────────────────────────────────────────────────────────────────────
```

### JSON Report

After tests complete, a detailed JSON report is saved to `tests/results/llm_tool_calling_report_TIMESTAMP.json`:

```json
{
  "timestamp": "2024-01-24T10:00:00",
  "summary": {
    "total_models": 28,
    "models_with_tool_calling": 18,
    "models_without_tool_calling": 10,
    "total_tests": 280,
    "total_passed": 198,
    "total_failed": 82,
    "success_rate": 70.7,
    "average_response_time": 2.15
  },
  "model_results": {
    "glm-4.7": {
      "total_tests": 10,
      "passed": 9,
      "failed": 1,
      "supports_tool_calling": true,
      "average_response_time": 1.85
    }
  }
}
```

## Understanding the Results

### Success Criteria

A model **supports tool calling** if:
- It makes at least one tool call when appropriate
- It properly formats tool calls with function name and arguments
- It respects the `tool_choice` parameter

A test **passes** if:
- The model calls the expected tools
- The tool parameters match expected values
- No tools are called when not needed

### Common Issues

1. **No tool calls made** - Model doesn't support tool calling or chose not to use tools
2. **Wrong tool called** - Model selected a different tool than expected
3. **Invalid parameters** - Tool called with wrong or missing parameters
4. **Verbose responses** - Model adds conversational text instead of just calling tools
5. **Timeout errors** - Model took too long to respond

## Troubleshooting

### Tests Running Slowly

- Use `--parallel` flag to run tests concurrently
- Test fewer models with `--models` flag
- Reduce `max_tokens` in test cases

### Connection Errors

- Verify LLM endpoint is running: `curl http://localhost:8087/v1/models`
- Check firewall settings
- Increase timeout value in code if needed

### Import Errors

```bash
# Make sure you're in the project root
cd /home/muham/development/vibeagent

# Activate venv
source venv/bin/activate

# Verify Python path
python -c "import sys; print(sys.path)"
```

## Next Steps

After running tests:

1. **Review the console output** to see detailed logs for each model
2. **Check the JSON report** for comprehensive metrics
3. **Identify top-performing models** based on success rate and response time
4. **Update your config** to use the best model for production
5. **Investigate failures** to understand why certain models don't work

## File Structure

```
tests/
├── __init__.py                     # Package initialization
├── llm_tool_calling_tester.py      # Main testing class
├── test_cases.py                   # Test case definitions
├── report_generator.py             # Report generation functions
├── run_llm_tests.py                # Test runner script
├── results/                        # Test reports directory
│   └── llm_tool_calling_report_*.json
└── README.md                       # This file
```

## Support

If you encounter issues:

1. Check the extensive logs in console output
2. Review the JSON report for error details
3. Verify the LLM endpoint is accessible
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

## Contributing

To add new test cases:

1. Edit `tests/test_cases.py`
2. Add a new test case dict with:
   - `name`: Test case name
   - `messages`: Conversation messages
   - `tools`: Available tools
   - `expected_tools`: Expected tool calls (optional)
   - `expect_no_tools`: Set to true if no tools should be called

3. Re-run tests to validate
