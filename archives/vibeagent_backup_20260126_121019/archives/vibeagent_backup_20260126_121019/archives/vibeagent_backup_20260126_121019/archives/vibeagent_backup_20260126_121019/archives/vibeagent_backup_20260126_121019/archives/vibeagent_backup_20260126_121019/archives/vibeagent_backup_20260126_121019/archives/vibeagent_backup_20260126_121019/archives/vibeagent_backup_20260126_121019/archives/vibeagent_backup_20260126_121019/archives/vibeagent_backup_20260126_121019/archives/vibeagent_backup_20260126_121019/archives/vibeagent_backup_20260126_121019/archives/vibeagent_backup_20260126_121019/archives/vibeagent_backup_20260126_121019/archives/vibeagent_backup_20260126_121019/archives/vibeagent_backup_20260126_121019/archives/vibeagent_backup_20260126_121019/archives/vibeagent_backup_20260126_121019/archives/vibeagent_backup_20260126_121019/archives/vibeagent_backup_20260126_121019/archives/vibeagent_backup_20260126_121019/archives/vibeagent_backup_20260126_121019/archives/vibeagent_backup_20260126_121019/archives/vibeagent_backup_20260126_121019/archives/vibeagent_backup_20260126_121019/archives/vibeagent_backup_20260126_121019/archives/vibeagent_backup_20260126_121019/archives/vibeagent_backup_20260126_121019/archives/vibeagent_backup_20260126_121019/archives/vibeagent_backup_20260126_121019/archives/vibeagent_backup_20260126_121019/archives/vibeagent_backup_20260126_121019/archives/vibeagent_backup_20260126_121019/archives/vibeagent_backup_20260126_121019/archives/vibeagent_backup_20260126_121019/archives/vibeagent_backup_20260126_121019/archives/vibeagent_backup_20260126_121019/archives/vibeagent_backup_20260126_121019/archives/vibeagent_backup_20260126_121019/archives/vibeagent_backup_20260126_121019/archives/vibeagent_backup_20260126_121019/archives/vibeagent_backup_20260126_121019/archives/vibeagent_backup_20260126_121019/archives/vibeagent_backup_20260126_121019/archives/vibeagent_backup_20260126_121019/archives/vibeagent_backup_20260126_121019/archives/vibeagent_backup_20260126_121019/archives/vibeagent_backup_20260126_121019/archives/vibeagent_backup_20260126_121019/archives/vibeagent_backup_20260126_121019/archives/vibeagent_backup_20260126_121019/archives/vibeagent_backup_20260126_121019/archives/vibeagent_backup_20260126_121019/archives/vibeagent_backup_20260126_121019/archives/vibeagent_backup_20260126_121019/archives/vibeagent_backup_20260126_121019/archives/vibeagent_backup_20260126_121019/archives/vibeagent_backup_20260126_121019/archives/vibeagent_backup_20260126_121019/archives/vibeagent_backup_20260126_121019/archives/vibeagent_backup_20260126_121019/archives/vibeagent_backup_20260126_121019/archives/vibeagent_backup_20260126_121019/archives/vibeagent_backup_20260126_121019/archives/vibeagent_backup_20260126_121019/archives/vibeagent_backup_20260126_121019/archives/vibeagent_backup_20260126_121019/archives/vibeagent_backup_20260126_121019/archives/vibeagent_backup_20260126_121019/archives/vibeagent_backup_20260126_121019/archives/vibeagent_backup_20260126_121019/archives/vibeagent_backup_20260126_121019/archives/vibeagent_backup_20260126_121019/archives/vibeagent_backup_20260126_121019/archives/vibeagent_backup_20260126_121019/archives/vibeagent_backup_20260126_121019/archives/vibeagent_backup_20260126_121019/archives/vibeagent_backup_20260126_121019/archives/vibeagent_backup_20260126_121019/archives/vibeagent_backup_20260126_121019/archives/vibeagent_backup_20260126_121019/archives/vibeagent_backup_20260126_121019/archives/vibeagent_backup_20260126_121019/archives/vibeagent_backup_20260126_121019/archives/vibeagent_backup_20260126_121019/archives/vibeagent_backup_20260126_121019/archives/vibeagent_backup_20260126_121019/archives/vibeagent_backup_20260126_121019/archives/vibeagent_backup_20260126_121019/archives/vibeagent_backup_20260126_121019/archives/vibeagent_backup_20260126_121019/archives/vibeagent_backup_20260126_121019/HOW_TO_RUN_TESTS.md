# 🚀 How to Run LLM Tool Calling Tests

## Quick Start (Recommended)

```bash
# Navigate to project directory
cd /home/muham/development/vibeagent

# Activate virtual environment
source venv/bin/activate

# Run tests on ALL 28 models (this will take 30-60 minutes)
# You'll see EVERYTHING - every request, response, tool call, etc.
python quick_test.py
```

## What You'll See

The test suite provides **extensive logging** showing:

### 1. Model List
```
================================================================================
FETCHING AVAILABLE MODELS
================================================================================
Request URL: http://localhost:8087/v1/models
✓ Successfully fetched 28 models
--------------------------------------------------------------------------------
  1. gemini-2.5-flash                    (by google)
  2. gemini-3-pro-preview                (by google)
  3. kimi-k2-thinking                   (by iflow)
  4. deepseek-v3.2-chat                 (by iflow)
  ...
 28. glm-4.7                             (by iflow)
================================================================================
```

### 2. Each Model Being Tested
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      TESTING MODEL: glm-4.7                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
Total test cases: 10
```

### 3. Each Test Case with Full Details
```
┌─ TEST 1/10: Simple Tool Call - ArXiv Search
│
│  📤 SENDING REQUEST TO LLM
│  Model: glm-4.7
│  Endpoint: http://localhost:8087/v1/chat/completions
│  Tool Choice: auto
│  Temperature: 0.7
│  Max Tokens: 1000
│  Messages (1):
│    [1] Role: user
│    Content: Search for papers about machine learning
│  Tools Available (1):
│    [1] arxiv_search: Search arXiv for academic papers
│
│  📥 RECEIVED RESPONSE (Status: 200)
│  Response Time: 1.234s
│
│  RESPONSE DETAILS:
│    ID: chat-1234567890
│    Model: glm-4.7
│    Usage:
│      Prompt Tokens: 245
│      Completion Tokens: 87
│      Total Tokens: 332
│
│    Finish Reason: tool_calls
│    Content: I'll search for machine learning papers on arXiv.
│
│    Tool Calls: 1
│      Tool Call #1:
│        ID: call_abc123
│        Type: function
│        Function: arxiv_search
│        Arguments:
│          query: machine learning
│          max_results: 10
│
│  VERIFICATION: ✓ PASSED
│  Expected Tools: ['arxiv_search']
│  Actual Tools: ['arxiv_search']
│ ✓ PASSED (response time: 1.234s)
│ └──────────────────────────────────────────────────────────────────────────────
```

### 4. Model Summary
```
================================================================================
MODEL glm-4.7 SUMMARY:
  Total Tests: 10
  Passed: 9
  Failed: 1
  Success Rate: 90.0%
  Avg Response Time: 1.85s
  Supports Tool Calling: ✓ YES
================================================================================
```

### 5. Final Report
```
================================================================================
🎉 TEST COMPLETED
================================================================================
Total models tested: 28
Total tests run: 280
Models with tool calling: 18
Models without tool calling: 10
Overall success rate: 70.7%
Average response time: 2.15s

📄 Detailed report: tests/results/quick_test_report.json
================================================================================
```

## Test Options

### Test Specific Models (Faster)

```bash
# Test only 2-3 models (takes ~2-5 minutes)
python quick_test.py --models glm-4.7 deepseek-v3.2

# Test a single model (takes ~1-2 minutes)
python quick_test.py --models glm-4.7
```

### Custom Output Location

```bash
# Save report to custom location
python quick_test.py --output /path/to/my_report.json
```

### Test Against Different LLM Endpoint

```bash
# If your LLM is running on a different port
python quick_test.py --base-url http://localhost:8080/v1
```

## What Gets Tested

### 10 Test Cases Per Model

1. **Simple Tool Call** - Basic arXiv search
2. **Multiple Tool Calls** - Search and save papers
3. **No Tools Needed** - General question (should NOT call tools)
4. **Complex Multi-Step** - Research, save, and summarize
5. **Parallel Tool Calls** - Multiple searches at once
6. **Error Handling** - Invalid parameters
7. **Conditional Tool Use** - Tool only used when needed
8. **Nested Tool Calls** - Tool results used in subsequent calls
9. **Tool with Many Parameters** - Complex parameter handling
10. **Tool Choice Override** - Force specific tool usage

### 28 Models Tested

All models from your LLM endpoint including:
- **GLM**: glm-4.7, glm-4.6
- **DeepSeek**: deepseek-v3, deepseek-v3.1, deepseek-v3.2, deepseek-r1
- **Qwen**: qwen3-max, qwen3-32b, qwen3-235b, qwen3-coder-plus
- **Kimi**: kimi-k2, kimi-k2-thinking
- **Gemini**: gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview
- **And more...**

## Understanding the Logs

### Key Information in Logs

1. **📤 SENDING REQUEST TO LLM** - What we're sending to the model
   - Model name
   - Messages (user input)
   - Available tools
   - Parameters (temperature, max_tokens, etc.)

2. **📥 RECEIVED RESPONSE** - What the model returned
   - Response time
   - Token usage
   - Finish reason (why did it stop?)
   - Content (what it said)
   - Tool calls (what tools it wants to use)
   - Arguments (parameters for each tool)

3. **VERIFICATION** - Did it do what we expected?
   - Which tools we expected it to call
   - Which tools it actually called
   - Pass/Fail status

### Common Results

✅ **PASSED** - Model called the right tools with right parameters  
❌ **FAILED** - Model called wrong tools or wrong parameters  
⚠️ **NO TOOLS** - Model didn't call any tools (may not support tool calling)  
⏱️ **TIMEOUT** - Model took too long to respond  
🚫 **ERROR** - Something went wrong (connection error, malformed response, etc.)

## After Tests Complete

### Console Summary

You'll see:
- Total models tested
- How many support tool calling
- Overall success rate
- Average response time
- Top 10 performing models
- Models that don't support tool calling

### JSON Report

Detailed report saved to `tests/results/quick_test_report.json` containing:
- Timestamp
- Summary statistics
- Full results for each model
- Every test case with:
  - Pass/fail status
  - Response time
  - Tool calls made
  - Parameters used
  - Any errors

## Troubleshooting

### "Failed to fetch models"

```bash
# Check if LLM endpoint is running
curl http://localhost:8087/v1/models

# If not running, start it
# (depends on your setup)
```

### "Connection timeout"

```bash
# Check if endpoint is accessible
curl -v http://localhost:8087/v1/models

# Check firewall
sudo ufw status
```

### Tests running too slowly

```bash
# Test fewer models first
python quick_test.py --models glm-4.7

# Then test more if needed
python quick_test.py --models glm-4.7 deepseek-v3.2 qwen3-max
```

### Want to see even more detail?

The logs are already very detailed, but you can also:

```bash
# Save logs to file while watching
python quick_test.py 2>&1 | tee test_output.log

# Or search logs for specific patterns
python quick_test.py 2>&1 | grep "TOOL CALLS"
python quick_test.py 2>&1 | grep "VERIFICATION"
```

## Next Steps After Testing

1. **Review the console output** - See which models performed best
2. **Check the JSON report** - Get detailed metrics
3. **Identify top performers** - Models with highest success rate
4. **Update your config** - Use the best model for production
5. **Investigate failures** - Understand why some models don't work

## Quick Reference

```bash
# Test all models (30-60 min)
python quick_test.py

# Test 3 models (2-5 min)
python quick_test.py --models glm-4.7 deepseek-v3.2 qwen3-max

# Test 1 model (1-2 min)
python quick_test.py --models glm-4.7

# Custom output
python quick_test.py --output my_report.json

# Different endpoint
python quick_test.py --base-url http://localhost:8080/v1
```

---

**Ready to start? Just run:**

```bash
cd /home/muham/development/vibeagent
source venv/bin/activate
python quick_test.py
```

You'll see EVERYTHING - every request, response, tool call, parameter, and verification result! 🎉