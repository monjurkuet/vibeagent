# How to Run VibeAgent with Full Logging

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run with full logging
python run_with_logs.py
```

## What This Does

The `run_with_logs.py` script will:

1. **Initialize Database** - Create all 20 tables in the database
2. **Load Configuration** - Read settings from `config/agent_config.json`
3. **Register Skills** - Enable all configured skills (ArXiv, Scraper, LLM, PocketBase)
4. **Initialize Orchestrators** - Set up all 4 orchestrators:
   - ToolOrchestrator (basic mode)
   - ToolOrchestrator (ReAct mode)
   - PlanExecuteOrchestrator
   - ToTOrchestrator
5. **Health Check** - Verify all skills are healthy
6. **Run Workflow** - Search arXiv for papers and process them
7. **Save to Database** - Store all operations in the database
8. **Display Statistics** - Show agent status and database stats

## Logging

All logs are output to:
- **Console** - Real-time output
- **`vibeagent.log`** - Full log file saved in project root

Log levels:
- `DEBUG` - Detailed debugging information
- `INFO` - General informational messages
- `WARNING` - Warning messages
- `ERROR` - Error messages

## Configuration

Edit `config/agent_config.json` to customize:

```json
{
  "agent": {
    "name": "VibeAgent",
    "version": "1.0.0"
  },
  "skills": {
    "arxiv": {
      "enabled": true,
      "max_results": 50,
      "months_back": 6
    },
    "scraper": {
      "enabled": true,
      "timeout": 10
    },
    "llm": {
      "enabled": true,
      "base_url": "http://localhost:8087/v1",
      "model": "glm-4.7"
    }
  },
  "topics": [
    "context engineering",
    "prompt engineering",
    "self-healing AI agents",
    "knowledge base building",
    "knowledge storage and retrieval",
    "vector database",
    "RAG retrieval augmented generation",
    "agent framework",
    "LLM agents"
  ]
}
```

## Features Enabled

### Database Features
- ✅ Session tracking
- ✅ Message storage
- ✅ Tool call tracking
- ✅ Reasoning steps (ReAct/ToT)
- ✅ Performance metrics
- ✅ Error recovery tracking
- ✅ Self-correction tracking
- ✅ Judge evaluations
- ✅ Test case tracking
- ✅ Analytics views

### Orchestrators
- ✅ ToolOrchestrator (basic)
- ✅ ToolOrchestrator (ReAct mode)
- ✅ PlanExecuteOrchestrator
- ✅ ToTOrchestrator

### Skills
- ✅ ArXiv search
- ✅ Web scraping
- ✅ LLM integration
- ✅ PocketBase storage (optional)

## Expected Output

```
================================================================================
🤖 VibeAgent Starting with Full Logging
================================================================================
📝 Configuration loaded from config/agent_config.json
   Agent Name: VibeAgent
   Version: 1.0.0

================================================================================
🗄️  Database Setup
================================================================================
✅ Database initialized at data/vibeagent.db
📊 Database tables: 20 tables
   - sessions
   - messages
   - llm_responses
   - tool_calls
   - tool_results
   - test_cases
   - test_runs
   - judge_evaluations
   - reasoning_steps
   - error_recovery
   - self_corrections
   - performance_metrics
   ...

================================================================================
🤖 Agent Initialization
================================================================================
✅ Agent 'VibeAgent' initialized

================================================================================
🔧 Registering Skills
================================================================================
📚 Registering ArXiv Skill...
   ✅ ArXiv Skill registered: arxiv_search
🌐 Registering Scraper Skill...
   ✅ Scraper Skill registered: scraper
🧠 Registering LLM Skill...
   ✅ LLM Skill registered: llm
      Base URL: http://localhost:8087/v1
      Model: glm-4.7

✅ Total skills registered: 3

================================================================================
🎯 Initializing Orchestrators
================================================================================
🔧 Initializing ToolOrchestrator...
   ✅ ToolOrchestrator initialized
🔄 Initializing ToolOrchestrator (ReAct mode)...
   ✅ ReAct Orchestrator initialized
📋 Initializing PlanExecuteOrchestrator...
   ✅ PlanExecuteOrchestrator initialized
🌳 Initializing ToTOrchestrator...
   ✅ ToTOrchestrator initialized

✅ Total orchestrators initialized: 4

================================================================================
🏥 Health Check
================================================================================
   ✅ arxiv_search
   ✅ scraper
   ✅ llm

================================================================================
🔍 Running arXiv Search Workflow
================================================================================
🔍 Topic: context engineering
--------------------------------------------------------------------------------
   Searching arXiv...
   ✅ Found 50 papers

   [1] Context Engineering for Large Language Models...
       ArXiv ID: 2401.12345
       Published: 2024-01-15
       🧠 Generating summary...
       ✅ Summary: This paper introduces a novel approach to context...
       💾 Saving to database...
       ✅ Saved to database (session_id: 1)
       💾 Saving to PocketBase...
       ✅ Saved to PocketBase

...

================================================================================
📊 Agent Status
================================================================================
   Total Skills: 3
   - arxiv_search: 3 executions
   - scraper: 0 executions
   - llm: 9 executions
   Total Executions: 12

================================================================================
🗄️  Database Statistics
================================================================================
   Sessions: 9
   Messages: 18
   Tool Calls: 9
   Reasoning Steps: 0
   Performance Metrics: 0

================================================================================
✅ VibeAgent Completed Successfully!
================================================================================
📝 Full log saved to: /home/muham/development/vibeagent/vibeagent.log
```

## Troubleshooting

### LLM Connection Failed
- Ensure your LLM API is running at the configured URL
- Check `config/agent_config.json` for correct `base_url` and `model`

### Database Errors
- Delete `data/vibeagent.db` and run again to recreate
- Check file permissions on `data/` directory

### ArXiv Search Failed
- Check internet connection
- ArXiv API may be temporarily unavailable

## Viewing Logs

```bash
# View full log
cat vibeagent.log

# View last 50 lines
tail -n 50 vibeagent.log

# Search for errors
grep ERROR vibeagent.log

# Search for database operations
grep "Database" vibeagent.log
```

## Alternative: Run API Server

For web interface and WebSocket support:

```bash
# Start API server
python -m api.main

# Or use uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 9000 --reload
```

Then access:
- API: http://localhost:9000
- Health check: http://localhost:9000/health
- WebSocket: ws://localhost:9000/ws

## Stopping

Press `Ctrl+C` to stop the agent gracefully.