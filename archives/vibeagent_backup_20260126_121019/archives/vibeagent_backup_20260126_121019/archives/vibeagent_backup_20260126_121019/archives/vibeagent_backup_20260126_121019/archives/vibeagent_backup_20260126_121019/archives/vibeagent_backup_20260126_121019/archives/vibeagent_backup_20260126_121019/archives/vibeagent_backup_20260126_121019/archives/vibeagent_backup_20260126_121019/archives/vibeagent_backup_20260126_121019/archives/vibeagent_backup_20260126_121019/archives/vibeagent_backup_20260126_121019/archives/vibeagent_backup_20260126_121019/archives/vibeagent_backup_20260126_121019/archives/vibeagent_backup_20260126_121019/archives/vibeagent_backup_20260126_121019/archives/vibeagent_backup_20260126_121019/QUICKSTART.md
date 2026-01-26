# 🚀 How to Run VibeAgent

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run with full logging
python run_with_logs.py
```

That's it! The agent will:
- ✅ Initialize database with all 20 tables
- ✅ Load configuration
- ✅ Register all skills (ArXiv, Scraper, LLM)
- ✅ Initialize all 4 orchestrators
- ✅ Run health checks
- ✅ Search arXiv for papers
- ✅ Process and summarize papers
- ✅ Save everything to database
- ✅ Display full statistics

## What You'll See

All logs are displayed in real-time AND saved to `vibeagent.log`

### Console Output
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

## Features Enabled

### Database (20 tables)
- ✅ Sessions tracking
- ✅ Messages storage
- ✅ Tool calls & results
- ✅ Reasoning steps
- ✅ Performance metrics
- ✅ Error recovery
- ✅ Self-corrections
- ✅ Judge evaluations
- ✅ Test cases & runs
- ✅ Analytics views

### Orchestrators (4 modes)
- ✅ ToolOrchestrator (basic)
- ✅ ToolOrchestrator (ReAct)
- ✅ PlanExecuteOrchestrator
- ✅ ToTOrchestrator

### Skills
- ✅ ArXiv search
- ✅ Web scraping
- ✅ LLM integration
- ✅ PocketBase (optional)

## Configuration

Edit `config/agent_config.json`:

```json
{
  "skills": {
    "llm": {
      "enabled": true,
      "base_url": "http://localhost:8087/v1",
      "model": "glm-4.7"
    }
  }
}
```

## View Logs

```bash
# View full log
cat vibeagent.log

# View last 50 lines
tail -n 50 vibeagent.log

# Search for errors
grep ERROR vibeagent.log
```

## Alternative: API Server

For web interface:

```bash
python -m api.main
```

Then visit:
- http://localhost:9000/health
- http://localhost:9000/docs (API docs)

## Requirements

- Python 3.10+
- LLM API running (configured in `config/agent_config.json`)
- Internet connection (for ArXiv search)

## Troubleshooting

**LLM connection failed?**
- Check your LLM API is running
- Verify URL in config

**Database errors?**
- Delete `data/vibeagent.db` and run again

**ArXiv search failed?**
- Check internet connection
- ArXiv API may be busy

## Stop

Press `Ctrl+C` to stop gracefully.