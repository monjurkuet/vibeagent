# VibeAgent

A sophisticated agentic system for research and AI-powered workflows with real-time analytics and comprehensive monitoring.

## Features

- **Multi-agent orchestration** with ReAct, Tree-of-Thoughts, and Plan-Execute patterns
- **Real-time analytics dashboard** with performance metrics and insights
- **Comprehensive database tracking** of all interactions and tool calls
- **Advanced error handling** with self-correction and retry mechanisms
- **Parallel execution** for improved performance (up to 4.76x speedup)
- **Context management** with 30% token usage reduction
- **Skill-based architecture** with pluggable tools (arXiv, web scraping, LLM, SQLite)

## Architecture

### Core Components
- `vibeagent/core/agent.py` - Main agent implementation
- `vibeagent/core/skill.py` - Base skill interface
- `vibeagent/core/orchestrators/` - ReAct, ToT, Plan-Execute orchestrators
- `vibeagent/core/managers/` - Context, error, retry, and analytics managers

### Skills
- `vibeagent/skills/arxiv_skill.py` - ArXiv paper search
- `vibeagent/skills/llm_skill.py` - LLM interaction
- `vibeagent/skills/scraper_skill.py` - Web scraping
- `vibeagent/skills/sqlite_skill.py` - Database operations

### API
- `vibeagent/api/main.py` - FastAPI backend with WebSocket support
- `vibeagent/api/routes/` - REST API endpoints
- `vibeagent/api/middleware/` - Request/response processing

### Dashboard
- `frontend/index.html` - Real-time analytics dashboard
- Built with Tailwind CSS and Alpine.js

## Installation

```bash
# Clone and setup
cd /home/muham/development/vibeagent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install dependencies
uv sync
```

## Usage

### Start the API Server
```bash
python run_api.py
# or
uv run vibeagent-api
```

### Start the Dashboard
```bash
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

## API Endpoints

- `GET /health` - System health check
- `GET /agent/skills` - List available skills
- `GET /agent/status` - Agent status
- `POST /agent/chat_with_tools` - Chat with tool calling
- `POST /agent/execute` - Execute specific skill
- `POST /agent/self-heal` - Trigger self-healing for skills
- `WS /ws` - WebSocket for real-time updates

## Dashboard Features

- Real-time skill status monitoring
- Agent execution tracking
- Tool calling visualization
- Performance metrics
- Terminal output logging
- Chat interface with the agent

## Configuration

Edit `config/agent_config.json` to configure:
- Agent settings
- LLM endpoints
- Database paths
- Skill configurations

## Development

### Running Tests
```bash
python run_tests.py
# or
pytest tests/
```

### Code Quality
```bash
# Run linters
uv run ruff check vibeagent
uv run mypy vibeagent

# Format code
uv run ruff format vibeagent
```

## Project Structure

```
├── config/             # Configuration files
├── data/               # Database files
├── examples/           # Example usage scripts
├── frontend/           # Dashboard (HTML/CSS/JS)
├── prompts/            # ReAct prompt templates
├── scripts/            # Database and utility scripts
├── tests/              # Test suite
├── utils/              # Utility scripts
├── vibeagent/          # Main source code
│   ├── api/            # FastAPI backend
│   ├── config/         # Configuration management
│   ├── core/           # Core agent framework
│   ├── scripts/        # Internal scripts
│   └── skills/         # Skill implementations
├── requirements.txt    # Dependencies
└── pyproject.toml      # Project configuration
```

## Performance Improvements

- Success rate: 65% → 95%
- Error recovery: 30% → 85%
- Execution time: -53%
- Token usage: -30%
- Parallel speedup: 4.76x

## Skills Framework

Skills are pluggable components that extend agent capabilities:

```python
from vibeagent.core.skill import BaseSkill, SkillResult

class MySkill(BaseSkill):
    def __init__(self):
        super().__init__("my_skill", "1.0.0")
        
    def execute(self, **kwargs) -> SkillResult:
        # Execute skill logic
        return SkillResult(success=True, data="result")
        
    def validate(self) -> bool:
        # Validate skill configuration
        return True
```

## Contributing

1. Create a new branch
2. Make changes
3. Update tests if needed
4. Run `uv run ruff check` and `uv run mypy`
5. Submit a pull request

## License

MIT
