# VibeAgent Dashboard

A lightweight dashboard for the VibeAgent framework with FastAPI backend and modern UI.

## 🚀 Quick Start

### Backend (FastAPI)

```bash
cd /home/muham/development/vibeagent/api

# Install dependencies
pip install -r requirements.txt

# Run the API server
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend

```bash
cd /home/muham/development/vibeagent/frontend

# Option 1: Python simple server
python -m http.server 3000

# Option 2: Using npm
npm start
```

The dashboard will be available at `http://localhost:3000`

## 📋 Features

- **Chat Interface** - Real-time chat with the agent via WebSocket
- **Task Queue** - View and manage agent tasks
- **Skills Dashboard** - Monitor skill status and health
- **Advanced Prompt** - Submit detailed prompts that get broken down into subtasks

## 🔌 API Endpoints

### Agent Control
- `GET /health` - Agent health check
- `GET /agent/skills` - List available skills
- `GET /agent/status` - Detailed agent status
- `POST /agent/prompt` - Process detailed prompt
- `POST /agent/execute` - Execute specific skill

### WebSocket
- `WS /ws` - Real-time updates for chat and status

## 🎨 UI Components

### Sidebar Navigation
- Chat - Interactive chat interface
- Tasks - Task queue and history
- Skills - Skill management and monitoring
- Prompt - Advanced prompt submission

### Main Content Area
- Dynamic tab switching
- Real-time status updates
- Responsive design

## 🔧 Configuration

Edit `../config/agent_config.json` to configure:
- Agent settings
- Skill configurations
- PocketBase connection
- Topics for arXiv search

## 📊 Architecture

```
┌─────────────┐
│   Frontend  │ (index.html)
└──────┬──────┘
       │ HTTP/WebSocket
┌──────▼──────┐
│ FastAPI API │ (main.py)
└──────┬──────┘
       │
┌──────▼──────┐
│   Agent     │ (core/agent.py)
└──────┬──────┘
       │
┌──────▼──────────────────┐
│ Skills                  │
│ - ArxivSkill            │
│ - ScraperSkill          │
│ - LLMSkill              │
│ - PocketBaseSkill       │
└─────────────────────────┘
```

## 🎯 Usage

1. Start the FastAPI backend
2. Start the frontend server
3. Open `http://localhost:3000` in your browser
4. Use the chat interface to interact with the agent
5. Submit advanced prompts for complex tasks
6. Monitor skills and tasks in the dashboard

## 📝 Example Prompts

### Simple Chat
```
Search for papers about prompt engineering from the last 6 months
```

### Advanced Prompt
```
I want to research the latest developments in:
1. Context engineering for LLMs
2. Self-healing AI agents
3. Knowledge base building strategies

For each topic, find papers from the last 6 months, summarize them,
and save the results to PocketBase.
```