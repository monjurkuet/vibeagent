# ArXiv Paper Agent

A clean agentic system for searching and summarizing arXiv papers on AI/ML topics.

## Features

- Searches arXiv for papers from the last 6 months
- Topics covered:
  - Context engineering
  - Prompt engineering
  - Self-healing AI agents
  - Knowledge base building
  - Knowledge storage and retrieval
  - Vector databases
  - RAG (Retrieval-Augmented Generation)
  - Agent frameworks
  - LLM agents

- Uses OpenAI-compatible API for comprehensive paper summarization
- Generates indexed markdown files for easy navigation
- Preserves all information with detailed summaries

## Installation

```bash
cd /home/muham/development/arxiv-agent
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the agent with your OpenAI-compatible API endpoint:

```bash
python arxiv_agent.py http://localhost:8087/v1
```

Or use the default endpoint:

```bash
python arxiv_agent.py
```

## Output

The agent creates:

1. **`papers/arxiv_papers_index.md`** - Main index with all papers organized by topic
2. **`papers/{arxiv_id}.md`** - Individual paper files with full summaries

## Configuration

Edit the `topics` list in `arxiv_agent.py` to customize search topics.

## Requirements

- Python 3.10+
- OpenAI-compatible API running on `http://localhost:8087/v1`