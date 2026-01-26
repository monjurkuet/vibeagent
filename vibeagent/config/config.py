"""Configuration for the agent framework."""

import json
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the agent."""

    def __init__(self, config_path: str = "config/agent_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.ports = self._load_ports()

    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "agent": {
                "name": "VibeAgent",
                "version": "1.0.0"
            },
            "skills": {
                "arxiv": {
                    "enabled": True,
                    "max_results": 50,
                    "months_back": 6
                },
                "scraper": {
                    "enabled": True,
                    "timeout": 10
                },
                "llm": {
                    "enabled": True,
                    "base_url": "http://localhost:8087/v1",
                    "model": "glm-4.7"
                }
            },
            "sqlite": {
                "database_path": "data/vibeagent.db"
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

    def _load_ports(self) -> Dict:
        """Load port configuration."""
        ports_file = self.config_path.parent / "ports.json"
        if ports_file.exists():
            with open(ports_file, 'r') as f:
                return json.load(f)
        return {
            "api": 8000,
            "dashboard": 8080,
            "database": 5432
        }

    def get(self, *keys, default=None) -> Any:
        """Get configuration value with dot notation."""
        value = self.config
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def get_api_port(self) -> int:
        """Get API port."""
        port = self.get("api", "port")
        if port is None:
            port = self.ports.get("api", 8000)
        return port
