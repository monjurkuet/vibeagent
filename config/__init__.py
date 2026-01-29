"""Configuration for the agent framework."""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for the agent."""

    def __init__(self, config_path: str = "config/agent_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.ports = self._load_ports()

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "agent": {"name": "VibeAgent", "version": "1.0.0"},
            "skills": {
                "arxiv": {"enabled": True, "max_results": 50, "months_back": 6},
                "scraper": {"enabled": True, "timeout": 10},
                "llm": {
                    "enabled": True,
                    "base_url": "http://localhost:8087/v1",
                    "model": "glm-4.7",
                },
            },
            "pocketbase": {"url": "http://localhost:8090", "email": "", "password": ""},
            "topics": [
                "context engineering",
                "prompt engineering",
                "self-healing AI agents",
                "knowledge base building",
                "knowledge storage and retrieval",
                "vector database",
                "RAG retrieval augmented generation",
                "agent framework",
                "LLM agents",
            ],
        }

    def _load_ports(self) -> dict:
        """Load port configuration."""
        ports_path = Path("config/ports.json")
        if ports_path.exists():
            with open(ports_path) as f:
                return json.load(f)
        return self._default_ports()

    def _default_ports(self) -> dict:
        """Return default port configuration."""
        return {
            "ports": {
                "api": 9000,
                "frontend": 9001,
                "websocket": 9000,
                "pocketbase": 9002,
                "arxiv": 9003,
                "scraper": 9004,
                "llm": 9005,
                "benchmark": 9006,
                "agent": 9007,
                "dashboard": 9008,
            },
            "port_blocks": {
                "api_block": "9000-9099",
                "services_block": "10000-10099",
                "internal_block": "20000-20099",
                "external_block": "30000-30099",
            },
            "available_ports": {
                "8000": "Alternative API",
                "8001": "Alternative Frontend",
                "8002": "WebSocket Alt",
                "8003": "PocketBase Alt",
                "8004": "Arxiv Alt",
                "8005": "Scraper Alt",
                "8006": "LLM Alt",
                "8007": "Benchmark Alt",
                "8008": "Agent Alt",
                "8009": "Dashboard Alt",
            },
        }

    def get_port(self, service: str) -> int:
        """Get port for a specific service."""
        port = self.ports.get("ports", {}).get(service)
        if port is None:
            # Fallback to default ports if service not found
            default_ports = self._default_ports()
            port = default_ports.get("ports", {}).get(service, 8001)
        return port

    def get_api_port(self) -> int:
        """Get API server port."""
        return self.get_port("api")

    def get_frontend_port(self) -> int:
        """Get frontend server port."""
        return self.get_port("frontend")

    def save(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def get(self, *keys) -> Any:
        """Get a configuration value."""
        value = self.config
        for key in keys:
            value = value.get(key)
            if value is None:
                return None
        return value

    def set(self, *keys, value):
        """Set a configuration value."""
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save()
