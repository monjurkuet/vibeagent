"""Base skill interface for the agent framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class SkillStatus(Enum):
    """Status of a skill."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class SkillResult:
    """Result from a skill execution."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSkill(ABC):
    """Base class for all skills."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.status = SkillStatus.ACTIVE
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """Execute the skill with given parameters."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the skill is properly configured."""

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Return list of dependencies required by this skill."""

    def health_check(self) -> bool:
        """Check if the skill is healthy."""
        try:
            return self.validate()
        except Exception:
            return False

    def get_info(self) -> dict:
        """Get skill information."""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
        }

    def _record_usage(self):
        """Record that the skill was used."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

    def _record_error(self):
        """Record an error."""
        self.error_count += 1
        self.status = SkillStatus.ERROR

    def activate(self):
        """Activate the skill."""
        self.status = SkillStatus.ACTIVE

    def deactivate(self):
        """Deactivate the skill."""
        self.status = SkillStatus.INACTIVE

    @property
    def parameters_schema(self) -> dict:
        """
        Return JSON Schema for the skill's parameters.

        This property should be overridden by subclasses to define the
        expected input parameters in JSON Schema format. The schema is used
        to generate tool calling definitions for AI models.

        Example implementation:
            return {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }

        Returns:
            Dict: JSON Schema describing the skill's parameters
        """
        return {}

    def get_tool_schema(self) -> dict:
        """
        Generate OpenAI-compatible function definition for this skill.

        This method creates a tool calling schema that can be used with
        OpenAI's function calling API and similar interfaces. It combines
        the skill's metadata with its parameters schema.

        Subclasses can override this method to customize the tool schema
        (e.g., add descriptions, modify the schema structure).

        Example output:
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                }
            }

        Returns:
            Dict: OpenAI-compatible function definition

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_tool_schema() to support tool calling"
        )
