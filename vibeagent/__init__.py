"""VibeAgent - A clean agentic system for searching and summarizing arXiv papers."""

__version__ = "2.0.0"
__author__ = "VibeAgent Team"
__email__ = "team@vibeagent.ai"

from .core import Agent, BaseSkill, SkillResult
from .skills import ArxivSkill, ScraperSkill, LLMSkill, SqliteSkill

__all__ = [
    "Agent",
    "BaseSkill", 
    "SkillResult",
    "ArxivSkill",
    "ScraperSkill",
    "LLMSkill",
    
    "SqliteSkill",
]
