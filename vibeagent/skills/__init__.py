"""Skills for the agent framework."""

from .arxiv_skill import ArxivSkill, Paper
from .elasticsearch_skill import ElasticsearchSkill
from .entity_extraction_skill import EntityExtractionSkill
from .firecrawl_skill import FirecrawlSkill
from .llm_skill import LLMSkill
from .multimodal_skill import MultiModalSkill
from .neo4j_skill import Neo4jSkill
from .postgresql_fulltext_skill import PostgreSQLFullTextSkill
from .postgresql_skill import PostgreSQLSkill
from .qdrant_skill import QdrantSkill
from .scraper_skill import ScraperSkill
from .sqlite_skill import SqliteSkill

__all__ = [
    "ArxivSkill",
    "Paper",
    "ElasticsearchSkill",
    "EntityExtractionSkill",
    "FirecrawlSkill",
    "LLMSkill",
    "MultiModalSkill",
    "Neo4jSkill",
    "PostgreSQLSkill",
    "PostgreSQLFullTextSkill",
    "QdrantSkill",
    "ScraperSkill",
    "SqliteSkill",
]
