"""SQLite storage skill for the agent framework."""

import sqlite3
import json
import os
from typing import List, Dict, Optional, Any

from core.skill import BaseSkill, SkillResult


class SqliteSkill(BaseSkill):
    """Skill for storing data in SQLite."""

    def __init__(
        self, db_path: str = "/home/muham/development/vibeagent/data/vibeagent.db"
    ):
        super().__init__("sqlite", "1.0.0")
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_db()
        self.activate()

    @property
    def parameters_schema(self) -> Dict:
        """JSON Schema for the skill's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save_paper", "get_paper", "list_papers"],
                    "description": "Action to perform on the database",
                },
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv paper ID (required for save_paper and get_paper)",
                },
                "paper_data": {
                    "type": "object",
                    "description": "Paper data object (required for save_paper)",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "published": {"type": "string"},
                        "abstract": {"type": "string"},
                        "summary": {"type": "string"},
                        "url": {"type": "string"},
                        "pdf_url": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["action"],
        }

    def get_tool_schema(self) -> Dict:
        """Get the tool schema for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Store and retrieve papers from SQLite database",
                "parameters": self.parameters_schema,
            },
        }

    def _ensure_db_directory(self):
        """Create the data directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _init_db(self):
        """Initialize the database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    published TEXT,
                    abstract TEXT,
                    summary TEXT,
                    url TEXT,
                    pdf_url TEXT,
                    topics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def validate(self) -> bool:
        """Validate the skill configuration."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1 FROM papers LIMIT 1")
            return True
        except Exception as e:
            print(f"SQLite validation failed: {e}")
            return False

    def get_dependencies(self) -> List[str]:
        """Return list of dependencies."""
        return []

    def execute(self, **kwargs) -> SkillResult:
        """Execute SQLite operations."""
        action = kwargs.pop("action", None)
        if not action:
            return SkillResult(success=False, error="No action specified")
        if action == "save_paper":
            return self._save_paper(**kwargs)
        elif action == "get_paper":
            return self._get_paper(**kwargs)
        elif action == "list_papers":
            return self._list_papers(**kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _save_paper(
        self,
        arxiv_id: str,
        title: str,
        authors: List[str],
        published: str,
        abstract: str,
        summary: str,
        url: str,
        pdf_url: str,
        topics: List[str],
    ) -> SkillResult:
        """Save a paper to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT arxiv_id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
                ).fetchone()

                paper_data = (
                    arxiv_id,
                    title,
                    json.dumps(authors),
                    published,
                    abstract,
                    summary,
                    url,
                    pdf_url,
                    json.dumps(topics),
                )

                if existing:
                    conn.execute(
                        """
                        UPDATE papers SET
                            title = ?, authors = ?, published = ?, abstract = ?,
                            summary = ?, url = ?, pdf_url = ?, topics = ?
                        WHERE arxiv_id = ?
                        """,
                        paper_data[1:] + (paper_data[0],),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO papers (
                            arxiv_id, title, authors, published, abstract,
                            summary, url, pdf_url, topics
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        paper_data,
                    )
                conn.commit()

            return SkillResult(
                success=True,
                data={
                    "arxiv_id": arxiv_id,
                    "action": "saved" if not existing else "updated",
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to save paper: {str(e)}")

    def _get_paper(self, arxiv_id: str) -> SkillResult:
        """Get a paper by arXiv ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
                ).fetchone()

            if row:
                columns = [
                    "arxiv_id",
                    "title",
                    "authors",
                    "published",
                    "abstract",
                    "summary",
                    "url",
                    "pdf_url",
                    "topics",
                    "created_at",
                ]
                paper = dict(zip(columns, row))
                paper["authors"] = json.loads(paper["authors"])
                paper["topics"] = json.loads(paper["topics"])
                return SkillResult(success=True, data=paper)
            return SkillResult(success=False, error=f"Paper not found: {arxiv_id}")
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to get paper: {str(e)}")

    def _list_papers(self, topic: Optional[str] = None, limit: int = 50) -> SkillResult:
        """List papers, optionally filtered by topic."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if topic:
                    cursor = conn.execute(
                        """
                        SELECT * FROM papers
                        WHERE topics LIKE ?
                        ORDER BY published DESC
                        LIMIT ?
                        """,
                        (f"%{topic}%", limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM papers
                        ORDER BY published DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )

                rows = cursor.fetchall()
                columns = [
                    "arxiv_id",
                    "title",
                    "authors",
                    "published",
                    "abstract",
                    "summary",
                    "url",
                    "pdf_url",
                    "topics",
                    "created_at",
                ]
                papers = []
                for row in rows:
                    paper = dict(zip(columns, row))
                    paper["authors"] = json.loads(paper["authors"])
                    paper["topics"] = json.loads(paper["topics"])
                    papers.append(paper)

            return SkillResult(
                success=True,
                data={"papers": papers, "total": len(papers)},
            )
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to list papers: {str(e)}")
