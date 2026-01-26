"""ArXiv search skill for the agent framework."""

import arxiv
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.skill import BaseSkill, SkillResult


@dataclass
class Paper:
    """Represents an arXiv paper."""

    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    abstract: str
    url: str
    pdf_url: str


class ArxivSkill(BaseSkill):
    """Skill for searching arXiv papers."""

    def __init__(self):
        super().__init__("arxiv_search", "1.0.0")
        self.client = arxiv.Client()

    @property
    def parameters_schema(self) -> Dict:
        """JSON Schema for the skill's parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for arXiv papers",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv categories to filter by (e.g., cs.AI, cs.LG)",
                },
            },
            "required": ["query"],
        }

    def get_tool_schema(self) -> Dict:
        """Get the tool schema for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Search arXiv for academic papers",
                "parameters": self.parameters_schema,
            },
        }

    def validate(self) -> bool:
        """Validate the skill configuration."""
        try:
            # Test connection
            search = arxiv.Search(query="test", max_results=1)
            list(self.client.results(search))
            return True
        except Exception as e:
            print(f"ArXiv validation failed: {e}")
            return False

    def get_dependencies(self) -> List[str]:
        """Return list of dependencies."""
        return ["arxiv"]

    def execute(
        self, query: str, max_results: int = 10, months_back: int = 6
    ) -> SkillResult:
        """Search arXiv for papers."""
        try:
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            papers = []
            for result in self.client.results(search):
                pub_date = result.published.replace(tzinfo=None)
                if pub_date >= cutoff_date:
                    papers.append(
                        Paper(
                            arxiv_id=result.entry_id.split("/")[-1],
                            title=result.title,
                            authors=[author.name for author in result.authors],
                            published=pub_date.strftime("%Y-%m-%d"),
                            abstract=result.summary.replace("\n", " "),
                            url=result.entry_id,
                            pdf_url=result.pdf_url,
                        )
                    )

            return SkillResult(
                success=True,
                data={
                    "query": query,
                    "papers": [p.__dict__ for p in papers],
                    "count": len(papers),
                },
                metadata={"max_results": max_results, "months_back": months_back},
            )
        except Exception as e:
            return SkillResult(success=False, error=f"ArXiv search failed: {str(e)}")
