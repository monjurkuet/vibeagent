#!/usr/bin/env python3
"""
VibeAgent - A clean agentic system for searching and summarizing arXiv papers.
"""

import os
import json
import requests
import arxiv
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Paper:
    """Represents an arXiv paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    abstract: str
    summary: str = ""
    url: str = ""
    pdf_url: str = ""


class PocketBaseClient:
    """Handles PocketBase database operations for papers."""

    def __init__(self, base_url: str = "http://localhost:8090", email: str = "", password: str = ""):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.token = None
        self.auth()

    def auth(self):
        """Authenticate with PocketBase."""
        if self.email and self.password:
            try:
                response = requests.post(
                    f"{self.base_url}/api/collections/users/auth-with-password",
                    json={
                        "identity": self.email,
                        "password": self.password
                    },
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                self.token = data['token']
            except Exception as e:
                print(f"Warning: PocketBase auth failed: {e}")

    def _get_headers(self):
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, endpoint: str, data: dict = None):
        """Make a request to PocketBase API."""
        url = f"{self.base_url}/api{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, headers=self._get_headers(), timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=self._get_headers(), json=data, timeout=10)
            elif method == "PATCH":
                response = requests.patch(url, headers=self._get_headers(), json=data, timeout=10)
            elif method == "DELETE":
                response = requests.delete(url, headers=self._get_headers(), timeout=10)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def save_paper(self, paper: Paper, topics: List[str]) -> bool:
        """Save a paper to PocketBase."""
        try:
            # First, try to get existing paper
            existing = self._request("GET", f"/collections/papers/records/{paper.arxiv_id}")

            paper_data = {
                "id": paper.arxiv_id,
                "title": paper.title,
                "authors": json.dumps(paper.authors),
                "published": paper.published,
                "abstract": paper.abstract,
                "summary": paper.summary,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "topics": topics
            }

            if existing:
                # Update existing paper
                self._request("PATCH", f"/collections/papers/records/{paper.arxiv_id}", paper_data)
            else:
                # Create new paper
                self._request("POST", "/collections/papers/records", paper_data)

            return True
        except Exception as e:
            print(f"Error saving paper to PocketBase: {e}")
            return False

    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve a paper by arXiv ID."""
        try:
            data = self._request("GET", f"/collections/papers/records/{arxiv_id}")
            if data:
                return Paper(
                    arxiv_id=data['id'],
                    title=data['title'],
                    authors=json.loads(data['authors']),
                    published=data['published'],
                    abstract=data['abstract'],
                    summary=data.get('summary', ''),
                    url=data['url'],
                    pdf_url=data['pdf_url']
                )
        except Exception as e:
            print(f"Error retrieving paper: {e}")
        return None

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if a paper already exists in PocketBase."""
        try:
            data = self._request("GET", f"/collections/papers/records/{arxiv_id}")
            return data is not None
        except Exception:
            return False

    def get_all_papers(self) -> List[Paper]:
        """Retrieve all papers from PocketBase."""
        try:
            data = self._request("GET", "/collections/papers/records?perPage=100&sort=-published")
            papers = []
            if data and 'items' in data:
                for item in data['items']:
                    papers.append(Paper(
                        arxiv_id=item['id'],
                        title=item['title'],
                        authors=json.loads(item['authors']),
                        published=item['published'],
                        abstract=item['abstract'],
                        summary=item.get('summary', ''),
                        url=item['url'],
                        pdf_url=item['pdf_url']
                    ))
            return papers
        except Exception as e:
            print(f"Error retrieving all papers: {e}")
            return []

    def get_papers_by_topic(self, topic: str) -> List[Paper]:
        """Retrieve all papers for a specific topic."""
        try:
            data = self._request("GET", f"/collections/papers/records?filter=(topics~'{topic}')&sort=-published")
            papers = []
            if data and 'items' in data:
                for item in data['items']:
                    papers.append(Paper(
                        arxiv_id=item['id'],
                        title=item['title'],
                        authors=json.loads(item['authors']),
                        published=item['published'],
                        abstract=item['abstract'],
                        summary=item.get('summary', ''),
                        url=item['url'],
                        pdf_url=item['pdf_url']
                    ))
            return papers
        except Exception as e:
            print(f"Error retrieving papers by topic: {e}")
            return []


class OpenAIClient:
    """Client for OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8087/v1"):
        self.base_url = base_url
        self.chat_url = f"{base_url}/chat/completions"

    def summarize_paper(self, paper: Paper) -> str:
        """Summarize a paper using the OpenAI API."""
        prompt = f"""Please provide a comprehensive summary of this arXiv paper. Include:
1. Full citation details
2. Complete abstract (verbatim)
3. Detailed summary of key contributions and objectives
4. Methodology approach and techniques used
5. Main results and findings
6. Conclusions and future work
7. Key references mentioned

Ensure no information is lost. Be thorough and detailed.

Title: {paper.title}
Authors: {', '.join(paper.authors)}
Published: {paper.published}
Abstract: {paper.abstract}
"""

        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": "glm-4.7",
                    "messages": [
                        {"role": "system", "content": "You are an expert research assistant specializing in academic paper summarization. Provide detailed, comprehensive summaries that preserve all important information."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error summarizing paper {paper.arxiv_id}: {e}")
            return f"# Summary\n\nFailed to generate summary: {str(e)}\n\n## Abstract\n\n{paper.abstract}\n"


class ArxivSearcher:
    """Handles arXiv paper searches."""

    def __init__(self):
        self.client = arxiv.Client()

    def search_papers(
        self,
        query: str,
        max_results: int = 50,
        months_back: int = 6
    ) -> List[Paper]:
        """Search arXiv for papers matching the query."""
        # Calculate date range
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)

        # Build search query with date filter
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        for result in self.client.results(search):
            pub_date = result.published.replace(tzinfo=None)
            if pub_date >= cutoff_date:
                paper = Paper(
                    arxiv_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    published=pub_date.strftime("%Y-%m-%d"),
                    abstract=result.summary.replace('\n', ' '),
                    url=result.entry_id,
                    pdf_url=result.pdf_url
                )
                papers.append(paper)

        return papers


class MarkdownGenerator:
    """Generates markdown files for papers."""

    def __init__(self, output_dir: str = "papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.index_file = self.output_dir / "arxiv_papers_index.md"

    def generate_paper_file(self, paper: Paper) -> str:
        """Generate a markdown file for a single paper."""
        filename = self.output_dir / f"{paper.arxiv_id}.md"

        content = f"""# {paper.title}

**arXiv ID:** {paper.arxiv_id}  
**Authors:** {', '.join(paper.authors)}  
**Published:** {paper.published}  
**URL:** {paper.url}  
**PDF:** {paper.pdf_url}

---

## Abstract

{paper.abstract}

---

{paper.summary}

---

*Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(filename)

    def generate_index(self, papers_by_topic: Dict[str, List[Paper]]):
        """Generate the index markdown file."""
        content = """# arXiv Papers Index

*Generated on: {date}*

---

""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        for topic, papers in papers_by_topic.items():
            content += f"## {topic}\n\n"
            content += f"*Total papers: {len(papers)}*\n\n"

            for paper in sorted(papers, key=lambda p: p.published, reverse=True):
                content += f"- **{paper.published}** - [{paper.title}]({paper.arxiv_id}.md)\n"
                content += f"  - Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}\n"
                content += f"  - arXiv ID: {paper.arxiv_id}\n\n"

            content += "---\n\n"

        with open(self.index_file, 'w', encoding='utf-8') as f:
            f.write(content)


class VibeAgent:
    """Main agent orchestrating the paper search and summarization process."""

    def __init__(
        self,
        openai_url: str = "http://localhost:8087/v1",
        pb_url: str = "http://localhost:8090",
        pb_email: str = "",
        pb_password: str = ""
    ):
        self.searcher = ArxivSearcher()
        self.openai = OpenAIClient(openai_url)
        self.generator = MarkdownGenerator()
        self.pb = PocketBaseClient(pb_url, pb_email, pb_password)

        # Topics to search
        self.topics = [
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

    def run(self):
        """Execute the agent workflow."""
        print("🤖 VibeAgent Starting...")
        print(f"📅 Searching papers from the last 6 months")
        print(f"🔍 Topics: {', '.join(self.topics)}")
        print(f"💾 PocketBase: {self.pb.base_url}\n")

        all_papers_by_topic = {}

        for topic in self.topics:
            print(f"📚 Searching for: {topic}...")
            papers = self.searcher.search_papers(topic, max_results=50, months_back=6)

            # Remove duplicates based on arxiv_id and skip existing papers
            seen_ids = set()
            unique_papers = []
            new_papers = []

            for paper in papers:
                if paper.arxiv_id not in seen_ids:
                    seen_ids.add(paper.arxiv_id)
                    if not self.pb.paper_exists(paper.arxiv_id):
                        unique_papers.append(paper)
                        new_papers.append(paper)

            print(f"   Found {len(papers)} papers, {len(new_papers)} new")

            if unique_papers:
                all_papers_by_topic[topic] = unique_papers

                # Summarize and generate markdown files for new papers only
                for i, paper in enumerate(new_papers, 1):
                    print(f"   📝 Summarizing [{i}/{len(new_papers)}]: {paper.title[:60]}...")
                    paper.summary = self.openai.summarize_paper(paper)
                    self.generator.generate_paper_file(paper)
                    self.pb.save_paper(paper, [topic])

        print("\n📋 Generating index...")
        self.generator.generate_index(all_papers_by_topic)

        total_new_papers = sum(len(papers) for papers in all_papers_by_topic.values())
        total_db_papers = len(self.pb.get_all_papers())
        print(f"\n✅ Done! Processed {total_new_papers} new papers across {len(all_papers_by_topic)} topics.")
        print(f"💾 Total papers in PocketBase: {total_db_papers}")
        print(f"📁 Output directory: {self.generator.output_dir.absolute()}")
        print(f"📄 Index file: {self.generator.index_file.absolute()}")


if __name__ == "__main__":
    import sys

    openai_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8087/v1"
    pb_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8090"
    pb_email = sys.argv[3] if len(sys.argv) > 3 else ""
    pb_password = sys.argv[4] if len(sys.argv) > 4 else ""

    agent = VibeAgent(openai_url, pb_url, pb_email, pb_password)
    agent.run()