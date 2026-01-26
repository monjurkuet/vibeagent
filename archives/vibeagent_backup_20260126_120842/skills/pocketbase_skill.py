"""PocketBase storage skill for the agent framework."""

import requests
import json
from typing import List, Dict, Optional, Any

from core.skill import BaseSkill, SkillResult


class PocketBaseSkill(BaseSkill):
    """Skill for storing data in PocketBase."""

    def __init__(
        self,
        base_url: str = "http://localhost:8090",
        email: str = "",
        password: str = "",
    ):
        super().__init__("pocketbase", "1.0.0")
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.token = None
        self.auth()

    @property
    def parameters_schema(self) -> Dict:
        """JSON Schema for the skill's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save_paper", "get_paper", "list_papers"],
                    "description": "Action to perform on PocketBase",
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
                "description": "Store and retrieve papers from PocketBase database",
                "parameters": self.parameters_schema,
            },
        }

    def auth(self):
        """Authenticate with PocketBase."""
        if self.email and self.password:
            try:
                print(f"🔐 Attempting PocketBase authentication with {self.email}...")
                response = requests.post(
                    f"{self.base_url}/api/collections/_superusers/auth-with-password",
                    json={"identity": self.email, "password": self.password},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                self.token = data["token"]
                self.activate()
                print(f"✅ PocketBase authenticated successfully")
            except requests.exceptions.HTTPError as e:
                print(f"❌ PocketBase auth failed: {e}")
                print(f"   Status: {e.response.status_code}")
                print(f"   Response: {e.response.text[:200]}")
                self.token = None
            except Exception as e:
                print(f"❌ PocketBase auth failed: {e}")
                self.token = None
        else:
            print(
                f"⚠️  PocketBase auth skipped: no credentials provided (email={bool(self.email)}, password={bool(self.password)})"
            )

    def _get_headers(self) -> Dict:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make a request to PocketBase API."""
        url = f"{self.base_url}/api{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, headers=self._get_headers(), timeout=10)
            elif method == "POST":
                response = requests.post(
                    url, headers=self._get_headers(), json=data, timeout=10
                )
            elif method == "PATCH":
                response = requests.patch(
                    url, headers=self._get_headers(), json=data, timeout=10
                )
            elif method == "DELETE":
                response = requests.delete(url, headers=self._get_headers(), timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def validate(self) -> bool:
        """Validate the skill configuration."""
        if not self.token:
            print(f"PocketBase validation failed: No authentication token")
            return False
        try:
            response = requests.get(
                f"{self.base_url}/api/collections",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 200:
                return True
            print(f"PocketBase validation failed: HTTP {response.status_code}")
            return False
        except Exception as e:
            print(f"PocketBase validation failed: {e}")
            return False

    def get_dependencies(self) -> List[str]:
        """Return list of dependencies."""
        return ["requests"]

    def execute(self, **kwargs) -> SkillResult:
        """Execute PocketBase operations."""
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
        """Save a paper to PocketBase."""
        try:
            # Check if paper exists by arxiv_id
            existing_data = self._request(
                "GET", f"/collections/papers/records?filter=(arxiv_id='{arxiv_id}')"
            )
            existing_record = (
                existing_data.get("items", [None])[0] if existing_data else None
            )

            paper_data = {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": json.dumps(authors),
                "published": published,
                "abstract": abstract,
                "summary": summary,
                "url": url,
                "pdf_url": pdf_url,
                "topics": topics,
            }

            if existing_record:
                # Update existing paper
                self._request(
                    "PATCH",
                    f"/collections/papers/records/{existing_record['id']}",
                    paper_data,
                )
            else:
                # Create new paper (PocketBase will auto-generate ID)
                self._request("POST", "/collections/papers/records", paper_data)

            return SkillResult(
                success=True,
                data={
                    "arxiv_id": arxiv_id,
                    "action": "saved" if not existing_record else "updated",
                },
            )
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to save paper: {str(e)}")

    def _get_paper(self, arxiv_id: str) -> SkillResult:
        """Get a paper by arXiv ID."""
        try:
            data = self._request(
                "GET", f"/collections/papers/records?filter=(arxiv_id='{arxiv_id}')"
            )
            if data and data.get("items"):
                return SkillResult(success=True, data=data["items"][0])
            return SkillResult(success=False, error=f"Paper not found: {arxiv_id}")
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to get paper: {str(e)}")

    def _list_papers(self, topic: Optional[str] = None, limit: int = 50) -> SkillResult:
        """List papers, optionally filtered by topic."""
        try:
            endpoint = "/collections/papers/records"
            params = f"?perPage={limit}&sort=-published"

            if topic:
                params += f"&filter=(topics~'{topic}')"

            data = self._request("GET", f"{endpoint}{params}")

            if data and "items" in data:
                return SkillResult(
                    success=True,
                    data={"papers": data["items"], "total": data.get("totalItems", 0)},
                )
            return SkillResult(success=True, data={"papers": [], "total": 0})
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to list papers: {str(e)}")
