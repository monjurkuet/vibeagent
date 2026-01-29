"""Web scraping skill for the agent framework."""

import requests
from bs4 import BeautifulSoup

from ..core.skill import BaseSkill, SkillResult


class ScraperSkill(BaseSkill):
    """Skill for scraping web pages."""

    def __init__(self):
        super().__init__("web_scraper", "1.0.0")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        self.activate()

    @property
    def parameters_schema(self) -> dict:
        """JSON Schema for the skill's parameters."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the web page to scrape",
                },
                "extract_text": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to extract text content",
                },
                "extract_links": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to extract links",
                },
            },
            "required": ["url"],
        }

    def get_tool_schema(self) -> dict:
        """Get the tool schema for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Scrape web pages and extract content",
                "parameters": self.parameters_schema,
            },
        }

    def validate(self) -> bool:
        """Validate the skill configuration."""
        try:
            # Test with a simple request
            response = self.session.get("https://httpbin.org/get", timeout=5)
            return response.status_code == 200
        except requests.exceptions.Timeout:
            print("Scraper validation timeout")
            return False
        except requests.exceptions.ConnectionError:
            print("Scraper connection error")
            return False
        except Exception as e:
            print(f"Scraper validation failed: {e}")
            return False

    def get_dependencies(self) -> list[str]:
        """Return list of dependencies."""
        return ["requests", "beautifulsoup4"]

    def execute(
        self, url: str, extract_text: bool = True, extract_links: bool = False
    ) -> SkillResult:
        """Scrape a web page."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(response.content),
            }

            if extract_text or extract_links:
                soup = BeautifulSoup(response.text, "html.parser")

                if extract_text:
                    result["title"] = soup.title.string if soup.title else ""
                    result["text"] = soup.get_text(separator="\n", strip=True)

                if extract_links:
                    result["links"] = [
                        {"href": link.get("href"), "text": link.get_text().strip()}
                        for link in soup.find_all("a", href=True)
                    ]

            return SkillResult(success=True, data=result)
        except requests.exceptions.HTTPError as e:
            return SkillResult(
                success=False,
                error=f"HTTP error {e.response.status_code}: {str(e)}"
            )
        except requests.exceptions.Timeout:
            return SkillResult(success=False, error="Request timed out")
        except requests.exceptions.RequestException as e:
            return SkillResult(success=False, error=f"Scraping failed: {str(e)}")
        except Exception as e:
            return SkillResult(success=False, error=f"Unexpected error: {str(e)}")
