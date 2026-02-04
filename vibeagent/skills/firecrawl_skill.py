"""Firecrawl Skill for AI-powered web scraping.

This skill uses Firecrawl to scrape websites and convert them to LLM-ready data.
Firecrawl handles JavaScript rendering, dynamic content, and provides clean markdown output.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.skill import BaseSkill, SkillResult, SkillStatus

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result from scraping a URL."""

    url: str
    title: str
    markdown: str
    html: str | None = None
    links: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FirecrawlSkill(BaseSkill):
    """Skill for AI-powered web scraping using Firecrawl."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = "https://api.firecrawl.dev/v1",
        timeout: int = 30,
    ):
        """Initialize Firecrawl skill.

        Args:
            api_key: Firecrawl API key (defaults to FIRECRAWL_API_KEY env var)
            api_url: Firecrawl API URL
            timeout: Request timeout in seconds
        """
        super().__init__(
            name="firecrawl",
            version="1.0.0",
            description="AI-powered web scraping with Firecrawl",
        )
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout
        self._client = None

    def _get_client(self) -> Any:
        """Get or create Firecrawl client."""
        if self._client is None:
            try:
                from firecrawl import FirecrawlApp

                self._client = FirecrawlApp(api_key=self.api_key, api_url=self.api_url)
            except ImportError:
                raise ImportError(
                    "Firecrawl is not installed. Install with: pip install firecrawl-py"
                )
        return self._client

    def validate(self) -> bool:
        """Validate skill configuration."""
        if not self.api_key:
            import os

            self.api_key = os.getenv("FIRECRAWL_API_KEY")
            if not self.api_key:
                logger.warning("Firecrawl API key not provided")
                return False
        return True

    def get_tool_schema(self) -> dict[str, Any]:
        """Get OpenAI function schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": "scrape_url",
                "description": "Scrape a website and convert to LLM-ready markdown using Firecrawl",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to scrape",
                        },
                        "formats": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["markdown", "html"]},
                            "description": "Output formats to include",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    def execute(
        self,
        action: str = "scrape",
        url: str | None = None,
        urls: list[str] | None = None,
        formats: list[str] | None = None,
        **kwargs,
    ) -> SkillResult:
        """Execute Firecrawl skill.

        Args:
            action: Action to perform (scrape, crawl, map)
            url: Single URL to scrape
            urls: Multiple URLs for batch operations
            formats: Output formats (markdown, html)

        Returns:
            SkillResult with scraped content
        """
        if not self.validate():
            return SkillResult(success=False, error="Invalid configuration")

        if action == "scrape":
            return self._scrape_url(url, formats or ["markdown"], **kwargs)
        elif action == "crawl":
            return self._crawl_site(url, **kwargs)
        elif action == "map":
            return self._map_site(url, **kwargs)
        elif action == "batch_scrape":
            return self._batch_scrape(urls, formats or ["markdown"], **kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _scrape_url(
        self,
        url: str,
        formats: list[str],
        include_links: bool = True,
        include_images: bool = True,
        **kwargs,
    ) -> SkillResult:
        """Scrape a single URL.

        Args:
            url: URL to scrape
            formats: Output formats (markdown, html)
            include_links: Whether to extract links
            include_images: Whether to extract images

        Returns:
            SkillResult with scraped content
        """
        try:
            client = self._get_client()

            params = {
                "formats": formats,
            }

            scrape_result = client.scrape_url(url, params=params)

            result = ScrapeResult(
                url=url,
                title=scrape_result.get("metadata", {}).get("title", url),
                markdown=scrape_result.get("markdown", ""),
                html=scrape_result.get("html"),
                metadata=scrape_result.get("metadata", {}),
            )

            # Extract links if available
            if include_links and "links" in scrape_result.get("metadata", {}):
                result.links = scrape_result["metadata"]["links"]

            # Extract images if available
            if include_images and "images" in scrape_result.get("metadata", {}):
                result.images = scrape_result["metadata"]["images"]

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "url": result.url,
                    "title": result.title,
                    "content": result.markdown,
                    "html": result.html,
                    "links": result.links,
                    "images": result.images,
                    "metadata": result.metadata,
                    "scraped_at": result.scraped_at,
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error scraping URL {url}: {e}")
            return SkillResult(success=False, error=f"Failed to scrape URL: {str(e)}")

    def _crawl_site(
        self,
        url: str,
        limit: int = 100,
        max_depth: int = 2,
        exclude_paths: list[str] | None = None,
        **kwargs,
    ) -> SkillResult:
        """Crawl an entire website.

        Args:
            url: Starting URL
            limit: Maximum number of pages to crawl
            max_depth: Maximum crawl depth
            exclude_paths: Paths to exclude from crawling

        Returns:
            SkillResult with crawled pages
        """
        try:
            client = self._get_client()

            crawl_params = {
                "limit": limit,
                "maxDepth": max_depth,
            }

            if exclude_paths:
                crawl_params["excludePaths"] = exclude_paths

            # Start crawl (this is async in Firecrawl)
            crawl_status = client.crawl_url(url, params=crawl_params, wait_until_done=True)

            results = []
            if crawl_status and "data" in crawl_status:
                for page in crawl_status["data"]:
                    results.append(
                        {
                            "url": page.get("url"),
                            "markdown": page.get("markdown", ""),
                            "metadata": page.get("metadata", {}),
                        }
                    )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "url": url,
                    "pages_crawled": len(results),
                    "results": results,
                    "status": crawl_status.get("status", "completed") if crawl_status else "unknown",
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error crawling site {url}: {e}")
            return SkillResult(success=False, error=f"Failed to crawl site: {str(e)}")

    def _map_site(self, url: str, **kwargs) -> SkillResult:
        """Map a website to discover all pages.

        Args:
            url: Starting URL

        Returns:
            SkillResult with discovered URLs
        """
        try:
            client = self._get_client()

            map_result = client.map_url(url)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "url": url,
                    "links": map_result.get("links", []) if map_result else [],
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error mapping site {url}: {e}")
            return SkillResult(success=False, error=f"Failed to map site: {str(e)}")

    def _batch_scrape(
        self,
        urls: list[str],
        formats: list[str],
        **kwargs,
    ) -> SkillResult:
        """Scrape multiple URLs in batch.

        Args:
            urls: List of URLs to scrape
            formats: Output formats (markdown, html)

        Returns:
            SkillResult with all scraped results
        """
        if not urls:
            return SkillResult(success=False, error="No URLs provided for batch scrape")

        try:
            client = self._get_client()

            params = {"formats": formats}

            # Batch scrape
            batch_result = client.batch_scrape_urls(urls, params=params)

            results = []
            if batch_result and "data" in batch_result:
                for item in batch_result["data"]:
                    results.append(
                        {
                            "url": item.get("url"),
                            "markdown": item.get("markdown", ""),
                            "metadata": item.get("metadata", {}),
                            "success": item.get("success", True),
                            "error": item.get("error"),
                        }
                    )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "total_urls": len(urls),
                    "successful": len([r for r in results if r.get("success")]),
                    "failed": len([r for r in results if not r.get("success")]),
                    "results": results,
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error in batch scrape: {e}")
            return SkillResult(success=False, error=f"Failed batch scrape: {str(e)}")

    def health_check(self) -> bool:
        """Check if Firecrawl is accessible."""
        try:
            self.validate()
            return self._get_client() is not None
        except Exception:
            return False