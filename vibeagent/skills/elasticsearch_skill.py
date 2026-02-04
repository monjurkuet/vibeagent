"""Elasticsearch Skill for keyword-based search.

This skill provides BM25 keyword search using Elasticsearch for traditional
information retrieval complementing vector-based semantic search.
"""

import logging
from datetime import datetime
from typing import Any

from ..core.skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class ElasticsearchSkill(BaseSkill):
    """Skill for keyword search using Elasticsearch."""

    def __init__(
        self,
        hosts: str | list[str] = "http://localhost:9200",
        index_name: str = "knowledge_base",
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize Elasticsearch skill.

        Args:
            hosts: Elasticsearch hosts (single or list)
            index_name: Default index name
            username: Elasticsearch username
            password: Elasticsearch password
            api_key: Elasticsearch API key
        """
        super().__init__(
            name="elasticsearch",
            version="1.0.0",
            description="Keyword-based search with Elasticsearch",
        )
        self.hosts = hosts
        self.index_name = index_name
        self.username = username
        self.password = password
        self.api_key = api_key
        self._client = None

    def _get_client(self) -> Any:
        """Get or create Elasticsearch client."""
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch

                auth = None
                if self.username and self.password:
                    auth = (self.username, self.password)

                self._client = Elasticsearch(
                    hosts=self.hosts,
                    basic_auth=auth,
                    api_key=self.api_key,
                    verify_certs=False,
                )

                # Create index if it doesn't exist
                self._ensure_index()

            except ImportError:
                raise ImportError(
                    "Elasticsearch not installed. Install with: pip install elasticsearch"
                )
        return self._client

    def _ensure_index(self):
        """Ensure index exists with proper mappings."""
        try:
            if not self._client.indices.exists(index=self.index_name):
                index_mapping = {
                    "mappings": {
                        "properties": {
                            "content": {"type": "text", "analyzer": "standard"},
                            "title": {"type": "text", "analyzer": "standard"},
                            "url": {"type": "keyword"},
                            "metadata": {"type": "object", "dynamic": True},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "content_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop"],
                                }
                            }
                        },
                    },
                }
                self._client.indices.create(index=self.index_name, body=index_mapping)
                logger.info(f"Created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error ensuring index: {e}")

    def validate(self) -> bool:
        """Validate skill configuration."""
        try:
            client = self._get_client()
            return client.ping()
        except Exception:
            return False

    def get_tool_schema(self) -> dict[str, Any]:
        """Get OpenAI function schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": "Search for documents using BM25 keyword search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results",
                            "default": 10,
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to search in",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self,
        action: str = "search",
        query: str | None = None,
        documents: list[dict] | None = None,
        limit: int = 10,
        index: str | None = None,
        fields: list[str] | None = None,
        filter: dict | None = None,
        ids: list[str] | None = None,
        **kwargs,
    ) -> SkillResult:
        """Execute Elasticsearch skill.

        Args:
            action: Action to perform (search, insert, delete, update)
            query: Search query
            documents: Documents to index
            limit: Number of results
            index: Index name
            fields: Fields to search in
            filter: Query filters
            ids: Document IDs to delete

        Returns:
            SkillResult with operation results
        """
        if not self.validate():
            return SkillResult(success=False, error="Invalid configuration")

        index_name = index or self.index_name

        if action == "search":
            return self._search(query, limit, index_name, fields, filter, **kwargs)
        elif action == "insert":
            return self._insert(documents, index_name, **kwargs)
        elif action == "delete":
            return self._delete(ids, index_name, **kwargs)
        elif action == "update":
            return self._update(documents, index_name, **kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _search(
        self,
        query: str,
        limit: int,
        index_name: str,
        fields: list[str] | None,
        filter: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Search for documents using BM25.

        Args:
            query: Search query
            limit: Number of results
            index_name: Index to search
            fields: Fields to search in
            filter: Query filters

        Returns:
            SkillResult with search results
        """
        try:
            client = self._get_client()

            search_fields = fields or ["content", "title"]
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": search_fields,
                                    "type": "best_fields",
                                }
                            }
                        ],
                    }
                },
                "size": limit,
                "_source": ["content", "title", "url", "metadata", "created_at"],
            }

            if filter:
                search_body["query"]["bool"]["filter"] = filter

            response = client.search(index=index_name, body=search_body)

            results = []
            for hit in response.get("hits", {}).get("hits", []):
                results.append(
                    {
                        "id": hit.get("_id"),
                        "score": hit.get("_score"),
                        "content": hit.get("_source", {}).get("content", ""),
                        "title": hit.get("_source", {}).get("title", ""),
                        "url": hit.get("_source", {}).get("url", ""),
                        "metadata": hit.get("_source", {}).get("metadata", {}),
                    }
                )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total": len(results),
                    "max_score": response.get("hits", {}).get("max_score"),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error searching: {e}")
            return SkillResult(success=False, error=f"Search failed: {str(e)}")

    def _insert(self, documents: list[dict], index_name: str, **kwargs) -> SkillResult:
        """Insert documents into index.

        Args:
            documents: List of documents to index
            index_name: Index name

        Returns:
            SkillResult with insert results
        """
        if not documents:
            return SkillResult(success=False, error="No documents to insert")

        try:
            client = self._get_client()

            from elasticsearch.helpers import bulk

            actions = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", f"{datetime.now().timestamp()}-{i}")
                action = {
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": {
                        "content": doc.get("content", ""),
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                        "metadata": doc.get("metadata", {}),
                        "created_at": datetime.now().isoformat(),
                    },
                }
                actions.append(action)

            success, failed = bulk(client, actions, index=index_name, raise_on_error=False)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "inserted": success,
                    "failed": len(failed) if failed else 0,
                    "total": len(documents),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error inserting documents: {e}")
            return SkillResult(success=False, error=f"Insert failed: {str(e)}")

    def _delete(self, ids: list[str] | None, index_name: str, **kwargs) -> SkillResult:
        """Delete documents from index.

        Args:
            ids: Document IDs to delete
            index_name: Index name

        Returns:
            SkillResult with delete results
        """
        if not ids:
            return SkillResult(success=False, error="No IDs provided")

        try:
            client = self._get_client()

            from elasticsearch.helpers import bulk

            actions = [{"_op_type": "delete", "_index": index_name, "_id": doc_id} for doc_id in ids]

            success, failed = bulk(client, actions, raise_on_error=False)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "deleted": success,
                    "failed": len(failed) if failed else 0,
                    "total": len(ids),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error deleting documents: {e}")
            return SkillResult(success=False, error=f"Delete failed: {str(e)}")

    def _update(self, documents: list[dict], index_name: str, **kwargs) -> SkillResult:
        """Update documents in index.

        Args:
            documents: Documents to update
            index_name: Index name

        Returns:
            SkillResult with update results
        """
        if not documents:
            return SkillResult(success=False, error="No documents to update")

        try:
            client = self._get_client()

            from elasticsearch.helpers import bulk

            actions = []
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    continue

                action = {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": doc_id,
                    "doc": {
                        "content": doc.get("content", ""),
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                        "metadata": doc.get("metadata", {}),
                        "updated_at": datetime.now().isoformat(),
                    },
                    "doc_as_upsert": True,
                }
                actions.append(action)

            success, failed = bulk(client, actions, raise_on_error=False)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "updated": success,
                    "failed": len(failed) if failed else 0,
                    "total": len(documents),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error updating documents: {e}")
            return SkillResult(success=False, error=f"Update failed: {str(e)}")

    def health_check(self) -> bool:
        """Check if Elasticsearch is accessible."""
        try:
            client = self._get_client()
            return client.ping()
        except Exception:
            return False