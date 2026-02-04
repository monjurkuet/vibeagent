"""Qdrant Skill for vector storage and retrieval.

This skill provides vector database operations using Qdrant for semantic search
and retrieval-augmented generation (RAG) applications.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document stored in vector database."""

    id: str
    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class QdrantSkill(BaseSkill):
    """Skill for vector database operations using Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "knowledge_base",
        vector_size: int = 1536,
        distance: str = "Cosine",
        embedding_model: str | None = None,
    ):
        """Initialize Qdrant skill.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key (optional)
            collection_name: Default collection name
            vector_size: Size of embedding vectors
            distance: Distance metric (Cosine, Euclid, Dot)
            embedding_model: Name of embedding model to use
        """
        super().__init__(
            name="qdrant",
            version="1.0.0",
            description="Vector database operations with Qdrant",
        )
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.embedding_model = embedding_model
        self._client = None
        self._embedding_model = None

    def _get_client(self) -> Any:
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                self._client = QdrantClient(url=self.url, api_key=self.api_key)

                # Create collection if it doesn't exist
                self._ensure_collection()

            except ImportError:
                raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")
        return self._client

    def _get_embedding_model(self) -> Any:
        """Get or create embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.embedding_model or "all-MiniLM-L6-v2"
                self._embedding_model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "Sentence transformers not installed. Install with: pip install sentence-transformers"
                )
        return self._embedding_model

    def _ensure_collection(self):
        """Ensure collection exists."""
        from qdrant_client.models import CollectionInfo, Distance, VectorParams

        try:
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE if self.distance == "Cosine" else Distance.EUCLID,
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")

    def validate(self) -> bool:
        """Validate skill configuration."""
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def get_tool_schema(self) -> dict[str, Any]:
        """Get OpenAI function schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "Search for similar documents in the vector database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 10,
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name to search",
                        },
                        "filter": {
                            "type": "object",
                            "description": "Metadata filters for search",
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
        collection: str | None = None,
        filter: dict | None = None,
        ids: list[str] | None = None,
        **kwargs,
    ) -> SkillResult:
        """Execute Qdrant skill.

        Args:
            action: Action to perform (search, insert, delete, update)
            query: Search query text
            documents: Documents to insert
            limit: Number of results
            collection: Collection name
            filter: Metadata filters
            ids: Document IDs to delete

        Returns:
            SkillResult with operation results
        """
        if not self.validate():
            return SkillResult(success=False, error="Invalid configuration")

        collection_name = collection or self.collection_name

        if action == "search":
            return self._search(query, limit, collection_name, filter, **kwargs)
        elif action == "insert":
            return self._insert(documents, collection_name, **kwargs)
        elif action == "delete":
            return self._delete(ids, collection_name, **kwargs)
        elif action == "update":
            return self._update(documents, collection_name, **kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        model = self._get_embedding_model()
        return model.encode(text, convert_to_numpy=False).tolist()

    def _search(
        self,
        query: str,
        limit: int,
        collection_name: str,
        filter: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Search for similar documents.

        Args:
            query: Search query
            limit: Number of results
            collection_name: Collection to search
            filter: Metadata filters

        Returns:
            SkillResult with search results
        """
        try:
            from qdrant_client.models import Filter, PointStruct, SearchRequest

            client = self._get_client()
            query_vector = self._generate_embedding(query)

            search_filter = None
            if filter:
                # Convert filter dict to Qdrant Filter
                # This is simplified - real implementation would need proper filter conversion
                pass

            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
            )

            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "content": result.payload.get("content", ""),
                        "metadata": result.payload.get("metadata", {}),
                    }
                )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "query": query,
                    "results": formatted_results,
                    "total": len(formatted_results),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error searching: {e}")
            return SkillResult(success=False, error=f"Search failed: {str(e)}")

    def _insert(self, documents: list[dict], collection_name: str, **kwargs) -> SkillResult:
        """Insert documents into collection.

        Args:
            documents: List of documents to insert
            collection_name: Collection name

        Returns:
            SkillResult with insert results
        """
        if not documents:
            return SkillResult(success=False, error="No documents to insert")

        try:
            from qdrant_client.models import PointStruct

            client = self._get_client()

            points = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                doc_id = doc.get("id", f"{datetime.now().timestamp()}-{i}")

                embedding = self._generate_embedding(content)

                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "metadata": metadata,
                        "created_at": datetime.now().isoformat(),
                    },
                )
                points.append(point)

            operation_info = client.upsert(
                collection_name=collection_name,
                points=points,
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "operation_id": operation_info.operation_id,
                    "status": operation_info.status,
                    "inserted": len(points),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error inserting documents: {e}")
            return SkillResult(success=False, error=f"Insert failed: {str(e)}")

    def _delete(self, ids: list[str] | None, collection_name: str, **kwargs) -> SkillResult:
        """Delete documents from collection.

        Args:
            ids: Document IDs to delete
            collection_name: Collection name

        Returns:
            SkillResult with delete results
        """
        if not ids:
            return SkillResult(success=False, error="No IDs provided")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = self._get_client()

            # Delete by IDs
            delete_filter = Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=id)) for id in ids]
            )

            operation_info = client.delete(
                collection_name=collection_name,
                points_selector=delete_filter,
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "operation_id": operation_info.operation_id,
                    "status": operation_info.status,
                    "deleted": len(ids),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error deleting documents: {e}")
            return SkillResult(success=False, error=f"Delete failed: {str(e)}")

    def _update(self, documents: list[dict], collection_name: str, **kwargs) -> SkillResult:
        """Update documents in collection.

        Args:
            documents: Documents to update
            collection_name: Collection name

        Returns:
            SkillResult with update results
        """
        if not documents:
            return SkillResult(success=False, error="No documents to update")

        try:
            from qdrant_client.models import PointStruct

            client = self._get_client()

            points = []
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                doc_id = doc.get("id")

                if not doc_id:
                    continue

                embedding = self._generate_embedding(content)

                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "metadata": metadata,
                        "updated_at": datetime.now().isoformat(),
                    },
                )
                points.append(point)

            operation_info = client.upsert(
                collection_name=collection_name,
                points=points,
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "operation_id": operation_info.operation_id,
                    "status": operation_info.status,
                    "updated": len(points),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error updating documents: {e}")
            return SkillResult(success=False, error=f"Update failed: {str(e)}")

    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception:
            return False