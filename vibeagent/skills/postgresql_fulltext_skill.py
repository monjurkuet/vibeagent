"""PostgreSQL Full-Text Search Skill with pgvector.

This skill provides unified search capabilities using PostgreSQL:
- Full-text search (tsvector, GIN indexes) - replaces Elasticsearch
- Vector similarity search (pgvector) - replaces Qdrant
- Knowledge graph (entities/relationships) - replaces Neo4j
- Trigram similarity for fuzzy matching
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class PostgreSQLFullTextSkill:
    """Skill for PostgreSQL with full-text search, vector search, and knowledge graph."""

    def __init__(
        self,
        connection_string: str = "postgresql://postgres:postgres@localhost:5432/vibeagent",
        vector_size: int = 1024,
        embedding_model: str = "bge-m3",
        embedding_url: str = "http://localhost:11434",
        use_ollama_embeddings: bool = True,
    ):
        """Initialize PostgreSQL Full-Text Search skill.

        Args:
            connection_string: PostgreSQL connection string
            vector_size: Size of embedding vectors
            embedding_model: Name of embedding model
            embedding_url: URL for embedding API (Ollama)
            use_ollama_embeddings: Use Ollama API for embeddings
        """
        self.name = "postgresql_fulltext"
        self.version = "2.0.0"
        self.connection_string = connection_string
        self.vector_size = vector_size
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url.rstrip("/")
        self.use_ollama_embeddings = use_ollama_embeddings
        self._conn = None

        self.status = "active"
        self.usage_count = 0
        self.last_used = None

        logger.info("PostgreSQLFullTextSkill initialized")

    def _get_connection(self) -> Any:
        """Get or create PostgreSQL connection."""
        if self._conn is None:
            try:
                import psycopg

                self._conn = psycopg.connect(self.connection_string, autocommit=True)
                self._ensure_tables()
            except ImportError:
                raise ImportError("psycopg not installed. Install with: pip install psycopg")
        return self._conn

    def _ensure_tables(self):
        """Ensure required tables, indexes, and extensions exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Enable extensions
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # pg_trigram may not be available on all systems
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trigram;")
            except Exception as e:
                logger.warning(f"pg_trigram extension not available (fuzzy search disabled): {e}")

            # Drop and recreate documents table with full-text search support
            vector_dim = self.vector_size
            cursor.execute(f"""
                DROP TABLE IF EXISTS documents CASCADE;
                CREATE TABLE documents (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    title TEXT,
                    embedding vector({vector_dim}),
                    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Drop and recreate entities table
            cursor.execute(f"""
                DROP TABLE IF EXISTS entities CASCADE;
                CREATE TABLE entities (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    label VARCHAR(100) NOT NULL,
                    properties JSONB DEFAULT '{{}}',
                    embedding vector({vector_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Drop and recreate relationships table
            cursor.execute("""
                DROP TABLE IF EXISTS relationships CASCADE;
                CREATE TABLE relationships (
                    id VARCHAR(255) PRIMARY KEY,
                    source_id VARCHAR(255) NOT NULL,
                    target_id VARCHAR(255) NOT NULL,
                    relationship_type VARCHAR(100) NOT NULL,
                    properties JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
                );
            """)

            # Create indexes for vector similarity search
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_entities_embedding
                ON entities USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            # Create indexes for full-text search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_tsv
                ON documents USING GIN (tsv);
            """)

            # Create trigram indexes for fuzzy search (only if pg_trigram is available)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_content_trgm
                    ON documents USING GIN (content gin_trgm_ops);
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_title_trgm
                    ON documents USING GIN (title gin_trgm_ops);
                """)
            except Exception:
                logger.warning("Trigram indexes not created (pg_trigram unavailable)")

            # Create GIN indexes for metadata search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_metadata
                ON documents USING GIN (metadata);
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_properties
                ON entities USING GIN (properties);
            """)

            conn.commit()
            logger.info("PostgreSQL tables and indexes created successfully")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
        finally:
            cursor.close()

    def validate(self) -> bool:
        """Validate skill configuration."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.close()
            return True
        except Exception:
            return False

    def get_dependencies(self) -> list[str]:
        """Return list of dependencies."""
        return ["psycopg"]

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama."""
        if self.use_ollama_embeddings:
            try:
                import requests

                response = requests.post(
                    f"{self.embedding_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("embedding", [])
            except Exception as e:
                logger.error(f"Ollama embedding error: {e}")
                raise
        return []

    def _record_usage(self):
        """Record that the skill was used."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

    # === Document Storage ===
    def upsert_document(self, document: dict) -> dict:
        """Upsert a document with embedding and full-text search."""
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            doc_id = document.get("id")
            content = document.get("content", "")
            title = document.get("title", "")
            metadata = document.get("metadata", {})

            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
            embedding = self._generate_embedding(title + " " + content)

            cursor.execute("""
                INSERT INTO documents (id, content, title, embedding, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    title = EXCLUDED.title,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP;
            """, (doc_id, content, title, list(embedding), metadata_json))

            conn.commit()
            cursor.close()

            self._record_usage()
            return {"success": True, "id": doc_id, "embedding_size": len(embedding)}

        except Exception as e:
            logger.error(f"Error upserting document: {e}")
            return {"success": False, "error": str(e)}

    # === Full-Text Search (replaces Elasticsearch) ===
    def fulltext_search(
        self,
        query: str,
        limit: int = 10,
        fields: list[str] | None = None,
        filter_dict: dict | None = None,
    ) -> dict:
        """Perform full-text search using PostgreSQL tsvector.

        This replaces Elasticsearch's BM25 search.

        Args:
            query: Search query text
            limit: Number of results
            fields: Fields to search (title, content, or both)
            filter_dict: Metadata filters

        Returns:
            Result dict with matching documents
        """
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            # Build query with ts_rank for BM25-like ranking
            sql = """
                SELECT id, title, content, metadata,
                       ts_rank(tsv, plainto_tsquery('english', %s)) as rank
                FROM documents
                WHERE tsv @@ plainto_tsquery('english', %s)
            """

            params = [query, query]

            # Add metadata filters if provided
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(f"metadata->>'{key}' = %s")
                    params.append(value)

                if conditions:
                    sql += " AND " + " AND ".join(conditions)

            sql += " ORDER BY rank DESC LIMIT %s;"
            params.append(limit)

            cursor.execute(sql, params)
            results = cursor.fetchall()
            cursor.close()

            documents = []
            for row in results:
                metadata = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                documents.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "metadata": metadata,
                    "score": float(row[4]),
                })

            self._record_usage()
            return {"success": True, "results": documents, "count": len(documents)}

        except Exception as e:
            logger.error(f"Error in full-text search: {e}")
            return {"success": False, "error": str(e)}

    # === Fuzzy Search (Trigram Similarity) ===
    def fuzzy_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> dict:
        """Perform fuzzy search using trigram similarity.

        Good for typos and partial matches.

        Args:
            query: Search query
            limit: Number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            Result dict with matching documents
        """
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, title, content, metadata,
                       similarity(content, %s) as sim_score
                FROM documents
                WHERE content % %s
                ORDER BY sim_score DESC
                LIMIT %s;
            """, (query, query, limit))

            results = cursor.fetchall()
            cursor.close()

            documents = []
            for row in results:
                metadata = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                if row[4] >= threshold:
                    documents.append({
                        "id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "metadata": metadata,
                        "score": float(row[4]),
                    })

            self._record_usage()
            return {"success": True, "results": documents, "count": len(documents)}

        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return {"success": False, "error": str(e)}

    # === Vector Search (Semantic Search) ===
    def vector_search(
        self,
        query: str,
        limit: int = 10,
        filter_dict: dict | None = None,
    ) -> dict:
        """Perform vector similarity search using pgvector.

        Args:
            query: Search query text
            limit: Number of results
            filter_dict: Metadata filters

        Returns:
            Result dict with matching documents
        """
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            query_embedding = self._generate_embedding(query)

            sql = """
                SELECT id, title, content, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents
            """

            params = [str(query_embedding)]

            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(f"metadata->>'{key}' = %s")
                    params.append(value)

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

            sql += " ORDER BY embedding <=> %s::vector LIMIT %s;"
            params.extend([str(query_embedding), limit])

            cursor.execute(sql, params)
            results = cursor.fetchall()
            cursor.close()

            documents = []
            for row in results:
                metadata = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                documents.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "metadata": metadata,
                    "score": float(row[4]),
                })

            self._record_usage()
            return {"success": True, "results": documents, "count": len(documents)}

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return {"success": False, "error": str(e)}

    # === Hybrid Search (Full-Text + Vector) ===
    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        fulltext_weight: float = 0.5,
        vector_weight: float = 0.5,
        filter_dict: dict | None = None,
    ) -> dict:
        """Perform hybrid search combining full-text and vector search.

        Uses weighted scoring similar to RRF (Reciprocal Rank Fusion).

        Args:
            query: Search query
            limit: Number of results
            fulltext_weight: Weight for full-text search (0-1)
            vector_weight: Weight for vector search (0-1)
            filter_dict: Metadata filters

        Returns:
            Result dict with combined results
        """
        try:
            # Get full-text results
            ft_result = self.fulltext_search(query, limit * 2, filter_dict=filter_dict)
            ft_docs = {doc["id"]: {"rank": i, "score": doc["score"]} for i, doc in enumerate(ft_result.get("results", []))}

            # Get vector results
            v_result = self.vector_search(query, limit * 2, filter_dict=filter_dict)
            v_docs = {doc["id"]: {"rank": i, "score": doc["score"]} for i, doc in enumerate(v_result.get("results", []))}

            # Combine using RRF
            combined_scores = {}
            all_ids = set(ft_docs.keys()) | set(v_docs.keys())

            k = 60  # RRF constant
            for doc_id in all_ids:
                rrf_score = 0.0

                if doc_id in ft_docs:
                    rrf_score += fulltext_weight / (k + ft_docs[doc_id]["rank"])

                if doc_id in v_docs:
                    rrf_score += vector_weight / (k + v_docs[doc_id]["rank"])

                combined_scores[doc_id] = rrf_score

            # Sort by combined score and get top results
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

            # Fetch full documents
            results = []
            for doc_id, score in sorted_results:
                doc = next((d for d in ft_result.get("results", []) + v_result.get("results", []) if d["id"] == doc_id), None)
                if doc:
                    results.append({**doc, "score": score})

            self._record_usage()
            return {"success": True, "results": results, "count": len(results)}

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {"success": False, "error": str(e)}

    # === Knowledge Graph Operations ===
    def upsert_entity(self, entity: dict) -> dict:
        """Upsert an entity in the knowledge graph."""
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            entity_id = entity.get("id")
            name = entity.get("name", "")
            label = entity.get("label", "Entity")
            properties = entity.get("properties", {})

            properties_json = json.dumps(properties) if isinstance(properties, dict) else properties
            entity_text = f"{name} {label} {properties.get('description', '')}"
            embedding = self._generate_embedding(entity_text)

            cursor.execute("""
                INSERT INTO entities (id, name, label, properties, embedding, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    label = EXCLUDED.label,
                    properties = EXCLUDED.properties,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP;
            """, (entity_id, name, label, properties_json, list(embedding)))

            conn.commit()
            cursor.close()

            self._record_usage()
            return {"success": True, "id": entity_id}

        except Exception as e:
            logger.error(f"Error upserting entity: {e}")
            return {"success": False, "error": str(e)}

    def upsert_relationship(self, relationship: dict) -> dict:
        """Upsert a relationship in the knowledge graph."""
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            rel_id = relationship.get("id")
            source_id = relationship.get("source_id")
            target_id = relationship.get("target_id")
            rel_type = relationship.get("relationship_type", "RELATED_TO")
            properties = relationship.get("properties", {})

            properties_json = json.dumps(properties) if isinstance(properties, dict) else properties

            cursor.execute("""
                INSERT INTO relationships (id, source_id, target_id, relationship_type, properties)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    relationship_type = EXCLUDED.relationship_type,
                    properties = EXCLUDED.properties;
            """, (rel_id, source_id, target_id, rel_type, properties_json))

            conn.commit()
            cursor.close()

            self._record_usage()
            return {"success": True, "id": rel_id}

        except Exception as e:
            logger.error(f"Error upserting relationship: {e}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM entities;")
            entity_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM relationships;")
            rel_count = cursor.fetchone()[0]

            cursor.close()

            return {
                "success": True,
                "documents": doc_count,
                "entities": entity_count,
                "relationships": rel_count,
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"success": False, "error": str(e)}

    def health_check(self) -> bool:
        """Check if skill is operational."""
        return self.validate()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("PostgreSQL connection closed")