"""PostgreSQL Skill with pgvector for vector storage and knowledge graph.

This skill provides unified vector storage and knowledge graph operations
using PostgreSQL with the pgvector extension, replacing both Qdrant and Neo4j.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PGVectorDocument:
    """Document stored in PostgreSQL with pgvector."""

    id: str
    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PGGEntity:
    """Entity in the knowledge graph."""

    id: str
    name: str
    label: str
    properties: dict = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PGGRelationship:
    """Relationship between entities."""

    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PostgreSQLSkill:
    """Skill for PostgreSQL with pgvector operations."""

    def __init__(
        self,
        connection_string: str = "postgresql://agentzero@localhost:5432/vibeagent",
        vector_size: int = 1024,
        embedding_model: str = "bge-m3",
        embedding_url: str = "http://localhost:11434",
        use_ollama_embeddings: bool = True,
    ):
        """Initialize PostgreSQL skill.

        Args:
            connection_string: PostgreSQL connection string
            vector_size: Size of embedding vectors (default 1024 for bge-m3)
            embedding_model: Name of embedding model
            embedding_url: URL for embedding API (Ollama)
            use_ollama_embeddings: Use Ollama API for embeddings
        """
        self.name = "postgresql"
        self.version = "1.0.0"
        self.connection_string = connection_string
        self.vector_size = vector_size
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url.rstrip("/")
        self.use_ollama_embeddings = use_ollama_embeddings
        self._conn = None
        self._embedding_model = None

        self.status = "active"
        self.usage_count = 0
        self.error_count = 0
        self.last_used = None

        logger.info("PostgreSQLSkill initialized")

    def _get_connection(self) -> Any:
        """Get or create PostgreSQL connection."""
        if self._conn is None:
            try:
                import psycopg

                self._conn = psycopg.connect(self.connection_string, autocommit=True)
                self._ensure_tables()
            except ImportError:
                raise ImportError(
                    "psycopg not installed. Install with: pip install psycopg"
                )
        return self._conn

    def _ensure_tables(self):
        """Ensure required tables and extensions exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table for vector storage
            vector_dim = self.vector_size
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({vector_dim}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create entities table for knowledge graph
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS entities (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    label VARCHAR(100) NOT NULL,
                    properties JSONB DEFAULT '{{}}',
                    embedding vector({vector_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create relationships table for knowledge graph
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id VARCHAR(255) PRIMARY KEY,
                    source_id VARCHAR(255) NOT NULL,
                    target_id VARCHAR(255) NOT NULL,
                    relationship_type VARCHAR(100) NOT NULL,
                    properties JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
                );
            """,)

            # Create indexes for vector similarity search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_embedding
                ON entities USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

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
        deps = ["psycopg"]
        if not self.use_ollama_embeddings:
            deps.append("sentence-transformers")
        return deps

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
                embedding = result.get("embedding", [])
                return embedding
            except Exception as e:
                logger.error(f"Ollama embedding error: {e}")
                raise
        return []

    def _record_usage(self):
        """Record that the skill was used."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

    def _record_error(self):
        """Record an error."""
        self.error_count += 1
        self.status = "error"

    # Vector storage operations
    def upsert_document(self, document: dict) -> dict:
        """Upsert a document with embedding.

        Args:
            document: Document dict with id, content, metadata

        Returns:
            Result dict
        """
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            doc_id = document.get("id")
            content = document.get("content", "")
            metadata = document.get("metadata", {})

            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata

            # Generate embedding
            embedding = self._generate_embedding(content)

            # Upsert document
            cursor.execute("""
                INSERT INTO documents (id, content, embedding, metadata, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP;
            """, (doc_id, content, list(embedding), metadata_json))

            conn.commit()
            cursor.close()

            self._record_usage()
            return {"success": True, "id": doc_id, "embedding_size": len(embedding)}

        except Exception as e:
            self._record_error()
            logger.error(f"Error upserting document: {e}")
            return {"success": False, "error": str(e)}

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        filter_dict: dict | None = None,
    ) -> dict:
        """Search documents by vector similarity.

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

            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Build query
            sql = """
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents
            """

            params = [str(query_embedding)]

            # Add metadata filters if provided
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
                metadata = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                documents.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": metadata,
                    "score": float(row[3]),
                })

            self._record_usage()
            return {"success": True, "results": documents, "count": len(documents)}

        except Exception as e:
            self._record_error()
            logger.error(f"Error searching documents: {e}")
            return {"success": False, "error": str(e)}

    # Knowledge graph operations
    def upsert_entity(self, entity: dict) -> dict:
        """Upsert an entity in the knowledge graph.

        Args:
            entity: Entity dict with id, name, label, properties

        Returns:
            Result dict
        """
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            entity_id = entity.get("id")
            name = entity.get("name", "")
            label = entity.get("label", "Entity")
            properties = entity.get("properties", {})

            # Convert properties to JSON string
            properties_json = json.dumps(properties) if isinstance(properties, dict) else properties

            # Generate embedding for entity name and properties
            entity_text = f"{name} {label} {properties.get('description', '')}"
            embedding = self._generate_embedding(entity_text)

            # Upsert entity
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
            self._record_error()
            logger.error(f"Error upserting entity: {e}")
            return {"success": False, "error": str(e)}

    def upsert_relationship(self, relationship: dict) -> dict:
        """Upsert a relationship in the knowledge graph.

        Args:
            relationship: Relationship dict with id, source_id, target_id, relationship_type

        Returns:
            Result dict
        """
        try:
            import json

            conn = self._get_connection()
            cursor = conn.cursor()

            rel_id = relationship.get("id")
            source_id = relationship.get("source_id")
            target_id = relationship.get("target_id")
            rel_type = relationship.get("relationship_type", "RELATED_TO")
            properties = relationship.get("properties", {})

            # Convert properties to JSON string
            properties_json = json.dumps(properties) if isinstance(properties, dict) else properties

            # Upsert relationship
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
            self._record_error()
            logger.error(f"Error upserting relationship: {e}")
            return {"success": False, "error": str(e)}

    def query_graph(self, cypher_query: str) -> dict:
        """Execute a graph-like query.

        Args:
            cypher_query: Query string (simplified Cypher-like syntax)

        Returns:
            Result dict
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Convert simple Cypher-like queries to SQL
            # This is a simplified implementation
            if cypher_query.startswith("MATCH"):
                # Parse basic MATCH queries
                if "RETURN" in cypher_query:
                    parts = cypher_query.split("RETURN")
                    match_part = parts[0].replace("MATCH", "").strip()
                    return_part = parts[1].strip()

                    if "(e)" in match_part:
                        # Query entities
                        cursor.execute(f"""
                            SELECT id, name, label, properties
                            FROM entities
                            LIMIT 100;
                        """)
                        results = cursor.fetchall()
                        entities = [
                            {"id": r[0], "name": r[1], "label": r[2], "properties": r[3]}
                            for r in results
                        ]
                        cursor.close()
                        return {"success": True, "entities": entities}

            # Default: execute as raw SQL
            cursor.execute(cypher_query)
            results = cursor.fetchall()
            cursor.close()

            self._record_usage()
            return {"success": True, "results": results}

        except Exception as e:
            self._record_error()
            logger.error(f"Error executing query: {e}")
            return {"success": False, "error": str(e)}

    def get_entity_relationships(self, entity_id: str) -> dict:
        """Get all relationships for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            Result dict with relationships
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT r.id, r.source_id, r.target_id, r.relationship_type, r.properties,
                       e1.name as source_name, e2.name as target_name
                FROM relationships r
                JOIN entities e1 ON r.source_id = e1.id
                JOIN entities e2 ON r.target_id = e2.id
                WHERE r.source_id = %s OR r.target_id = %s;
            """, (entity_id, entity_id))

            results = cursor.fetchall()
            cursor.close()

            relationships = [
                {
                    "id": r[0],
                    "source_id": r[1],
                    "target_id": r[2],
                    "relationship_type": r[3],
                    "properties": r[4],
                    "source_name": r[5],
                    "target_name": r[6],
                }
                for r in results
            ]

            self._record_usage()
            return {"success": True, "relationships": relationships, "count": len(relationships)}

        except Exception as e:
            self._record_error()
            logger.error(f"Error getting relationships: {e}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Statistics dict
        """
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