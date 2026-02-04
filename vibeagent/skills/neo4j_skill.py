"""Neo4j Skill for knowledge graph operations.

This skill provides graph database operations using Neo4j for storing
entities, relationships, and building knowledge graphs for GraphRAG.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity in the knowledge graph."""

    id: str
    label: str
    properties: dict = field(default_factory=dict)
    embeddings: list[float] | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Relationship:
    """Relationship between entities."""

    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class Neo4jSkill(BaseSkill):
    """Skill for knowledge graph operations using Neo4j."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str | None = None,
        database: str = "neo4j",
    ):
        """Initialize Neo4j skill.

        Args:
            uri: Neo4j server URI
            username: Database username
            password: Database password
            database: Database name
        """
        super().__init__(
            name="neo4j",
            version="1.0.0",
            description="Knowledge graph operations with Neo4j",
        )
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    def _get_driver(self) -> Any:
        """Get or create Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                if not self.password:
                    import os

                    self.password = os.getenv("NEO4J_PASSWORD", "")

                self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            except ImportError:
                raise ImportError("Neo4j driver not installed. Install with: pip install neo4j")
        return self._driver

    def _ensure_constraints(self):
        """Ensure graph constraints exist."""
        try:
            with self._get_driver().session(database=self.database) as session:
                # Create uniqueness constraints for common entity types
                constraints = [
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX document_url IF NOT EXISTS FOR (d:Document) ON (d.url)",
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception:
                        # Constraint might already exist
                        pass
        except Exception as e:
            logger.error(f"Error ensuring constraints: {e}")

    def validate(self) -> bool:
        """Validate skill configuration."""
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            self._ensure_constraints()
            return True
        except Exception:
            return False

    def get_tool_schema(self) -> dict[str, Any]:
        """Get OpenAI function schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": "graph_search",
                "description": "Search the knowledge graph for entities and relationships",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Cypher query or natural language query",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Type of entity to search for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results",
                            "default": 10,
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
        cypher: str | None = None,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        limit: int = 10,
        **kwargs,
    ) -> SkillResult:
        """Execute Neo4j skill.

        Args:
            action: Action to perform (search, insert_entity, insert_relationship, execute_cypher)
            query: Natural language query
            cypher: Cypher query
            entities: Entities to insert
            relationships: Relationships to insert
            limit: Number of results

        Returns:
            SkillResult with operation results
        """
        if not self.validate():
            return SkillResult(success=False, error="Invalid configuration")

        if action == "search":
            return self._search(query, limit, **kwargs)
        elif action == "insert_entity":
            return self._insert_entity(entities, **kwargs)
        elif action == "insert_relationship":
            return self._insert_relationship(relationships, **kwargs)
        elif action == "execute_cypher":
            return self._execute_cypher(cypher, **kwargs)
        elif action == "get_neighbors":
            return self._get_neighbors(query, limit, **kwargs)
        elif action == "find_path":
            return self._find_path(query, **kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _search(self, query: str, limit: int, **kwargs) -> SkillResult:
        """Search for entities and relationships.

        Args:
            query: Search query
            limit: Number of results

        Returns:
            SkillResult with search results
        """
        try:
            with self._get_driver().session(database=self.database) as session:
                # Simple full-text search on entities
                cypher = """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query OR e.description CONTAINS $query
                RETURN e
                LIMIT $limit
                """

                result = session.run(cypher, query=query, limit=limit)

                entities = []
                for record in result:
                    entity_node = record["e"]
                    entities.append(
                        {
                            "id": entity_node.get("id"),
                            "label": entity_node.labels[0] if entity_node.labels else "Entity",
                            "name": entity_node.get("name"),
                            "description": entity_node.get("description"),
                            "properties": dict(entity_node),
                        }
                    )

                self._record_usage()
                return SkillResult(
                    success=True,
                    data={
                        "query": query,
                        "entities": entities,
                        "total": len(entities),
                    },
                )

        except Exception as e:
            self._record_error()
            logger.error(f"Error searching graph: {e}")
            return SkillResult(success=False, error=f"Search failed: {str(e)}")

    def _insert_entity(self, entities: list[dict] | None, **kwargs) -> SkillResult:
        """Insert entities into the graph.

        Args:
            entities: Entities to insert

        Returns:
            SkillResult with insert results
        """
        if not entities:
            return SkillResult(success=False, error="No entities to insert")

        try:
            with self._get_driver().session(database=self.database) as session:
                inserted_count = 0

                for entity in entities:
                    entity_id = entity.get("id")
                    label = entity.get("label", "Entity")
                    properties = entity.get("properties", {})

                    cypher = f"""
                    MERGE (e:{label} {{id: $id}})
                    SET e += $properties
                    SET e.updated_at = $updated_at
                    RETURN e
                    """

                    try:
                        session.run(
                            cypher,
                            id=entity_id,
                            properties=properties,
                            updated_at=datetime.now().isoformat(),
                        )
                        inserted_count += 1
                    except Exception as e:
                        logger.error(f"Error inserting entity {entity_id}: {e}")

                self._record_usage()
                return SkillResult(
                    success=True,
                    data={
                        "inserted": inserted_count,
                        "total": len(entities),
                    },
                )

        except Exception as e:
            self._record_error()
            logger.error(f"Error inserting entities: {e}")
            return SkillResult(success=False, error=f"Insert failed: {str(e)}")

    def _insert_relationship(self, relationships: list[dict] | None, **kwargs) -> SkillResult:
        """Insert relationships into the graph.

        Args:
            relationships: Relationships to insert

        Returns:
            SkillResult with insert results
        """
        if not relationships:
            return SkillResult(success=False, error="No relationships to insert")

        try:
            with self._get_driver().session(database=self.database) as session:
                inserted_count = 0

                for rel in relationships:
                    source_id = rel.get("source_id")
                    target_id = rel.get("target_id")
                    rel_type = rel.get("relationship_type", "RELATED_TO")
                    properties = rel.get("properties", {})

                    cypher = f"""
                    MATCH (source), (target)
                    WHERE source.id = $source_id AND target.id = $target_id
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r += $properties
                    SET r.updated_at = $updated_at
                    RETURN r
                    """

                    try:
                        session.run(
                            cypher,
                            source_id=source_id,
                            target_id=target_id,
                            properties=properties,
                            updated_at=datetime.now().isoformat(),
                        )
                        inserted_count += 1
                    except Exception as e:
                        logger.error(f"Error inserting relationship {source_id}->{target_id}: {e}")

                self._record_usage()
                return SkillResult(
                    success=True,
                    data={
                        "inserted": inserted_count,
                        "total": len(relationships),
                    },
                )

        except Exception as e:
            self._record_error()
            logger.error(f"Error inserting relationships: {e}")
            return SkillResult(success=False, error=f"Insert failed: {str(e)}")

    def _execute_cypher(self, cypher: str, **kwargs) -> SkillResult:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query

        Returns:
            SkillResult with query results
        """
        if not cypher:
            return SkillResult(success=False, error="No Cypher query provided")

        try:
            with self._get_driver().session(database=self.database) as session:
                result = session.run(cypher)

                results = []
                for record in result:
                    results.append(dict(record))

                self._record_usage()
                return SkillResult(
                    success=True,
                    data={
                        "results": results,
                        "total": len(results),
                    },
                )

        except Exception as e:
            self._record_error()
            logger.error(f"Error executing Cypher: {e}")
            return SkillResult(success=False, error=f"Query failed: {str(e)}")

    def _get_neighbors(self, entity_id: str, limit: int, **kwargs) -> SkillResult:
        """Get neighbors of an entity.

        Args:
            entity_id: Entity ID
            limit: Number of results

        Returns:
            SkillResult with neighbors
        """
        try:
            with self._get_driver().session(database=self.database) as session:
                cypher = """
                MATCH (source:Entity {id: $entity_id})-[r]-(neighbor)
                RETURN type(r) as relationship, neighbor, properties(r) as rel_props
                LIMIT $limit
                """

                result = session.run(cypher, entity_id=entity_id, limit=limit)

                neighbors = []
                for record in result:
                    neighbor_node = record["neighbor"]
                    neighbors.append(
                        {
                            "relationship": record["relationship"],
                            "neighbor": {
                                "id": neighbor_node.get("id"),
                                "label": neighbor_node.labels[0] if neighbor_node.labels else "Entity",
                                "name": neighbor_node.get("name"),
                                "properties": dict(neighbor_node),
                            },
                            "relationship_properties": record["rel_props"],
                        }
                    )

                self._record_usage()
                return SkillResult(
                    success=True,
                    data={
                        "entity_id": entity_id,
                        "neighbors": neighbors,
                        "total": len(neighbors),
                    },
                )

        except Exception as e:
            self._record_error()
            logger.error(f"Error getting neighbors: {e}")
            return SkillResult(success=False, error=f"Get neighbors failed: {str(e)}")

    def _find_path(self, source_id: str, target_id: str | None = None, **kwargs) -> SkillResult:
        """Find path between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID (optional)

        Returns:
            SkillResult with path
        """
        try:
            with self._get_driver().session(database=self.database) as session:

                if target_id:
                    cypher = """
                    MATCH path = shortestPath((source:Entity {id: $source_id})-[*]-(target:Entity {id: $target_id}))
                    RETURN path
                    """
                    result = session.run(cypher, source_id=source_id, target_id=target_id)
                else:
                    cypher = """
                    MATCH (source:Entity {id: $source_id})-[*1..3]-(target:Entity)
                    RETURN path
                    LIMIT 5
                    """
                    result = session.run(cypher, source_id=source_id)

                paths = []
                for record in result:
                    path = record["path"]
                    nodes = [dict(node) for node in path.nodes]
                    relationships = [dict(rel) for rel in path.relationships]
                    paths.append({"nodes": nodes, "relationships": relationships})

                self._record_usage()
                return SkillResult(
                    success=True,
                    data={
                        "source_id": source_id,
                        "target_id": target_id,
                        "paths": paths,
                        "total": len(paths),
                    },
                )

        except Exception as e:
            self._record_error()
            logger.error(f"Error finding path: {e}")
            return SkillResult(success=False, error=f"Find path failed: {str(e)}")

    def health_check(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            with driver.session(database=self.database) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False