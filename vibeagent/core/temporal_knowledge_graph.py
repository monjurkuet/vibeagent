"""Temporal Knowledge Graph for tracking changes over time.

This system tracks the evolution of knowledge, allowing queries about
historical states and trends in the knowledge base.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class TemporalEntity:
    """Entity with temporal tracking."""

    entity_id: str
    label: str
    name: str
    description: str
    valid_from: str
    valid_to: str | None = None
    properties: dict = field(default_factory=dict)
    version: int = 1
    embedding: list[float] | None = None


@dataclass
class KnowledgeSnapshot:
    """Snapshot of knowledge at a point in time."""

    timestamp: str
    entity_count: int
    relationship_count: int
    metadata: dict = field(default_factory=dict)


class TemporalKnowledgeGraph:
    """Temporal knowledge graph system."""

    def __init__(self, neo4j_skill: Any | None = None, max_versions: int = 10):
        """Initialize Temporal Knowledge Graph.

        Args:
            neo4j_skill: Neo4j skill for graph storage
            max_versions: Maximum versions to keep per entity
        """
        self.neo4j_skill = neo4j_skill
        self.max_versions = max_versions
        self.snapshots: list[KnowledgeSnapshot] = []
        logger.info("TemporalKnowledgeGraph initialized")

    def create_snapshot(self, metadata: dict | None = None) -> SkillResult:
        """Create a snapshot of the current knowledge state.

        Args:
            metadata: Additional metadata for the snapshot

        Returns:
            SkillResult with snapshot information
        """
        try:
            if not self.neo4j_skill:
                return SkillResult(success=False, error="Neo4j skill not available")

            # Get current entity and relationship counts
            entity_count_result = self.neo4j_skill.execute(
                action="execute_cypher",
                cypher="MATCH (e:Entity) RETURN count(e) as count",
            )

            relationship_count_result = self.neo4j_skill.execute(
                action="execute_cypher",
                cypher="MATCH ()-[r]->() RETURN count(r) as count",
            )

            entity_count = entity_count_result.data.get("results", [{}])[0].get("count", 0)
            relationship_count = relationship_count_result.data.get("results", [{}])[0].get("count", 0)

            snapshot = KnowledgeSnapshot(
                timestamp=datetime.now().isoformat(),
                entity_count=entity_count,
                relationship_count=relationship_count,
                metadata=metadata or {},
            )

            self.snapshots.append(snapshot)

            # Store snapshot in graph
            self.neo4j_skill.execute(
                action="insert_entity",
                entities=[
                    {
                        "id": f"snapshot_{snapshot.timestamp}",
                        "label": "Snapshot",
                        "properties": {
                            "timestamp": snapshot.timestamp,
                            "entity_count": entity_count,
                            "relationship_count": relationship_count,
                            **(metadata or {}),
                        },
                    }
                ],
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "snapshot_id": f"snapshot_{snapshot.timestamp}",
                    "timestamp": snapshot.timestamp,
                    "entity_count": entity_count,
                    "relationship_count": relationship_count,
                    "total_snapshots": len(self.snapshots),
                },
            )

        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return SkillResult(success=False, error=f"Snapshot creation failed: {str(e)}")

    def get_state_at_time(self, timestamp: str) -> SkillResult:
        """Get knowledge state at a specific time.

        Args:
            timestamp: ISO format timestamp

        Returns:
            SkillResult with state information
        """
        try:
            if not self.neo4j_skill:
                return SkillResult(success=False, error="Neo4j skill not available")

            # Find closest snapshot before or at the timestamp
            target_time = datetime.fromisoformat(timestamp)
            closest_snapshot = None

            for snapshot in self.snapshots:
                snapshot_time = datetime.fromisoformat(snapshot.timestamp)
                if snapshot_time <= target_time:
                    if closest_snapshot is None or snapshot_time > datetime.fromisoformat(
                        closest_snapshot.timestamp
                    ):
                        closest_snapshot = snapshot

            if closest_snapshot:
                return SkillResult(
                    success=True,
                    data={
                        "target_timestamp": timestamp,
                        "closest_snapshot": closest_snapshot.timestamp,
                        "entity_count": closest_snapshot.entity_count,
                        "relationship_count": closest_snapshot.relationship_count,
                        "metadata": closest_snapshot.metadata,
                    },
                )
            else:
                return SkillResult(success=False, error="No snapshot found before the specified time")

        except Exception as e:
            logger.error(f"Error getting state at time: {e}")
            return SkillResult(success=False, error=f"State retrieval failed: {str(e)}")

    def track_entity_evolution(self, entity_id: str) -> SkillResult:
        """Track the evolution of an entity over time.

        Args:
            entity_id: Entity ID to track

        Returns:
            SkillResult with evolution history
        """
        try:
            if not self.neo4j_skill:
                return SkillResult(success=False, error="Neo4j skill not available")

            # Query entity versions from graph
            cypher = f"""
            MATCH (e:Entity {{id: $entity_id}})
            OPTIONAL MATCH (e)-[r:HAS_VERSION]->(v:EntityVersion)
            RETURN e, v
            ORDER BY v.version DESC
            """

            result = self.neo4j_skill.execute(action="execute_cypher", cypher=cypher, entity_id=entity_id)

            if result.success:
                records = result.data.get("results", [])
                evolution = []

                for record in records:
                    current = record.get("e", {})
                    version = record.get("v", {})

                    evolution.append(
                        {
                            "current": {
                                "name": current.get("name"),
                                "description": current.get("description"),
                                "properties": dict(current),
                            },
                            "version": {
                                "number": version.get("version"),
                                "valid_from": version.get("valid_from"),
                                "valid_to": version.get("valid_to"),
                                "properties": dict(version),
                            }
                            if version
                            else None,
                        }
                    )

                return SkillResult(
                    success=True,
                    data={
                        "entity_id": entity_id,
                        "evolution": evolution,
                        "total_versions": len([e for e in evolution if e["version"] is not None]),
                    },
                )

            return SkillResult(success=False, error="Entity not found")

        except Exception as e:
            logger.error(f"Error tracking entity evolution: {e}")
            return SkillResult(success=False, error=f"Evolution tracking failed: {str(e)}")

    def query_temporal(
        self, query: str, time_range: tuple[str, str] | None = None
    ) -> SkillResult:
        """Query knowledge graph with temporal constraints.

        Args:
            query: Query string
            time_range: Optional time range (start, end)

        Returns:
            SkillResult with temporal query results
        """
        try:
            if not self.neo4j_skill:
                return SkillResult(success=False, error="Neo4j skill not available")

            if time_range:
                start_time, end_time = time_range
                cypher = f"""
                MATCH (e:Entity)
                WHERE e.valid_from >= $start_time AND (e.valid_to IS NULL OR e.valid_to <= $end_time)
                RETURN e
                LIMIT 100
                """

                result = self.neo4j_skill.execute(
                    action="execute_cypher", cypher=cypher, start_time=start_time, end_time=end_time
                )
            else:
                # Return current state
                cypher = f"""
                MATCH (e:Entity)
                WHERE e.valid_to IS NULL OR e.valid_to > datetime()
                RETURN e
                LIMIT 100
                """

                result = self.neo4j_skill.execute(action="execute_cypher", cypher=cypher)

            if result.success:
                entities = result.data.get("results", [])
                return SkillResult(
                    success=True,
                    data={
                        "query": query,
                        "time_range": time_range,
                        "entities": entities,
                        "count": len(entities),
                    },
                )

            return SkillResult(success=False, error="Query failed")

        except Exception as e:
            logger.error(f"Error in temporal query: {e}")
            return SkillResult(success=False, error=f"Temporal query failed: {str(e)}")

    def update_entity(self, entity_id: str, updates: dict) -> SkillResult:
        """Update an entity with temporal tracking.

        Args:
            entity_id: Entity ID
            updates: Updates to apply

        Returns:
            SkillResult with update results
        """
        try:
            if not self.neo4j_skill:
                return SkillResult(success=False, error="Neo4j skill not available")

            now = datetime.now().isoformat()

            # Get current entity
            current_result = self.neo4j_skill.execute(
                action="execute_cypher",
                cypher=f"MATCH (e:Entity {{id: $entity_id}}) RETURN e",
                entity_id=entity_id,
            )

            if not current_result.success or not current_result.data.get("results"):
                return SkillResult(success=False, error="Entity not found")

            current_entity = current_result.data["results"][0]

            # Mark current version as expired
            self.neo4j_skill.execute(
                action="execute_cypher",
                cypher=f"""
                MATCH (e:Entity {{id: $entity_id}})
                SET e.valid_to = $now
                """,
                entity_id=entity_id,
                now=now,
            )

            # Create new version
            new_version = {
                **dict(current_entity),
                **updates,
                "valid_from": now,
                "valid_to": None,
                "version": current_entity.get("version", 0) + 1,
            }

            # Insert new entity
            self.neo4j_skill.execute(
                action="insert_entity",
                entities=[
                    {
                        "id": entity_id,
                        "label": new_version.get("label", "Entity"),
                        "properties": new_version,
                    }
                ],
            )

            # Link versions
            self.neo4j_skill.execute(
                action="execute_cypher",
                cypher=f"""
                MATCH (current:Entity {{id: $entity_id}}), (new:Entity {{id: $entity_id}})
                WHERE current.valid_to IS NOT NULL
                CREATE (current)-[:HAS_VERSION]->(new)
                """,
                entity_id=entity_id,
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "entity_id": entity_id,
                    "version": new_version["version"],
                    "valid_from": now,
                    "updates": updates,
                },
            )

        except Exception as e:
            logger.error(f"Error updating entity: {e}")
            return SkillResult(success=False, error=f"Update failed: {str(e)}")

    def get_trends(self, time_range: tuple[str, str] | None = None) -> SkillResult:
        """Get knowledge trends over time.

        Args:
            time_range: Time range for trend analysis

        Returns:
            SkillResult with trend data
        """
        try:
            if time_range:
                start_time, end_time = time_range
            else:
                # Default to last 30 days
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)
                time_range = (start_time.isoformat(), end_time.isoformat())

            # Get snapshots in range
            snapshots_in_range = [
                s
                for s in self.snapshots
                if start_time <= datetime.fromisoformat(s.timestamp) <= end_time
            ]

            if len(snapshots_in_range) < 2:
                return SkillResult(
                    success=False, error="Insufficient data for trend analysis (need at least 2 snapshots)"
                )

            # Calculate trends
            entity_growth = (
                snapshots_in_range[-1].entity_count - snapshots_in_range[0].entity_count
            )
            relationship_growth = (
                snapshots_in_range[-1].relationship_count - snapshots_in_range[0].relationship_count
            )

            return SkillResult(
                success=True,
                data={
                    "time_range": time_range,
                    "snapshots_analyzed": len(snapshots_in_range),
                    "entity_growth": entity_growth,
                    "relationship_growth": relationship_growth,
                    "entity_growth_rate": (
                        entity_growth / len(snapshots_in_range) if entity_growth != 0 else 0
                    ),
                    "relationship_growth_rate": (
                        relationship_growth / len(snapshots_in_range)
                        if relationship_growth != 0
                        else 0
                    ),
                    "snapshots": [
                        {"timestamp": s.timestamp, "entity_count": s.entity_count, "relationship_count": s.relationship_count}
                        for s in snapshots_in_range
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return SkillResult(success=False, error=f"Trend analysis failed: {str(e)}")

    def _record_usage(self):
        """Record usage statistics."""
        pass

    def get_stats(self) -> dict:
        """Get temporal graph statistics.

        Returns:
            Statistics
        """
        return {
            "total_snapshots": len(self.snapshots),
            "max_versions": self.max_versions,
            "neo4j_available": self.neo4j_skill is not None,
            "snapshot_range": (
                (self.snapshots[0].timestamp, self.snapshots[-1].timestamp)
                if self.snapshots
                else (None, None)
            ),
        }

    def health_check(self) -> bool:
        """Check if temporal graph is operational."""
        return self.neo4j_skill is not None and self.neo4j_skill.health_check()