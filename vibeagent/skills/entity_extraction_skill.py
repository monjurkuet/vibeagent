"""Entity Extraction Skill for knowledge graph construction.

This skill uses LLM to extract entities, relationships, and concepts from text
for building and enriching knowledge graphs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Entity extracted from text."""

    id: str
    name: str
    label: str
    description: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ExtractedRelationship:
    """Relationship extracted from text."""

    source_id: str
    target_id: str
    relationship_type: str
    description: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0


class EntityExtractionSkill(BaseSkill):
    """Skill for extracting entities and relationships from text using LLM."""

    def __init__(
        self,
        llm_skill: Any | None = None,
        model: str = "gpt-4",
        min_confidence: float = 0.7,
    ):
        """Initialize Entity Extraction skill.

        Args:
            llm_skill: LLMSkill instance for extraction
            model: Model name to use
            min_confidence: Minimum confidence threshold
        """
        super().__init__(
            name="entity_extraction",
            version="1.0.0",
            description="Extract entities and relationships for knowledge graphs",
        )
        self.llm_skill = llm_skill
        self.model = model
        self.min_confidence = min_confidence

    def validate(self) -> bool:
        """Validate skill configuration."""
        return True

    def get_tool_schema(self) -> dict[str, Any]:
        """Get OpenAI function schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities and relationships from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract from",
                        },
                        "entity_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of entities to extract",
                        },
                    },
                    "required": ["text"],
                },
            },
        }

    def execute(
        self,
        action: str = "extract",
        text: str | None = None,
        documents: list[dict] | None = None,
        entity_types: list[str] | None = None,
        extract_relationships: bool = True,
        **kwargs,
    ) -> SkillResult:
        """Execute Entity Extraction skill.

        Args:
            action: Action to perform (extract, extract_from_documents)
            text: Text to extract from
            documents: Documents to extract from
            entity_types: Types of entities to extract
            extract_relationships: Whether to extract relationships

        Returns:
            SkillResult with extracted entities and relationships
        """
        if not text and not documents:
            return SkillResult(success=False, error="No text or documents provided")

        if action == "extract":
            return self._extract_from_text(text, entity_types, extract_relationships, **kwargs)
        elif action == "extract_from_documents":
            return self._extract_from_documents(documents, entity_types, extract_relationships, **kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _extract_from_text(
        self,
        text: str,
        entity_types: list[str] | None,
        extract_relationships: bool,
        **kwargs,
    ) -> SkillResult:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: Types of entities to extract
            extract_relationships: Whether to extract relationships

        Returns:
            SkillResult with extracted data
        """
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(text, entity_types, extract_relationships)

            # Call LLM
            if self.llm_skill:
                llm_result = self.llm_skill.execute(
                    prompt=prompt,
                    system_prompt="You are an expert at extracting entities and relationships from text. Respond only with valid JSON.",
                    max_tokens=4000,
                )
                if not llm_result.success:
                    return SkillResult(success=False, error=llm_result.error)
                response_text = llm_result.data.get("content", "")
            else:
                return SkillResult(success=False, error="LLM skill not provided")

            # Parse JSON response
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re

                json_match = re.search(r"\{[\s\S]*\}", response_text)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                else:
                    return SkillResult(success=False, error="Failed to parse LLM response")

            # Process entities
            entities = []
            for entity_data in extracted_data.get("entities", []):
                entity = ExtractedEntity(
                    id=self._generate_entity_id(entity_data.get("name", "")),
                    name=entity_data.get("name", ""),
                    label=entity_data.get("label", "Entity"),
                    description=entity_data.get("description", ""),
                    properties=entity_data.get("properties", {}),
                    confidence=entity_data.get("confidence", 1.0),
                )
                if entity.confidence >= self.min_confidence:
                    entities.append(entity)

            # Process relationships
            relationships = []
            if extract_relationships:
                for rel_data in extracted_data.get("relationships", []):
                    relationship = ExtractedRelationship(
                        source_id=self._generate_entity_id(rel_data.get("source", "")),
                        target_id=self._generate_entity_id(rel_data.get("target", "")),
                        relationship_type=rel_data.get("type", "RELATED_TO"),
                        description=rel_data.get("description", ""),
                        properties=rel_data.get("properties", {}),
                        confidence=rel_data.get("confidence", 1.0),
                    )
                    if relationship.confidence >= self.min_confidence:
                        relationships.append(relationship)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "text": text,
                    "entities": [self._entity_to_dict(e) for e in entities],
                    "relationships": [self._relationship_to_dict(r) for r in relationships],
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                },
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Error extracting entities: {e}")
            return SkillResult(success=False, error=f"Extraction failed: {str(e)}")

    def _extract_from_documents(
        self,
        documents: list[dict] | None,
        entity_types: list[str] | None,
        extract_relationships: bool,
        **kwargs,
    ) -> SkillResult:
        """Extract entities from multiple documents.

        Args:
            documents: Documents to extract from
            entity_types: Types of entities to extract
            extract_relationships: Whether to extract relationships

        Returns:
            SkillResult with extracted data
        """
        if not documents:
            return SkillResult(success=False, error="No documents provided")

        all_entities = {}
        all_relationships = []

        for doc in documents:
            text = doc.get("content", "")
            doc_id = doc.get("id", "")

            result = self._extract_from_text(text, entity_types, extract_relationships, **kwargs)

            if result.success:
                # Add document ID to entities
                for entity in result.data.get("entities", []):
                    entity_id = entity["id"]
                    if entity_id not in all_entities:
                        all_entities[entity_id] = entity
                    # Track which documents mention this entity
                    if "document_ids" not in all_entities[entity_id]:
                        all_entities[entity_id]["document_ids"] = []
                    if doc_id not in all_entities[entity_id]["document_ids"]:
                        all_entities[entity_id]["document_ids"].append(doc_id)

                # Track relationships
                for rel in result.data.get("relationships", []):
                    all_relationships.append(rel)

        self._record_usage()
        return SkillResult(
            success=True,
            data={
                "documents_processed": len(documents),
                "entities": list(all_entities.values()),
                "relationships": all_relationships,
                "entity_count": len(all_entities),
                "relationship_count": len(all_relationships),
            },
        )

    def _build_extraction_prompt(
        self,
        text: str,
        entity_types: list[str] | None,
        extract_relationships: bool,
    ) -> str:
        """Build extraction prompt for LLM.

        Args:
            text: Text to extract from
            entity_types: Types of entities to extract
            extract_relationships: Whether to extract relationships

        Returns:
            Extraction prompt
        """
        entity_types_str = ", ".join(entity_types) if entity_types else "Person, Organization, Location, Concept, Event"

        prompt = f"""Extract entities and relationships from the following text.

Text:
{text}

Extract entities of these types: {entity_types_str}

For each entity, provide:
- name: The exact name of the entity
- label: The type/category of the entity
- description: A brief description
- properties: Any additional properties (as key-value pairs)
- confidence: Your confidence in this extraction (0.0 to 1.0)
"""

        if extract_relationships:
            prompt += """

Also extract relationships between entities:
- source: The source entity name
- target: The target entity name
- type: The type of relationship (e.g., "WORKS_AT", "LOCATED_IN", "PART_OF", "RELATED_TO")
- description: A brief description
- properties: Any additional properties
- confidence: Your confidence in this extraction (0.0 to 1.0)
"""

        prompt += """

Respond with valid JSON in this format:
{
  "entities": [
    {
      "name": "Entity Name",
      "label": "EntityType",
      "description": "Description",
      "properties": {"key": "value"},
      "confidence": 0.95
    }
  ],
  "relationships": [
    {
      "source": "Entity1",
      "target": "Entity2",
      "type": "RELATIONSHIP_TYPE",
      "description": "Description",
      "properties": {"key": "value"},
      "confidence": 0.90
    }
  ]
}

Do not include any explanation outside the JSON. Respond only with the JSON object."""

        return prompt

    def _generate_entity_id(self, name: str) -> str:
        """Generate a unique entity ID from name.

        Args:
            name: Entity name

        Returns:
            Entity ID
        """
        import hashlib

        name_normalized = name.lower().replace(" ", "_").replace("-", "_")
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name_normalized}_{hash_suffix}"

    def _entity_to_dict(self, entity: ExtractedEntity) -> dict:
        """Convert entity to dictionary.

        Args:
            entity: ExtractedEntity object

        Returns:
            Dictionary representation
        """
        return {
            "id": entity.id,
            "name": entity.name,
            "label": entity.label,
            "description": entity.description,
            "properties": entity.properties,
            "confidence": entity.confidence,
        }

    def _relationship_to_dict(self, relationship: ExtractedRelationship) -> dict:
        """Convert relationship to dictionary.

        Args:
            relationship: ExtractedRelationship object

        Returns:
            Dictionary representation
        """
        return {
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "relationship_type": relationship.relationship_type,
            "description": relationship.description,
            "properties": relationship.properties,
            "confidence": relationship.confidence,
        }

    def health_check(self) -> bool:
        """Check if skill is operational."""
        return self.llm_skill is not None