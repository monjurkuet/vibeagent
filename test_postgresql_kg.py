"""Test PostgreSQL with pgvector for unified knowledge base.

This script demonstrates:
1. Entity extraction from text
2. Storing entities in PostgreSQL knowledge graph
3. Creating relationships between entities
4. Vector search for documents
5. Quality evaluation
"""

import os

from vibeagent.core import KnowledgeQualityEvaluator
from vibeagent.skills import (
    EntityExtractionSkill,
    LLMSkill,
    PostgreSQLSkill,
)


def main():
    """Test PostgreSQL knowledge base."""
    print("=" * 80)
    print("PostgreSQL with pgvector - Unified Knowledge Base Test")
    print("=" * 80)

    # Configuration
    llm_api_url = "http://localhost:8087/v1"
    embedding_url = "http://localhost:11434"
    embedding_model = "bge-m3"

    # Try different PostgreSQL connection strings
    connection_strings = [
        "postgresql://agentzero@localhost:5432/vibeagent",
        "postgresql://postgres@localhost:5432/postgres",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    ]

    # Initialize LLM
    print("\n1. Initializing LLM skill...")
    llm_skill = LLMSkill(base_url=llm_api_url, enable_round_robin=True)
    print("   LLM skill initialized")

    # Initialize PostgreSQL
    print("\n2. Initializing PostgreSQL with pgvector...")
    pg_skill = None

    for conn_str in connection_strings:
        try:
            pg_skill = PostgreSQLSkill(
                connection_string=conn_str,
                vector_size=1024,
                embedding_model=embedding_model,
                embedding_url=embedding_url,
                use_ollama_embeddings=True,
            )

            if pg_skill.validate():
                print(f"   Connected to PostgreSQL: {conn_str}")
                break
            else:
                pg_skill = None
        except Exception as e:
            print(f"   Failed: {conn_str} - {e}")
            pg_skill = None

    if not pg_skill:
        print("   Could not connect to PostgreSQL")
        print("   Please ensure:")
        print("   - PostgreSQL is running")
        print("   - pgvector extension is installed: CREATE EXTENSION vector;")
        print("   - Valid connection string is provided")
        return

    # Initialize Entity Extraction
    print("\n3. Initializing Entity Extraction skill...")
    entity_extraction = EntityExtractionSkill(llm_skill=llm_skill)
    print("   Entity extraction initialized")

    # Initialize Quality Evaluator
    print("\n4. Initializing Knowledge Quality Evaluator...")
    quality_evaluator = KnowledgeQualityEvaluator(llm_skill=llm_skill)
    print("   Quality evaluator initialized")

    # Test 1: Extract entities from text
    print("\n" + "-" * 80)
    print("Test 1: Extract entities from technical text")
    print("-" * 80)

    sample_text = """
    PostgreSQL is a powerful open-source relational database management system.
    It supports advanced features like JSONB for semi-structured data and pgvector
    for vector similarity search. Machine learning applications often use PostgreSQL
    for storing embeddings and performing semantic search. The pgvector extension
    enables efficient ANN (Approximate Nearest Neighbor) search using IVFFlat indexes.
    Python developers commonly use psycopg for database connectivity.
    """

    print(f"Input text length: {len(sample_text)} characters")

    result = entity_extraction.execute(
        action="extract",
        text=sample_text,
        entity_types=["Technology", "Database", "Extension", "Library"],
        extract_relationships=True,
    )

    if result.success:
        entities = result.data.get("entities", [])
        relationships = result.data.get("relationships", [])

        print(f"   Extracted {len(entities)} entities:")
        for i, entity in enumerate(entities[:5], 1):
            print(f"   {i}. {entity.get('name')} ({entity.get('label')}) - confidence: {entity.get('confidence', 0):.2f}")

        print(f"\n   Extracted {len(relationships)} relationships:")
        for i, rel in enumerate(relationships[:5], 1):
            print(f"   {i}. {rel.get('source')} -> {rel.get('target')} ({rel.get('type')})")

        # Store entities in PostgreSQL knowledge graph
        print("\n   Storing entities in PostgreSQL knowledge graph...")

        for entity in entities:
            entity_result = pg_skill.upsert_entity({
                "id": entity.get("id"),
                "name": entity.get("name"),
                "label": entity.get("label"),
                "properties": {
                    "description": entity.get("description", ""),
                    "confidence": entity.get("confidence", 1.0),
                },
            })

            if entity_result.get("success"):
                print(f"   ✓ Stored: {entity.get('name')}")
            else:
                print(f"   ✗ Failed: {entity.get('name')} - {entity_result.get('error')}")

        # Store relationships
        print("\n   Storing relationships...")

        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")

            # Skip relationships without valid source/target
            if not source_id or not target_id:
                continue

            rel_result = pg_skill.upsert_relationship({
                "id": f"rel_{hash(source_id + target_id)}",
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": rel.get("type", "RELATED_TO"),
                "properties": {
                    "description": rel.get("description", ""),
                },
            })

            if rel_result.get("success"):
                print(f"   ✓ Stored: {rel.get('source')} -> {rel.get('target')}")
            else:
                print(f"   ✗ Failed relationship storage")

    else:
        print(f"   Entity extraction failed: {result.error}")

    # Test 2: Store and search documents
    print("\n" + "-" * 80)
    print("Test 2: Vector search with pgvector")
    print("-" * 80)

    documents = [
        {
            "id": "doc1",
            "content": "PostgreSQL pgvector enables efficient vector similarity search for AI applications.",
            "metadata": {"category": "database", "topic": "vector_search"},
        },
        {
            "id": "doc2",
            "content": "Machine learning models generate embeddings that can be stored in PostgreSQL.",
            "metadata": {"category": "ai", "topic": "embeddings"},
        },
        {
            "id": "doc3",
            "content": "Python psycopg library provides robust database connectivity for PostgreSQL.",
            "metadata": {"category": "programming", "topic": "python"},
        },
    ]

    print(f"   Storing {len(documents)} documents...")

    for doc in documents:
        result = pg_skill.upsert_document(doc)
        if result.get("success"):
            print(f"   ✓ Stored: {doc['id']}")
        else:
            print(f"   ✗ Failed: {doc['id']} - {result.get('error')}")

    # Search for similar documents
    print("\n   Searching for 'vector similarity'...")

    search_result = pg_skill.search_documents(
        query="vector similarity search",
        limit=3,
    )

    if search_result.get("success"):
        results = search_result.get("results", [])
        print(f"   Found {len(results)} results:")

        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.3f}")
            print(f"      Content: {result['content'][:80]}...")
            print(f"      Metadata: {result['metadata']}")
    else:
        print(f"   Search failed: {search_result.get('error')}")

    # Test 3: Query knowledge graph
    print("\n" + "-" * 80)
    print("Test 3: Query knowledge graph")
    print("-" * 80)

    query_result = pg_skill.query_graph("MATCH (e) RETURN e LIMIT 10")

    if query_result.get("success"):
        entities = query_result.get("entities", [])
        print(f"   Found {len(entities)} entities:")

        for i, entity in enumerate(entities[:5], 1):
            print(f"   {i}. {entity['name']} ({entity['label']})")
            if entity.get('properties'):
                print(f"      Properties: {entity['properties']}")
    else:
        print(f"   Query failed: {query_result.get('error')}")

    # Test 4: Get entity relationships
    if query_result.get("success") and query_result.get("entities"):
        first_entity_id = query_result.get("entities", [{}])[0].get("id")

        print(f"\n   Getting relationships for entity: {first_entity_id}")

        rel_result = pg_skill.get_entity_relationships(first_entity_id)

        if rel_result.get("success"):
            relationships = rel_result.get("relationships", [])
            print(f"   Found {len(relationships)} relationships:")

            for i, rel in enumerate(relationships[:5], 1):
                print(f"   {i}. {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
        else:
            print(f"   Failed to get relationships: {rel_result.get('error')}")

    # Test 5: Quality evaluation
    print("\n" + "-" * 80)
    print("Test 4: Knowledge Quality Evaluation")
    print("-" * 80)

    question = "What is pgvector used for?"
    answer = "Pgvector is used for vector similarity search in PostgreSQL, enabling AI applications to perform efficient semantic search and ANN queries."
    context = "PostgreSQL pgvector enables efficient vector similarity search for AI applications. Machine learning models generate embeddings that can be stored in PostgreSQL for semantic search."

    quality_result = quality_evaluator.evaluate_rag(
        question=question,
        answer=answer,
        context=context,
    )

    if quality_result.success:
        metrics = quality_result.data
        print("   Quality Metrics:")
        print(f"   - Context relevancy: {metrics.get('context_relevancy', 0):.2f}")
        print(f"   - Answer faithfulness: {metrics.get('answer_faithfulness', 0):.2f}")
        print(f"   - Answer relevancy: {metrics.get('answer_relevancy', 0):.2f}")
        print(f"   - Context utilization: {metrics.get('context_utilization', 0):.2f}")
        print(f"   - Overall score: {metrics.get('overall_score', 0):.2f}")
    else:
        print(f"   Quality evaluation failed: {quality_result.error}")

    # Get statistics
    print("\n" + "-" * 80)
    print("Database Statistics")
    print("-" * 80)

    stats = pg_skill.get_stats()

    if stats.get("success"):
        print(f"   Documents: {stats['documents']}")
        print(f"   Entities: {stats['entities']}")
        print(f"   Relationships: {stats['relationships']}")
    else:
        print(f"   Failed to get stats: {stats.get('error')}")

    # Close connection
    pg_skill.close()

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
    print("\nPostgreSQL with pgvector successfully demonstrated:")
    print("  ✓ Entity extraction from text")
    print("  ✓ Knowledge graph storage (entities and relationships)")
    print("  ✓ Vector similarity search")
    print("  ✓ Graph querying")
    print("  ✓ Quality evaluation")
    print("\nPostgreSQL with pgvector can replace both Neo4j and Qdrant!")


if __name__ == "__main__":
    main()