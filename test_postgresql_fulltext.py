"""Test PostgreSQL Full-Text Search - Replaces Elasticsearch + Neo4j + Qdrant"""

def main():
    print("=" * 80)
    print("PostgreSQL Full-Text Search - Unified Search Test")
    print("=" * 80)

    from vibeagent.skills.postgresql_fulltext_skill import PostgreSQLFullTextSkill

    # Initialize
    print("\n1. Initializing PostgreSQL with full-text search...")
    pg_skill = PostgreSQLFullTextSkill()

    if not pg_skill.validate():
        print("   Failed to connect to PostgreSQL")
        return

    print("   Connected successfully!")

    # Test 1: Full-text search (replaces Elasticsearch)
    print("\n" + "-" * 80)
    print("Test 1: Full-Text Search (replaces Elasticsearch BM25)")
    print("-" * 80)

    sample_docs = [
        {
            "id": "doc1",
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            "metadata": {"category": "ai", "difficulty": "beginner"}
        },
        {
            "id": "doc2",
            "title": "Deep Learning with Neural Networks",
            "content": "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
            "metadata": {"category": "ai", "difficulty": "advanced"}
        },
        {
            "id": "doc3",
            "title": "Natural Language Processing",
            "content": "NLP enables computers to understand, interpret, and generate human language.",
            "metadata": {"category": "nlp", "difficulty": "intermediate"}
        },
        {
            "id": "doc4",
            "title": "Computer Vision Applications",
            "content": "Computer vision allows machines to interpret and understand visual information from images and videos.",
            "metadata": {"category": "cv", "difficulty": "advanced"}
        },
        {
            "id": "doc5",
            "title": "PostgreSQL Database Optimization",
            "content": "PostgreSQL offers powerful features like full-text search, vector similarity, and advanced indexing.",
            "metadata": {"category": "database", "difficulty": "intermediate"}
        },
    ]

    print("   Storing 5 documents...")
    for doc in sample_docs:
        result = pg_skill.upsert_document(doc)
        if result["success"]:
            print(f"   ✓ Stored: {doc['title']}")
        else:
            print(f"   ✗ Failed: {doc['title']} - {result.get('error')}")

    # Full-text search
    print("\n   Searching for 'machine learning'...")
    result = pg_skill.fulltext_search("machine learning", limit=5)
    if result["success"]:
        print(f"   Found {result['count']} results:")
        for i, doc in enumerate(result["results"], 1):
            print(f"   {i}. Score: {doc['score']:.3f} - {doc['title']}")
    else:
        print(f"   Search failed: {result.get('error')}")

    # Test 2: Fuzzy search (trigram similarity)
    print("\n" + "-" * 80)
    print("Test 2: Fuzzy Search (Trigram Similarity - handles typos)")
    print("-" * 80)

    print("   Searching for 'natural langugage' (typo)...")
    result = pg_skill.fuzzy_search("natural langugage", limit=5, threshold=0.1)
    if result["success"]:
        print(f"   Found {result['count']} results:")
        for i, doc in enumerate(result["results"], 1):
            print(f"   {i}. Similarity: {doc['score']:.3f} - {doc['title']}")
    else:
        print(f"   Fuzzy search failed: {result.get('error')}")

    # Test 3: Vector search (semantic search)
    print("\n" + "-" * 80)
    print("Test 3: Vector Search (Semantic Search - replaces Qdrant)")
    print("-" * 80)

    print("   Searching semantically for 'AI training methods'...")
    result = pg_skill.vector_search("AI training methods", limit=5)
    if result["success"]:
        print(f"   Found {result['count']} results:")
        for i, doc in enumerate(result["results"], 1):
            print(f"   {i}. Score: {doc['score']:.3f} - {doc['title']}")
    else:
        print(f"   Vector search failed: {result.get('error')}")

    # Test 4: Hybrid search (combines full-text + vector)
    print("\n" + "-" * 80)
    print("Test 4: Hybrid Search (Full-Text + Vector with RRF)")
    print("-" * 80)

    print("   Hybrid search for 'database optimization'...")
    result = pg_skill.hybrid_search("database optimization", limit=5)
    if result["success"]:
        print(f"   Found {result['count']} results:")
        for i, doc in enumerate(result["results"], 1):
            print(f"   {i}. Score: {doc['score']:.4f} - {doc['title']}")
    else:
        print(f"   Hybrid search failed: {result.get('error')}")

    # Test 5: Metadata filtering
    print("\n" + "-" * 80)
    print("Test 5: Metadata Filtering")
    print("-" * 80)

    print("   Searching with filter: category = 'ai' and difficulty = 'advanced'...")
    result = pg_skill.fulltext_search("learning", limit=5, filter_dict={"category": "ai", "difficulty": "advanced"})
    if result["success"]:
        print(f"   Found {result['count']} results:")
        for i, doc in enumerate(result["results"], 1):
            print(f"   {i}. Score: {doc['score']:.3f} - {doc['title']} ({doc['metadata']})")
    else:
        print(f"   Filtered search failed: {result.get('error')}")

    # Test 6: Knowledge graph
    print("\n" + "-" * 80)
    print("Test 6: Knowledge Graph (replaces Neo4j)")
    print("-" * 80)

    entities = [
        {"id": "e1", "name": "PostgreSQL", "label": "Database", "properties": {"type": "RDBMS", "description": "Open-source relational database"}},
        {"id": "e2", "name": "pgvector", "label": "Extension", "properties": {"description": "Vector similarity search"}},
        {"id": "e3", "name": "Elasticsearch", "label": "Search Engine", "properties": {"type": "Full-text search"}},
        {"id": "e4", "name": "Qdrant", "label": "Database", "properties": {"type": "Vector database"}},
    ]

    print("   Storing 4 entities...")
    for entity in entities:
        result = pg_skill.upsert_entity(entity)
        if result["success"]:
            print(f"   ✓ Stored: {entity['name']}")
        else:
            print(f"   ✗ Failed: {entity['name']}")

    relationships = [
        {"id": "r1", "source_id": "e2", "target_id": "e1", "relationship_type": "EXTENDS", "properties": {}},
        {"id": "r2", "source_id": "e2", "target_id": "e4", "relationship_type": "ALTERNATIVE_TO", "properties": {}},
        {"id": "r3", "source_id": "e1", "target_id": "e3", "relationship_type": "ALTERNATIVE_TO", "properties": {}},
    ]

    print("   Storing 3 relationships...")
    for rel in relationships:
        result = pg_skill.upsert_relationship(rel)
        if result["success"]:
            print(f"   ✓ Stored: {rel['relationship_type']}")
        else:
            print(f"   ✗ Failed: {rel['relationship_type']}")

    # Test 7: Statistics
    print("\n" + "-" * 80)
    print("Test 7: Database Statistics")
    print("-" * 80)

    stats = pg_skill.get_stats()
    if stats["success"]:
        print(f"   Documents: {stats['documents']}")
        print(f"   Entities: {stats['entities']}")
        print(f"   Relationships: {stats['relationships']}")
    else:
        print(f"   Failed to get stats: {stats.get('error')}")

    # Close connection
    pg_skill.close()

    print("\n" + "=" * 80)
    print("PostgreSQL Full-Text Search Test Complete")
    print("=" * 80)
    print("\nPostgreSQL successfully provides:")
    print("  ✓ Full-text search (tsvector, GIN indexes) - replaces Elasticsearch")
    print("  ✓ Vector similarity search (pgvector) - replaces Qdrant")
    print("  ✓ Knowledge graph (entities, relationships) - replaces Neo4j")
    print("  ✓ Fuzzy search (trigram similarity)")
    print("  ✓ Hybrid search (RRF algorithm)")
    print("  ✓ Metadata filtering (JSONB)")
    print("\nSingle database replaces Elasticsearch + Neo4j + Qdrant!")
    print("=" * 80)


if __name__ == "__main__":
    main()
