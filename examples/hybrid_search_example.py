"""Hybrid Search Example with RRF.

This example demonstrates the hybrid search system that combines
vector search, keyword search, and graph-based search using
Reciprocal Rank Fusion (RRF) for optimal results.
"""

import os

from vibeagent.core import HybridSearchOrchestrator
from vibeagent.skills import ElasticsearchSkill, LLMSkill, Neo4jSkill, QdrantSkill


def main():
    """Run the hybrid search example."""
    print("=" * 80)
    print("Hybrid Search with RRF Example")
    print("=" * 80)

    # Configuration
    openai_api_url = "http://localhost:8087/v1"
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    embedding_url = "http://localhost:11434/"
    embedding_model = "bge-m3"
    embedding_vector_size = 1024

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # Initialize skills
    print("\n1. Initializing search skills...")

    # Vector database (Qdrant)
    qdrant_skill = QdrantSkill(
        url=qdrant_url,
        collection_name="knowledge_base",
        vector_size=embedding_vector_size,
        embedding_url=embedding_url,
        embedding_model=embedding_model,
    )
    print("   Qdrant (Vector Search) initialized")

    # Keyword search (Elasticsearch)
    elasticsearch_skill = ElasticsearchSkill(
        hosts=elasticsearch_url,
        index_name="knowledge_base",
    )
    print("   Elasticsearch (Keyword Search) initialized")

    # Knowledge graph (Neo4j)
    neo4j_skill = Neo4jSkill(
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
    )
    print("   Neo4j (Graph Search) initialized")

    # Initialize hybrid search orchestrator
    print("\n2. Initializing Hybrid Search Orchestrator...")
    hybrid_search = HybridSearchOrchestrator(
        vector_skill=qdrant_skill,
        keyword_skill=elasticsearch_skill,
        graph_skill=neo4j_skill,
        rrf_k=60,  # RRF constant
        vector_weight=1.0,
        keyword_weight=0.5,
        graph_weight=0.3,
    )
    print("   Hybrid search initialized")

    # Example: Add sample documents
    print("\n3. Adding sample documents...")

    sample_documents = [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"category": "ai", "author": "AI Researcher"},
        },
        {
            "id": "doc2",
            "content": "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
            "metadata": {"category": "ai", "author": "DL Expert"},
        },
        {
            "id": "doc3",
            "content": "Natural language processing enables computers to understand and generate human language.",
            "metadata": {"category": "nlp", "author": "NLP Specialist"},
        },
        {
            "id": "doc4",
            "content": "Computer vision allows machines to interpret and understand visual information from images.",
            "metadata": {"category": "cv", "author": "CV Researcher"},
        },
        {
            "id": "doc5",
            "content": "Reinforcement learning trains agents to make decisions through trial and error.",
            "metadata": {"category": "ai", "author": "RL Expert"},
        },
    ]

    # Store in vector database
    qdrant_result = qdrant_skill.execute(
        action="upsert",
        documents=sample_documents,
    )
    print(f"   Vector storage: {qdrant_result.success}")

    # Store in Elasticsearch
    es_result = elasticsearch_skill.execute(
        action="bulk_index",
        documents=sample_documents,
    )
    print(f"   Keyword storage: {es_result.success}")

    # Store in knowledge graph
    entities = [
        {
            "id": "ml",
            "name": "Machine Learning",
            "label": "Technology",
            "properties": {"type": "AI", "applications": ["classification", "regression"]},
        },
        {
            "id": "dl",
            "name": "Deep Learning",
            "label": "Technology",
            "properties": {"type": "AI", "architecture": "neural_networks"},
        },
        {
            "id": "nlp",
            "name": "Natural Language Processing",
            "label": "Technology",
            "properties": {"type": "AI", "modality": "text"},
        },
    ]

    relationships = [
        {
            "from_entity": "dl",
            "to_entity": "ml",
            "relationship_type": "IS_A",
        },
    ]

    neo4j_result = neo4j_skill.execute(
        action="insert_entity",
        entities=entities,
        relationships=relationships,
    )
    print(f"   Graph storage: {neo4j_result.success}")

    # Example: Vector-only search
    print("\n" + "-" * 80)
    print("Example 1: Vector-Only Search")
    print("-" * 80)

    vector_result = hybrid_search.search(
        query="neural networks and deep learning",
        limit=3,
        sources=["vector"],
    )

    if vector_result.success:
        results = vector_result.data.get("results", [])
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('title', result.get('id', 'No title'))}")
            print(f"      Score: {result.get('score', 0):.3f}")
            print(f"      Sources: {result.get('sources', [])}")

    # Example: Keyword-only search
    print("\n" + "-" * 80)
    print("Example 2: Keyword-Only Search")
    print("-" * 80)

    keyword_result = hybrid_search.search(
        query="artificial intelligence machine learning",
        limit=3,
        sources=["keyword"],
    )

    if keyword_result.success:
        results = keyword_result.data.get("results", [])
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('title', result.get('id', 'No title'))}")
            print(f"      Score: {result.get('score', 0):.3f}")
            print(f"      Sources: {result.get('sources', [])}")

    # Example: Hybrid search (vector + keyword)
    print("\n" + "-" * 80)
    print("Example 3: Hybrid Search (Vector + Keyword)")
    print("-" * 80)

    hybrid_result = hybrid_search.search(
        query="neural networks deep learning",
        limit=3,
        sources=["vector", "keyword"],
    )

    if hybrid_result.success:
        results = hybrid_result.data.get("results", [])
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('title', result.get('id', 'No title'))}")
            print(f"      RRF Score: {result.get('score', 0):.3f}")
            print(f"      Vector Rank: {result.get('rank_vector', 'N/A')}")
            print(f"      Keyword Rank: {result.get('rank_keyword', 'N/A')}")
            print(f"      Sources: {result.get('sources', [])}")

    # Example: Full hybrid search (vector + keyword + graph)
    print("\n" + "-" * 80)
    print("Example 4: Full Hybrid Search (Vector + Keyword + Graph)")
    print("-" * 80)

    full_result = hybrid_search.search(
        query="machine learning techniques",
        limit=5,
        sources=["vector", "keyword", "graph"],
    )

    if full_result.success:
        results = full_result.data.get("results", [])
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('title', result.get('id', 'No title'))}")
            print(f"      RRF Score: {result.get('score', 0):.3f}")
            print(f"      Vector Rank: {result.get('rank_vector', 'N/A')}")
            print(f"      Keyword Rank: {result.get('rank_keyword', 'N/A')}")
            print(f"      Graph Rank: {result.get('rank_graph', 'N/A')}")
            print(f"      Sources: {result.get('sources', [])}")

    # Example: Search with filters
    print("\n" + "-" * 80)
    print("Example 5: Search with Metadata Filters")
    print("-" * 80)

    filter_result = hybrid_search.search(
        query="learning",
        limit=3,
        sources=["vector", "keyword"],
        filter={"category": "ai"},
    )

    if filter_result.success:
        results = filter_result.data.get("results", [])
        print(f"   Found {len(results)} results with category='ai'")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('title', result.get('id', 'No title'))}")
            print(f"      Score: {result.get('score', 0):.3f}")
            print(f"      Metadata: {result.get('metadata', {})}")

    print("\n" + "=" * 80)
    print("Hybrid Search Example Complete")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("  - Vector search for semantic similarity")
    print("  - Keyword search for exact matches")
    print("  - Graph search for relationship-based retrieval")
    print("  - Reciprocal Rank Fusion (RRF) for result combination")
    print("  - Configurable source weights")
    print("  - Metadata filtering")
    print("=" * 80)


if __name__ == "__main__":
    main()
