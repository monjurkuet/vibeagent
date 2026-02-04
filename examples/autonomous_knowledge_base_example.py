"""Autonomous Knowledge Base Gathering Example.

This example demonstrates the complete autonomous knowledge base gathering system,
including:
- Web scraping with Firecrawl
- Vector storage and retrieval with Qdrant
- Keyword search with Elasticsearch
- Knowledge graph with Neo4j
- Hybrid search with RRF
- Autonomous research agent
- Continuous learning loop
- Multi-agent collaboration
"""

import os

from vibeagent.core import (
    AutonomousResearchAgent,
    ContinuousLearningLoop,
    HybridSearchOrchestrator,
    KnowledgeQualityEvaluator,
    MultiAgentCollaboration,
    TemporalKnowledgeGraph,
)
from vibeagent.skills import (
    ElasticsearchSkill,
    EntityExtractionSkill,
    FirecrawlSkill,
    LLMSkill,
    MultiModalSkill,
    Neo4jSkill,
    QdrantSkill,
)


def main():
    """Run the autonomous knowledge base example."""
    print("=" * 80)
    print("Autonomous Knowledge Base Gathering Example")
    print("=" * 80)

    # Configuration
    openai_api_url = "http://localhost:8087/v1"
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    embedding_url = "http://localhost:11434/"
    embedding_model = "bge-m3"
    embedding_vector_size = 1024

    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # Initialize base LLM skill
    print("\n1. Initializing LLM skill...")
    llm_skill = LLMSkill(
        base_url=openai_api_url,
        api_key=openai_api_key,
        model="gpt-4",
        timeout=60,
    )
    print("   LLM skill initialized")

    # Initialize skills
    print("\n2. Initializing autonomous knowledge base skills...")

    # Vector database (Qdrant)
    qdrant_skill = QdrantSkill(
        url=qdrant_url,
        collection_name="knowledge_base",
        vector_size=embedding_vector_size,
        embedding_url=embedding_url,
        embedding_model=embedding_model,
    )
    print("   - Qdrant (Vector Database) initialized")

    # Keyword search (Elasticsearch)
    elasticsearch_skill = ElasticsearchSkill(
        url=elasticsearch_url,
        index_name="knowledge_base",
    )
    print("   - Elasticsearch (Keyword Search) initialized")

    # Knowledge graph (Neo4j)
    neo4j_skill = Neo4jSkill(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    print("   - Neo4j (Knowledge Graph) initialized")

    # Web scraping (Firecrawl)
    if firecrawl_api_key:
        firecrawl_skill = FirecrawlSkill(
            api_key=firecrawl_api_key,
        )
        print("   - Firecrawl (Web Scraping) initialized")
    else:
        firecrawl_skill = None
        print("   - Firecrawl skipped (no API key)")

    # Entity extraction
    entity_extraction_skill = EntityExtractionSkill(
        llm_skill=llm_skill,
    )
    print("   - Entity Extraction initialized")

    # Multi-modal processing
    multimodal_skill = MultiModalSkill(
        embedding_url=embedding_url,
        embedding_model=embedding_model,
    )
    print("   - MultiModal processing initialized")

    # Hybrid search orchestrator
    print("\n3. Initializing Hybrid Search Orchestrator...")
    hybrid_search = HybridSearchOrchestrator(
        vector_skill=qdrant_skill,
        keyword_skill=elasticsearch_skill,
        graph_skill=neo4j_skill,
    )
    print("   Hybrid search initialized with RRF algorithm")

    # Autonomous research agent
    print("\n4. Initializing Autonomous Research Agent...")
    research_agent = AutonomousResearchAgent(
        name="research_agent",
        llm_skill=llm_skill,
        firecrawl_skill=firecrawl_skill,
        qdrant_skill=qdrant_skill,
        elasticsearch_skill=elasticsearch_skill,
    )
    print("   Research agent initialized")

    # Temporal knowledge graph
    print("\n5. Initializing Temporal Knowledge Graph...")
    temporal_graph = TemporalKnowledgeGraph(
        neo4j_skill=neo4j_skill,
    )
    print("   Temporal knowledge graph initialized")

    # Knowledge quality evaluator
    print("\n6. Initializing Knowledge Quality Evaluator...")
    quality_evaluator = KnowledgeQualityEvaluator(
        llm_skill=llm_skill,
    )
    print("   Quality evaluator initialized")

    # Continuous learning loop
    print("\n7. Initializing Continuous Learning Loop...")
    learning_loop = ContinuousLearningLoop(
        evaluator=quality_evaluator,
        vector_skill=qdrant_skill,
        keyword_skill=elasticsearch_skill,
        graph_skill=neo4j_skill,
    )
    print("   Learning loop initialized")

    # Multi-agent collaboration
    print("\n8. Initializing Multi-Agent Collaboration...")
    collaboration = MultiAgentCollaboration()
    collaboration.register_agent(research_agent, role="researcher")
    print("   Multi-agent collaboration initialized")

    # Example: Scrape a website
    if firecrawl_skill:
        print("\n" + "-" * 80)
        print("Example: Web Scraping with Firecrawl")
        print("-" * 80)

        result = firecrawl_skill.execute(
            action="scrape",
            url="https://example.com",
            formats=["markdown"],
        )

        if result.success:
            print(f"   Scraped content length: {len(result.data.get('markdown', ''))} characters")

            # Store in vector database
            print("\n   Storing in vector database...")
            qdrant_result = qdrant_skill.execute(
                action="upsert",
                documents=[{
                    "id": "example_com",
                    "content": result.data.get("markdown", ""),
                    "metadata": {
                        "source": "https://example.com",
                        "title": result.data.get("title", "Example"),
                    },
                }],
            )
            print(f"   Storage result: {qdrant_result.success}")

    # Example: Autonomous research
    print("\n" + "-" * 80)
    print("Example: Autonomous Research")
    print("-" * 80)

    research_result = research_agent.autonomous_research(
        topic="machine learning best practices",
        max_iterations=3,
    )

    if research_result.success:
        progress = research_result.data
        print(f"   Research completed:")
        print(f"   - Sources processed: {progress['sources_processed']}")
        print(f"   - Total iterations: {progress['total_iterations']}")
        print(f"   - Status: {progress['status']}")

    # Example: Hybrid search
    print("\n" + "-" * 80)
    print("Example: Hybrid Search with RRF")
    print("-" * 80)

    search_result = hybrid_search.search(
        query="machine learning algorithms",
        limit=5,
        sources=["vector", "keyword"],
    )

    if search_result.success:
        results = search_result.data.get("results", [])
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result.get('title', 'No title')} (Score: {result.get('score', 0):.3f})")

    # Example: Entity extraction
    print("\n" + "-" * 80)
    print("Example: Entity Extraction and Knowledge Graph")
    print("-" * 80)

    sample_text = """
    Machine learning is a subset of artificial intelligence (AI) that enables systems
    to learn and improve from experience. Neural networks are a key technique in
    deep learning, which is itself a branch of machine learning. Popular frameworks
    include TensorFlow and PyTorch.
    """

    entity_result = entity_extraction_skill.execute(
        action="extract_from_text",
        text=sample_text,
        entity_types=["Technology", "Framework", "Method"],
        extract_relationships=True,
    )

    if entity_result.success:
        entities = entity_result.data.get("entities", [])
        relationships = entity_result.data.get("relationships", [])
        print(f"   Extracted {len(entities)} entities")
        for entity in entities[:3]:
            print(f"   - {entity.get('name')} ({entity.get('label')})")
        print(f"   Extracted {len(relationships)} relationships")

        # Store in Neo4j
        neo4j_result = neo4j_skill.execute(
            action="insert_entity",
            entities=entities,
        )
        if neo4j_result.success:
            print("   Entities stored in knowledge graph")

    # Example: Temporal knowledge snapshot
    print("\n" + "-" * 80)
    print("Example: Temporal Knowledge Graph Snapshot")
    print("-" * 80)

    snapshot_result = temporal_graph.create_snapshot(
        metadata={"phase": "initial_ingestion"},
    )

    if snapshot_result.success:
        print(f"   Snapshot created: {snapshot_result.data['snapshot_id']}")
        print(f"   Entity count: {snapshot_result.data['entity_count']}")
        print(f"   Relationship count: {snapshot_result.data['relationship_count']}")

    # Example: Knowledge quality evaluation
    print("\n" + "-" * 80)
    print("Example: Knowledge Quality Evaluation")
    print("-" * 80)

    quality_result = quality_evaluator.evaluate_rag(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI that enables systems to learn from data.",
        context="Machine learning algorithms build models based on sample data.",
    )

    if quality_result.success:
        metrics = quality_result.data
        print(f"   Context relevancy: {metrics.get('context_relevancy', 0):.2f}")
        print(f"   Answer faithfulness: {metrics.get('answer_faithfulness', 0):.2f}")
        print(f"   Answer relevancy: {metrics.get('answer_relevancy', 0):.2f}")
        print(f"   Overall score: {metrics.get('overall_score', 0):.2f}")

    # Example: Multi-agent collaboration
    print("\n" + "-" * 80)
    print("Example: Multi-Agent Collaboration")
    print("-" * 80)

    session_result = collaboration.create_session(
        agent_ids=["research_agent"],
        task_description="Research AI ethics principles",
        shared_context={"domain": "artificial_intelligence"},
    )

    if session_result.success:
        print(f"   Session created: {session_result.data['session_id']}")

        task_result = collaboration.assign_task(
            session_id=session_result.data['session_id'],
            task_description="Find sources about AI ethics",
            agent_id="research_agent",
        )
        print(f"   Task assigned: {task_result.data['task_id']}")

    # Example: Continuous learning iteration
    print("\n" + "-" * 80)
    print("Example: Continuous Learning Loop")
    print("-" * 80)

    learning_result = learning_loop.run_learning_iteration()

    if learning_result.success:
        print(f"   Learning iteration completed")
        print(f"   Quality score: {learning_result.data.get('quality_score', 0):.2f}")
        improvements = learning_result.data.get('improvements', [])
        print(f"   Suggested improvements: {len(improvements)}")

    print("\n" + "=" * 80)
    print("Autonomous Knowledge Base Example Complete")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  - Web scraping with Firecrawl")
    print("  - Vector storage and semantic search with Qdrant")
    print("  - Keyword search with Elasticsearch")
    print("  - Knowledge graph with Neo4j")
    print("  - Hybrid search with Reciprocal Rank Fusion")
    print("  - Autonomous research agent")
    print("  - Entity extraction and relationship detection")
    print("  - Temporal knowledge tracking")
    print("  - Quality evaluation with multiple metrics")
    print("  - Multi-agent collaboration")
    print("  - Continuous learning and self-improvement")
    print("=" * 80)


if __name__ == "__main__":
    main()
