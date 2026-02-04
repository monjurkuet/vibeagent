"""Comprehensive End-to-End Test for VibeAgent

Tests all major features:
1. LLM round-robin model selection
2. PostgreSQL full-text search, vector search, knowledge graph
3. Entity extraction
4. Hybrid search with RRF
5. Quality evaluation
6. Autonomous research agent
7. Multi-agent collaboration
"""

import json
from datetime import datetime

from vibeagent.core import (
    AutonomousResearchAgent,
    ContinuousLearningLoop,
    HybridSearchOrchestrator,
    KnowledgeQualityEvaluator,
    MultiAgentCollaboration,
)
from vibeagent.skills import (
    EntityExtractionSkill,
    LLMSkill,
    PostgreSQLFullTextSkill,
)


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.duration = 0

    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} - {self.name}"


class VibeAgentETETest:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def run_test(self, test_func, name: str):
        """Run a test and capture results."""
        print(f"\n{'='*80}")
        print(f"Test: {name}")
        print('='*80)
        
        result = TestResult(name)
        start = datetime.now()
        
        try:
            details = test_func()
            result.passed = True
            result.details = details
            print(f"✓ Test passed")
        except Exception as e:
            result.passed = False
            result.error = str(e)
            print(f"✗ Test failed: {e}")
        
        result.duration = (datetime.now() - start).total_seconds()
        self.results.append(result)
        return result

    def test_llm_round_robin(self):
        """Test LLM round-robin model selection."""
        print("\nInitializing LLM with round-robin...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        
        print("Making 3 requests to verify model rotation...")
        models_used = []
        for i in range(3):
            result = llm.execute(prompt=f"Test message {i+1}", max_tokens=50)
            if result.success:
                model = result.data.get("model")
                models_used.append(model)
                print(f"  Request {i+1}: {model}")
            else:
                raise Exception(f"LLM request failed: {result.error}")
        
        return {
            "total_requests": len(models_used),
            "unique_models": len(set(models_used)),
            "models": models_used,
            "rotation": len(set(models_used)) > 1
        }

    def test_postgresql_fulltext(self):
        """Test PostgreSQL full-text search."""
        print("\nInitializing PostgreSQL with full-text search...")
        pg = PostgreSQLFullTextSkill()
        
        if not pg.validate():
            raise Exception("PostgreSQL connection failed")
        
        print("Storing sample documents...")
        docs = [
            {"id": "ft1", "title": "Python Programming", "content": "Python is a versatile programming language for web development, data science, and automation.", "metadata": {"category": "programming"}},
            {"id": "ft2", "title": "Machine Learning", "content": "Machine learning algorithms enable computers to learn from data and make predictions.", "metadata": {"category": "ai"}},
            {"id": "ft3", "title": "Database Systems", "content": "PostgreSQL is a powerful relational database with advanced features like full-text search and vector operations.", "metadata": {"category": "database"}},
        ]
        
        for doc in docs:
            result = pg.upsert_document(doc)
            if not result["success"]:
                raise Exception(f"Failed to store document: {result.get('error')}")
        
        print("Testing full-text search...")
        result = pg.fulltext_search("python programming", limit=5)
        if not result["success"]:
            raise Exception(f"Full-text search failed: {result.get('error')}")
        
        print(f"  Found {result['count']} results")
        for i, doc in enumerate(result["results"][:2], 1):
            print(f"  {i}. {doc['title']} (score: {doc['score']:.3f})")
        
        return {
            "documents_stored": len(docs),
            "search_results": result["count"],
            "top_result": result["results"][0]["title"] if result["results"] else None
        }

    def test_postgresql_vector(self):
        """Test PostgreSQL vector search."""
        print("\nTesting vector similarity search...")
        pg = PostgreSQLFullTextSkill()
        
        print("Searching semantically for 'data science'...")
        result = pg.vector_search("data science", limit=5)
        if not result["success"]:
            raise Exception(f"Vector search failed: {result.get('error')}")
        
        print(f"  Found {result['count']} results")
        for i, doc in enumerate(result["results"][:2], 1):
            print(f"  {i}. {doc['title']} (similarity: {doc['score']:.3f})")
        
        return {
            "search_results": result["count"],
            "top_score": result["results"][0]["score"] if result["results"] else 0,
            "avg_score": sum(d["score"] for d in result["results"]) / len(result["results"]) if result["results"] else 0
        }

    def test_entity_extraction(self):
        """Test entity extraction from text."""
        print("\nInitializing entity extraction...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        extractor = EntityExtractionSkill(llm_skill=llm)
        
        text = """
        PostgreSQL is an open-source relational database management system developed by the PostgreSQL Global Development Group.
        It supports both SQL and JSON queries, making it versatile for modern applications. pgvector is an extension that
        enables vector similarity search for AI and machine learning applications.
        """
        
        print(f"Extracting entities from text ({len(text)} chars)...")
        result = extractor.execute(action="extract", text=text, extract_relationships=True)
        
        if not result.success:
            raise Exception(f"Entity extraction failed: {result.error}")
        
        entities = result.data.get("entities", [])
        relationships = result.data.get("relationships", [])
        
        print(f"  Found {len(entities)} entities, {len(relationships)} relationships")
        for entity in entities[:3]:
            print(f"  - {entity.get('name')} ({entity.get('label')})")
        
        return {
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships),
            "sample_entities": [{"name": e.get("name"), "label": e.get("label")} for e in entities[:3]]
        }

    def test_knowledge_graph(self):
        """Test knowledge graph operations."""
        print("\nTesting knowledge graph storage...")
        pg = PostgreSQLFullTextSkill()
        
        entities = [
            {"id": "kg_e1", "name": "PostgreSQL", "label": "Database", "properties": {"type": "RDBMS"}},
            {"id": "kg_e2", "name": "pgvector", "label": "Extension", "properties": {"purpose": "vector search"}},
            {"id": "kg_e3", "name": "AI/ML", "label": "Field", "properties": {}},
        ]
        
        print("  Storing entities...")
        for entity in entities:
            result = pg.upsert_entity(entity)
            if not result["success"]:
                raise Exception(f"Failed to store entity: {result.get('error')}")
        
        relationships = [
            {"id": "kg_r1", "source_id": "kg_e2", "target_id": "kg_e1", "relationship_type": "EXTENDS", "properties": {}},
            {"id": "kg_r2", "source_id": "kg_e2", "target_id": "kg_e3", "relationship_type": "SUPPORTS", "properties": {}},
        ]
        
        print("  Storing relationships...")
        for rel in relationships:
            result = pg.upsert_relationship(rel)
            if not result["success"]:
                raise Exception(f"Failed to store relationship: {result.get('error')}")
        
        stats = pg.get_stats()
        print(f"  Knowledge graph stats: {stats}")
        
        return {
            "entities": stats.get("entities"),
            "relationships": stats.get("relationships"),
            "documents": stats.get("documents")
        }

    def test_hybrid_search(self):
        """Test hybrid search combining full-text and vector search."""
        print("\nTesting hybrid search with RRF...")
        pg = PostgreSQLFullTextSkill()
        
        print("  Performing hybrid search for 'database vector'...")
        result = pg.hybrid_search("database vector", limit=5, fulltext_weight=0.5, vector_weight=0.5)
        
        if not result["success"]:
            raise Exception(f"Hybrid search failed: {result.get('error')}")
        
        print(f"  Found {result['count']} results with RRF scoring")
        for i, doc in enumerate(result["results"][:2], 1):
            print(f"  {i}. {doc['title']} (RRF score: {doc['score']:.4f})")
        
        return {
            "results_count": result["count"],
            "top_score": result["results"][0]["score"] if result["results"] else 0
        }

    def test_quality_evaluation(self):
        """Test knowledge quality evaluation metrics."""
        print("\nTesting quality evaluation...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        pg = PostgreSQLFullTextSkill()
        
        evaluator = KnowledgeQualityEvaluator(llm_skill=llm, vector_skill=pg)
        
        question = "What is PostgreSQL?"
        context = "PostgreSQL is an open-source relational database management system that supports SQL, JSON, and vector operations."
        answer = "PostgreSQL is a relational database system that supports multiple data formats including SQL and JSON."
        
        print("  Evaluating RAG response quality...")
        result = evaluator.evaluate_rag(question=question, context=context, answer=answer)
        
        if not result.success:
            raise Exception(f"Quality evaluation failed: {result.error}")
        
        metrics = result.data
        print(f"  Metrics computed: {list(metrics.keys())}")
        
        return {
            "metrics": list(metrics.keys()),
            "context_relevancy": metrics.get("context_relevancy"),
            "answer_faithfulness": metrics.get("answer_faithfulness"),
            "overall_score": metrics.get("overall_score")
        }

    def test_autonomous_research(self):
        """Test autonomous research agent."""
        print("\nTesting autonomous research agent...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        
        # Initialize agent directly with llm_skill parameter
        agent = AutonomousResearchAgent(llm_skill=llm)
        pg = PostgreSQLFullTextSkill()
        agent.register_skill(pg)
        
        print("  Running autonomous research (simplified)...")
        result = agent.generate_research_plan("PostgreSQL vector extensions")
        
        if not result.success:
            raise Exception(f"Research plan generation failed: {result.error}")
        
        plan = result.data
        print(f"  Generated plan with {len(plan.get('search_queries', []))} queries")
        for i, query in enumerate(plan.get("search_queries", [])[:2], 1):
            print(f"  {i}. {query}")
        
        return {
            "plan_generated": True,
            "queries_count": len(plan.get("search_queries", [])),
            "has_summary": "summary" in plan
        }

    def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration."""
        print("\nTesting multi-agent collaboration...")
        collaboration = MultiAgentCollaboration()
        
        # Create agent instances
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        agent1 = AutonomousResearchAgent(llm_skill=llm, name="research_agent")
        agent2 = AutonomousResearchAgent(llm_skill=llm, name="analysis_agent")
        
        # Register agents
        print("  Registering agents...")
        agent1_id = collaboration.register_agent(agent1, role="researcher")
        agent2_id = collaboration.register_agent(agent2, role="analyzer")
        
        print("  Creating collaboration session...")
        result = collaboration.create_session(
            agent_ids=[agent1_id, agent2_id],
            task_description="Test collaboration",
            shared_context={"test": True}
        )
        
        if not result.success:
            raise Exception(f"Session creation failed: {result.error}")
        
        session = result.data
        print(f"  Session created: {session.get('session_id')}")
        print(f"  Agents: {session.get('agents')}")
        
        return {
            "session_id": session.get("session_id"),
            "agents_count": len(session.get("agents", [])),
            "messages_count": len(session.get("messages", []))
        }

    def run_all_tests(self):
        """Run all end-to-end tests."""
        print("="*80)
        print("VibeAgent Comprehensive End-to-End Test Suite")
        print("="*80)
        print(f"Started at: {self.start_time}")
        
        # Run all tests
        self.run_test(self.test_llm_round_robin, "LLM Round-Robin Model Selection")
        self.run_test(self.test_postgresql_fulltext, "PostgreSQL Full-Text Search")
        self.run_test(self.test_postgresql_vector, "PostgreSQL Vector Search")
        self.run_test(self.test_entity_extraction, "Entity Extraction")
        self.run_test(self.test_knowledge_graph, "Knowledge Graph Operations")
        self.run_test(self.test_hybrid_search, "Hybrid Search with RRF")
        self.run_test(self.test_quality_evaluation, "Quality Evaluation Metrics")
        self.run_test(self.test_autonomous_research, "Autonomous Research Agent")
        self.run_test(self.test_multi_agent_collaboration, "Multi-Agent Collaboration")

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate final test report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        print("\n" + "="*80)
        print("TEST SUMMARY REPORT")
        print("="*80)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success Rate: {(passed/len(self.results)*100):.1f}%")
        print("\nDetailed Results:")
        print("-"*80)
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} - {result.name} ({result.duration:.2f}s)")
            if result.error:
                print(f"       Error: {result.error}")
            if result.details:
                print(f"       Details: {json.dumps(result.details, indent=10, default=str)[:200]}...")
        
        print("\n" + "="*80)
        print("VibeAgent E2E Test Complete")
        print("="*80)


def main():
    tester = VibeAgentETETest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
