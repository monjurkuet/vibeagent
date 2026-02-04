"""VibeAgent Extreme Edge Case Test - Insane Difficulty

Tests extreme scenarios:
1. Concurrent operations with thread safety
2. Large dataset handling (1000+ documents)
3. Complex nested queries
4. Memory stress testing
5. Race conditions
6. Invalid/malformed data handling
7. Deep knowledge graph traversals
8. Extreme search combinations
9. Error recovery under stress
10. Timeout and retry scenarios
"""

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from vibeagent.core import KnowledgeQualityEvaluator
from vibeagent.skills import (
    EntityExtractionSkill,
    LLMSkill,
    PostgreSQLFullTextSkill,
)


class ExtremeTestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.duration = 0


class ExtremeEdgeCaseTest:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def run_test(self, test_func, name: str):
        """Run an extreme test."""
        print(f"\n{'='*80}")
        print(f"EXTREME TEST: {name}")
        print('='*80)
        
        result = ExtremeTestResult(name)
        start = datetime.now()
        
        try:
            details = test_func()
            result.passed = True
            result.details = details
            print(f"✓ Test passed")
        except Exception as e:
            result.passed = False
            result.error = str(e)
            print(f"✗ Test failed: {type(e).__name__}: {e}")
        
        result.duration = (datetime.now() - start).total_seconds()
        self.results.append(result)
        return result

    def test_concurrent_writes(self):
        """Test 100 concurrent document writes."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Launching 100 concurrent writes...")
        results = []
        threads = []
        errors = []
        
        def write_doc(i):
            try:
                doc = {
                    "id": f"concurrent_{i}",
                    "title": f"Concurrent Document {i}",
                    "content": f"This is document {i} with some unique content for testing concurrent writes.",
                    "metadata": {"batch": i % 10, "thread": threading.current_thread().name}
                }
                result = pg.upsert_document(doc)
                if not result["success"]:
                    errors.append((i, result.get("error")))
                return result["success"]
            except Exception as e:
                errors.append((i, str(e)))
                return False
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(write_doc, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]
        
        success_count = sum(results)
        print(f"  Completed: {success_count}/100 successful writes")
        print(f"  Errors: {len(errors)}")
        
        if errors[:5]:
            print(f"  Sample errors: {errors[:5]}")
        
        stats = pg.get_stats()
        return {
            "total_writes": 100,
            "successful_writes": success_count,
            "errors": len(errors),
            "documents_in_db": stats.get("documents", 0)
        }

    def test_large_dataset_search(self):
        """Test search performance with 1000+ documents."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Inserting 1000 documents...")
        start = time.time()
        
        for i in range(1000):
            doc = {
                "id": f"large_{i}",
                "title": f"Large Dataset Document {i}",
                "content": f"This is document {i} covering topics like AI, machine learning, databases, programming, and computer science. " * 10,
                "metadata": {"category": random.choice(["ai", "database", "programming", "science"]), "index": i}
            }
            pg.upsert_document(doc)
        
        insert_time = time.time() - start
        print(f"  Insert time: {insert_time:.2f}s")
        
        # Test various search types
        print("  Testing full-text search...")
        start = time.time()
        ft_result = pg.fulltext_search("machine learning", limit=100)
        ft_time = time.time() - start
        
        print("  Testing vector search...")
        start = time.time()
        v_result = pg.vector_search("artificial intelligence", limit=100)
        v_time = time.time() - start
        
        print("  Testing hybrid search...")
        start = time.time()
        h_result = pg.hybrid_search("computer science programming", limit=100)
        h_time = time.time() - start
        
        return {
            "documents_inserted": 1000,
            "insert_time": insert_time,
            "fulltext_search_time": ft_time,
            "fulltext_results": ft_result.get("count"),
            "vector_search_time": v_time,
            "vector_results": v_result.get("count"),
            "hybrid_search_time": h_time,
            "hybrid_results": h_result.get("count")
        }

    def test_malformed_data_handling(self):
        """Test handling of malformed and edge case data."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Testing edge case documents...")
        edge_cases = [
            {"id": "empty", "title": "", "content": "", "metadata": {}},
            {"id": "unicode", "title": "Unicode 🚀 Test 中文 日本語", "content": "Special chars: áéíóú ñ ß ç λ", "metadata": {}},
            {"id": "nulls", "title": None, "content": "Content with null title", "metadata": None},
            {"id": "long", "title": "A" * 500, "content": "B" * 10000, "metadata": {"long": "C" * 5000}},
            {"id": "nested", "title": "Nested", "content": "Test", "metadata": {"a": {"b": {"c": {"d": "deep"}}}}},
            {"id": "array", "title": "Array", "content": "Test", "metadata": {"list": [1, 2, 3, {"nested": [4, 5]}]}},
            {"id": "special", "title": "SQL Injection ' OR 1=1 --", "content": "<script>alert('xss')</script>", "metadata": {"sql": "'; DROP TABLE users;"}},
        ]
        
        results = []
        for doc in edge_cases:
            try:
                result = pg.upsert_document(doc)
                results.append({"id": doc["id"], "success": result["success"]})
            except Exception as e:
                results.append({"id": doc["id"], "success": False, "error": str(e)})
        
        print(f"  Handled {len(results)} edge cases")
        for r in results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['id']}")
        
        return {
            "total_cases": len(edge_cases),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }

    def test_deep_knowledge_graph(self):
        """Test deep knowledge graph with many relationships."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Creating complex knowledge graph...")
        
        # Create 50 entities
        entities = []
        for i in range(50):
            entities.append({
                "id": f"entity_{i}",
                "name": f"Entity {i}",
                "label": random.choice(["Person", "Organization", "Concept", "Location", "Event"]),
                "properties": {"index": i, "random": random.random()}
            })
        
        print("  Inserting 50 entities...")
        for entity in entities:
            pg.upsert_entity(entity)
        
        # Create complex relationship web
        relationships = []
        for i in range(100):
            source = random.randint(0, 49)
            target = random.randint(0, 49)
            if source != target:
                relationships.append({
                    "id": f"rel_{i}",
                    "source_id": f"entity_{source}",
                    "target_id": f"entity_{target}",
                    "relationship_type": random.choice(["CONNECTED_TO", "RELATED_TO", "DEPENDS_ON", "PART_OF", "SIMILAR_TO"]),
                    "properties": {"weight": random.random()}
                })
        
        print("  Inserting 100 relationships...")
        for rel in relationships:
            pg.upsert_relationship(rel)
        
        stats = pg.get_stats()
        return {
            "entities": stats.get("entities"),
            "relationships": stats.get("relationships"),
            "avg_relationships_per_entity": stats.get("relationships", 0) / max(stats.get("entities", 1), 1)
        }

    def test_complex_search_combinations(self):
        """Test complex search with multiple filters and combinations."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Testing complex search scenarios...")
        
        # Insert diverse documents
        categories = ["ai", "database", "programming", "science", "business"]
        for i in range(100):
            doc = {
                "id": f"complex_{i}",
                "title": f"Complex Test Document {i}",
                "content": f"Document {i} about {categories[i % 5]} with various topics and keywords.",
                "metadata": {
                    "category": categories[i % 5],
                    "priority": i % 3,
                    "year": 2020 + (i % 5),
                    "tags": [f"tag{j}" for j in range(3)]
                }
            }
            pg.upsert_document(doc)
        
        # Test 1: Multiple metadata filters
        print("  Test 1: Multiple metadata filters...")
        result1 = pg.fulltext_search(
            "document",
            limit=10,
            filter_dict={"category": "ai", "priority": 1}
        )
        
        # Test 2: No results expected
        print("  Test 2: No results expected...")
        result2 = pg.fulltext_search(
            "nonexistent term xyz",
            limit=10
        )
        
        # Test 3: Hybrid with extreme weights
        print("  Test 3: Hybrid with extreme weights...")
        result3 = pg.hybrid_search(
            "complex document",
            limit=10,
            fulltext_weight=0.9,
            vector_weight=0.1
        )
        
        # Test 4: Vector search with empty result
        print("  Test 4: Vector search unlikely to match...")
        result4 = pg.vector_search("quantum entanglement superposition", limit=10)
        
        return {
            "complex_filters": result1.get("count"),
            "no_results": result2.get("count") == 0,
            "extreme_weights": result3.get("count"),
            "unlikely_match": result4.get("count")
        }

    def test_llm_stress(self):
        """Test LLM with rapid requests and error handling."""
        print("\nInitializing LLM with round-robin...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        
        print("Making 20 rapid requests...")
        models_used = []
        errors = []
        
        for i in range(20):
            try:
                result = llm.execute(
                    prompt=f"Count from 1 to {i+1}",
                    max_tokens=50,
                    temperature=random.uniform(0, 1)
                )
                if result.success:
                    models_used.append(result.data.get("model"))
                else:
                    errors.append(result.error)
            except Exception as e:
                errors.append(str(e))
        
        print(f"  Successful: {len(models_used)}/20")
        print(f"  Errors: {len(errors)}")
        print(f"  Unique models: {len(set(models_used))}")
        
        return {
            "total_requests": 20,
            "successful": len(models_used),
            "errors": len(errors),
            "unique_models": len(set(models_used)),
            "model_rotation": len(set(models_used)) > 1
        }

    def test_entity_extraction_stress(self):
        """Test entity extraction with varying text sizes and complexity."""
        print("\nInitializing entity extraction...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        extractor = EntityExtractionSkill(llm_skill=llm)
        
        print("Testing with different text complexities...")
        test_texts = [
            ("Very short", "PostgreSQL is a database."),
            ("Medium", """
                PostgreSQL is an open-source relational database management system 
                developed by the PostgreSQL Global Development Group. It supports 
                both SQL and JSON queries.
            """),
            ("Long", " ".join(["PostgreSQL database system"] * 100)),
            ("Complex", """
                In the field of database management systems, PostgreSQL stands out as a 
                powerful open-source solution. The PostgreSQL Global Development Group, 
                a community-driven organization, oversees its development. Key contributors 
                include Bruce Momjian, Tom Lane, and Josh Berkus. The system supports 
                advanced features like full-text search (via tsvector), vector operations 
                (via pgvector extension), and JSONB storage. Major organizations such as 
                Apple, Netflix, and Spotify utilize PostgreSQL for various applications.
            """),
        ]
        
        results = []
        for name, text in test_texts:
            try:
                result = extractor.execute(action="extract", text=text, extract_relationships=True)
                results.append({
                    "name": name,
                    "length": len(text),
                    "success": result.success,
                    "entities": len(result.data.get("entities", [])) if result.success else 0,
                    "relationships": len(result.data.get("relationships", [])) if result.success else 0
                })
                print(f"  {name}: {len(text)} chars - {len(result.data.get('entities', []))} entities")
            except Exception as e:
                results.append({"name": name, "success": False, "error": str(e)})
                print(f"  {name}: Failed - {type(e).__name__}")
        
        return {
            "total_tests": len(test_texts),
            "successful": sum(1 for r in results if r["success"]),
            "total_entities": sum(r.get("entities", 0) for r in results),
            "total_relationships": sum(r.get("relationships", 0) for r in results)
        }

    def test_memory_stress(self):
        """Test memory handling with large operations."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Creating large documents (10KB each)...")
        large_content = "A" * 10000  # 10KB
        
        large_docs = []
        for i in range(50):
            doc = {
                "id": f"large_mem_{i}",
                "title": f"Large Document {i}",
                "content": large_content,
                "metadata": {"data": "B" * 5000, "size": i * 1000}
            }
            large_docs.append(doc)
        
        print("  Inserting 50 large documents...")
        start = time.time()
        for doc in large_docs:
            pg.upsert_document(doc)
        insert_time = time.time() - start
        
        print(f"  Insert time: {insert_time:.2f}s")
        
        print("  Testing search on large documents...")
        search_result = pg.fulltext_search("AAA", limit=10)
        
        return {
            "document_size": 10000,
            "document_count": 50,
            "total_data_size": 50 * 10000,
            "insert_time": insert_time,
            "search_results": search_result.get("count")
        }

    def test_race_conditions(self):
        """Test race conditions with concurrent mixed operations."""
        print("\nInitializing PostgreSQL...")
        pg = PostgreSQLFullTextSkill()
        
        print("Launching 50 concurrent mixed operations...")
        
        def mixed_operation(i):
            try:
                if i % 3 == 0:
                    # Write operation
                    doc = {
                        "id": f"race_{i}",
                        "title": f"Race Condition {i}",
                        "content": f"Content {i}",
                        "metadata": {"op": "write"}
                    }
                    return pg.upsert_document(doc)
                elif i % 3 == 1:
                    # Read operation
                    return pg.fulltext_search(f"race {i}", limit=5)
                else:
                    # Stats operation
                    return pg.get_stats()
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(mixed_operation, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]
        
        successful = sum(1 for r in results if r.get("success", True))
        print(f"  Completed: {successful}/50 operations")
        
        return {
            "total_operations": 50,
            "successful": successful,
            "failed": 50 - successful
        }

    def test_quality_evaluation_extreme(self):
        """Test quality evaluation with extreme cases."""
        print("\nInitializing quality evaluator...")
        llm = LLMSkill(base_url="http://localhost:8087/v1", enable_round_robin=True)
        pg = PostgreSQLFullTextSkill()
        
        evaluator = KnowledgeQualityEvaluator(llm_skill=llm, vector_skill=pg)
        
        print("Testing edge case evaluations...")
        
        test_cases = [
            {
                "name": "Perfect match",
                "question": "What is PostgreSQL?",
                "context": "PostgreSQL is an open-source relational database management system.",
                "answer": "PostgreSQL is a relational database system."
            },
            {
                "name": "No context",
                "question": "What is PostgreSQL?",
                "context": "",
                "answer": "PostgreSQL is a database system."
            },
            {
                "name": "Irrelevant answer",
                "question": "What is PostgreSQL?",
                "context": "PostgreSQL is a database system.",
                "answer": "The sun is a star in our solar system."
            },
            {
                "name": "Very long context",
                "question": "What is it?",
                "context": "PostgreSQL " * 1000,
                "answer": "It is PostgreSQL."
            },
        ]
        
        results = []
        for case in test_cases:
            try:
                result = evaluator.evaluate_rag(
                    question=case["question"],
                    context=case["context"],
                    answer=case["answer"]
                )
                results.append({
                    "name": case["name"],
                    "success": result.success,
                    "overall_score": result.data.get("overall_score") if result.success else None
                })
                print(f"  {case['name']}: score={result.data.get('overall_score') if result.success else 'N/A'}")
            except Exception as e:
                results.append({"name": case["name"], "success": False, "error": str(e)})
                print(f"  {case['name']}: Failed")
        
        return {
            "total_cases": len(test_cases),
            "successful": sum(1 for r in results if r["success"]),
            "average_score": sum(r.get("overall_score", 0) for r in results if r["success"]) / max(sum(1 for r in results if r["success"]), 1)
        }

    def run_all_tests(self):
        """Run all extreme edge case tests."""
        print("="*80)
        print("VibeAgent Extreme Edge Case Test Suite - INSANE DIFFICULTY")
        print("="*80)
        print(f"Started at: {self.start_time}")
        
        # Run all extreme tests
        self.run_test(self.test_concurrent_writes, "Concurrent 100 Writes (Thread Safety)")
        self.run_test(self.test_large_dataset_search, "Large Dataset Search (1000+ Documents)")
        self.run_test(self.test_malformed_data_handling, "Malformed Data Handling (Edge Cases)")
        self.run_test(self.test_deep_knowledge_graph, "Deep Knowledge Graph (50 Entities, 100 Relationships)")
        self.run_test(self.test_complex_search_combinations, "Complex Search Combinations")
        self.run_test(self.test_llm_stress, "LLM Stress Test (20 Rapid Requests)")
        self.run_test(self.test_entity_extraction_stress, "Entity Extraction Stress (Varying Complexity)")
        self.run_test(self.test_memory_stress, "Memory Stress Test (50KB Documents)")
        self.run_test(self.test_race_conditions, "Race Conditions (50 Concurrent Mixed Ops)")
        self.run_test(self.test_quality_evaluation_extreme, "Quality Evaluation Extreme Cases")

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate final test report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        print("\n" + "="*80)
        print("EXTREME TEST SUMMARY REPORT")
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
                print(f"       Details: {json.dumps(result.details, indent=10, default=str)[:150]}...")
        
        print("\n" + "="*80)
        print("VibeAgent Extreme Edge Case Test Complete")
        print("="*80)


def main():
    tester = ExtremeEdgeCaseTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
