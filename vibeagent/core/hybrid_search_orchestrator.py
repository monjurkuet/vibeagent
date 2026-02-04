"""Hybrid Search Orchestrator with Reciprocal Rank Fusion (RRF).

This orchestrator combines results from vector, keyword, and graph search
using RRF to provide superior retrieval performance.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from multiple sources."""

    id: str
    content: str
    score: float
    source: str  # "vector", "keyword", "graph", "hybrid"
    metadata: dict = field(default_factory=dict)
    vector_score: float | None = None
    keyword_score: float | None = None
    graph_score: float | None = None
    rank_vector: int | None = None
    rank_keyword: int | None = None
    rank_graph: int | None = None


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    enable_vector_search: bool = True
    enable_keyword_search: bool = True
    enable_graph_search: bool = False
    vector_weight: float = 1.0
    keyword_weight: float = 0.5
    graph_weight: float = 0.3
    rrf_k: int = 60  # RRF constant
    top_k: int = 10
    rerank: bool = True


class HybridSearchOrchestrator:
    """Orchestrator for hybrid search with RRF."""

    def __init__(
        self,
        vector_skill: BaseSkill | None = None,
        keyword_skill: BaseSkill | None = None,
        graph_skill: BaseSkill | None = None,
        config: HybridSearchConfig | None = None,
    ):
        """Initialize Hybrid Search Orchestrator.

        Args:
            vector_skill: QdrantSkill for vector search
            keyword_skill: ElasticsearchSkill for keyword search
            graph_skill: Neo4jSkill for graph search
            config: Hybrid search configuration
        """
        self.vector_skill = vector_skill
        self.keyword_skill = keyword_skill
        self.graph_skill = graph_skill
        self.config = config or HybridSearchConfig()
        logger.info("HybridSearchOrchestrator initialized")

    def search(
        self,
        query: str,
        limit: int | None = None,
        filters: dict | None = None,
        **kwargs,
    ) -> SkillResult:
        """Perform hybrid search using RRF.

        Args:
            query: Search query
            limit: Number of results to return
            filters: Metadata filters
            **kwargs: Additional parameters

        Returns:
            SkillResult with combined search results
        """
        limit = limit or self.config.top_k

        # Collect results from all enabled sources
        vector_results = []
        keyword_results = []
        graph_results = []

        # Vector search
        if self.config.enable_vector_search and self.vector_skill:
            try:
                result = self.vector_skill.execute(
                    action="search",
                    query=query,
                    limit=limit * 2,  # Get more for better RRF
                    filter=filters,
                )
                if result.success:
                    vector_results = result.data.get("results", [])
                    logger.info(f"Vector search returned {len(vector_results)} results")
            except Exception as e:
                logger.error(f"Vector search failed: {e}")

        # Keyword search
        if self.config.enable_keyword_search and self.keyword_skill:
            try:
                result = self.keyword_skill.execute(
                    action="search",
                    query=query,
                    limit=limit * 2,
                    filter=filters,
                )
                if result.success:
                    keyword_results = result.data.get("results", [])
                    logger.info(f"Keyword search returned {len(keyword_results)} results")
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")

        # Graph search
        if self.config.enable_graph_search and self.graph_skill:
            try:
                result = self.graph_skill.execute(
                    action="search",
                    query=query,
                    limit=limit,
                )
                if result.success:
                    graph_entities = result.data.get("entities", [])
                    # Convert graph entities to search results format
                    graph_results = [
                        {
                            "id": entity.get("id"),
                            "content": entity.get("description", ""),
                            "score": 1.0,
                            "metadata": entity,
                        }
                        for entity in graph_entities
                    ]
                    logger.info(f"Graph search returned {len(graph_results)} results")
            except Exception as e:
                logger.error(f"Graph search failed: {e}")

        # Apply RRF to combine results
        combined_results = self._apply_rrf(
            vector_results=vector_results,
            keyword_results=keyword_results,
            graph_results=graph_results,
            query=query,
        )

        # Sort by combined score and limit
        combined_results.sort(key=lambda x: x.score, reverse=True)
        final_results = combined_results[:limit]

        # Rerank if enabled
        if self.config.rerank and len(final_results) > 1:
            final_results = self._rerank_results(query, final_results)

        self._record_usage()
        return SkillResult(
            success=True,
            data={
                "query": query,
                "results": [self._search_result_to_dict(r) for r in final_results],
                "total": len(final_results),
                "sources": {
                    "vector_count": len(vector_results),
                    "keyword_count": len(keyword_results),
                    "graph_count": len(graph_results),
                },
                "config": {
                    "vector_weight": self.config.vector_weight,
                    "keyword_weight": self.config.keyword_weight,
                    "graph_weight": self.config.graph_weight,
                },
            },
        )

    def _apply_rrf(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        graph_results: list[dict],
        query: str,
    ) -> list[SearchResult]:
        """Apply Reciprocal Rank Fusion to combine results.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            graph_results: Results from graph search
            query: Original query

        Returns:
            Combined and ranked results
        """
        # Track all unique documents
        result_map = {}

        # Process vector results
        for i, result in enumerate(vector_results):
            doc_id = result.get("id", str(i))
            if doc_id not in result_map:
                result_map[doc_id] = SearchResult(
                    id=doc_id,
                    content=result.get("content", ""),
                    score=0.0,
                    source="hybrid",
                    metadata=result.get("metadata", {}),
                )
            result_map[doc_id].vector_score = result.get("score", 0.0)
            result_map[doc_id].rank_vector = i + 1
            result_map[doc_id].metadata.update(result.get("metadata", {}))

        # Process keyword results
        for i, result in enumerate(keyword_results):
            doc_id = result.get("id", str(len(vector_results) + i))
            if doc_id not in result_map:
                result_map[doc_id] = SearchResult(
                    id=doc_id,
                    content=result.get("content", ""),
                    score=0.0,
                    source="hybrid",
                    metadata=result.get("metadata", {}),
                )
            result_map[doc_id].keyword_score = result.get("score", 0.0)
            result_map[doc_id].rank_keyword = i + 1
            result_map[doc_id].metadata.update(result.get("metadata", {}))

        # Process graph results
        for i, result in enumerate(graph_results):
            doc_id = result.get("id", str(len(vector_results) + len(keyword_results) + i))
            if doc_id not in result_map:
                result_map[doc_id] = SearchResult(
                    id=doc_id,
                    content=result.get("content", ""),
                    score=0.0,
                    source="hybrid",
                    metadata=result.get("metadata", {}),
                )
            result_map[doc_id].graph_score = result.get("score", 0.0)
            result_map[doc_id].rank_graph = i + 1
            result_map[doc_id].metadata.update(result.get("metadata", {}))

        # Calculate RRF scores
        k = self.config.rrf_k
        for doc_id, result in result_map.items():
            rrf_score = 0.0

            if result.rank_vector is not None:
                rrf_score += self.config.vector_weight / (k + result.rank_vector)

            if result.rank_keyword is not None:
                rrf_score += self.config.keyword_weight / (k + result.rank_keyword)

            if result.rank_graph is not None:
                rrf_score += self.config.graph_weight / (k + result.rank_graph)

            result.score = rrf_score

        return list(result_map.values())

    def _rerank_results(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Rerank results based on query relevance.

        Args:
            query: Search query
            results: Results to rerank

        Returns:
            Reranked results
        """
        try:
            # Simple relevance-based reranking
            # In production, use a more sophisticated reranking model
            scored_results = []

            for result in results:
                # Boost score based on content overlap with query
                query_terms = set(query.lower().split())
                content_terms = set(result.content.lower().split())

                overlap = len(query_terms & content_terms)
                relevance_boost = min(overlap * 0.1, 0.5)  # Max 0.5 boost

                result.score += relevance_boost
                scored_results.append(result)

            return sorted(scored_results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def _search_result_to_dict(self, result: SearchResult) -> dict:
        """Convert SearchResult to dictionary.

        Args:
            result: SearchResult object

        Returns:
            Dictionary representation
        """
        return {
            "id": result.id,
            "content": result.content,
            "score": result.score,
            "source": result.source,
            "metadata": result.metadata,
            "vector_score": result.vector_score,
            "keyword_score": result.keyword_score,
            "graph_score": result.graph_score,
            "ranks": {
                "vector": result.rank_vector,
                "keyword": result.rank_keyword,
                "graph": result.rank_graph,
            },
        }

    def _record_usage(self):
        """Record usage statistics."""
        pass  # Could be extended to track metrics

    def get_stats(self) -> dict:
        """Get search statistics.

        Returns:
            Search statistics
        """
        return {
            "config": {
                "enable_vector_search": self.config.enable_vector_search,
                "enable_keyword_search": self.config.enable_keyword_search,
                "enable_graph_search": self.config.enable_graph_search,
                "vector_weight": self.config.vector_weight,
                "keyword_weight": self.config.keyword_weight,
                "graph_weight": self.config.graph_weight,
            },
            "skills_available": {
                "vector": self.vector_skill is not None,
                "keyword": self.keyword_skill is not None,
                "graph": self.graph_skill is not None,
            },
        }

    def health_check(self) -> bool:
        """Check if orchestrator is healthy."""
        skills = [self.vector_skill, self.keyword_skill, self.graph_skill]
        return any(skill is not None for skill in skills)