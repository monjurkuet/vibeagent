"""Context management system for efficient conversation handling."""

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context available."""

    FULL = "full"
    SUMMARY = "summary"
    RELEVANT = "relevant"
    MINIMAL = "minimal"


class CompressionStrategy(Enum):
    """Strategies for context compression."""

    IMPORTANCE_BASED = "importance_based"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class ContextConfig:
    """Configuration for context management."""

    max_context_tokens: int = 8000
    summary_threshold: int = 4000
    importance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "user_message": 1.0,
            "assistant_message": 0.8,
            "tool_call": 0.9,
            "tool_result": 0.7,
            "error": 1.2,
            "system_message": 0.5,
        }
    )
    recency_weight: float = 0.4
    compression_strategy: CompressionStrategy = CompressionStrategy.HYBRID
    summary_compression_ratio: float = 0.65
    cache_size: int = 100
    enable_semantic_search: bool = True
    min_importance_score: float = 0.3


@dataclass
class MessageScore:
    """Score for a message."""

    message_index: int
    importance_score: float
    recency_score: float
    final_score: float
    factors: dict[str, float] = field(default_factory=dict)


@dataclass
class ContextSummary:
    """Summary of conversation context."""

    original_messages: int
    summarized_messages: int
    token_reduction: float
    summary_text: str
    key_points: list[str]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContextAnalysis:
    """Analysis results for context."""

    total_messages: int
    total_tokens: int
    redundant_messages: int
    compression_potential: float
    quality_score: float
    gaps: list[str]
    suggestions: list[str]


@dataclass
class ContextWindow:
    """A window of context messages."""

    messages: list[dict]
    start_index: int
    end_index: int
    tokens: int
    importance_score: float


class ContextManager:
    """Manages conversation context with sliding window and summarization."""

    def __init__(
        self,
        config: ContextConfig | None = None,
        db_manager=None,
        llm_skill=None,
    ):
        """Initialize context manager.

        Args:
            config: Configuration for context management
            db_manager: DatabaseManager for persistence
            llm_skill: LLMSkill for summarization
        """
        self.config = config or ContextConfig()
        self.db_manager = db_manager
        self.llm_skill = llm_skill

        self._context_cache: dict[str, list[dict]] = {}
        self._summary_cache: dict[str, ContextSummary] = {}
        self._usage_patterns: dict[str, list[datetime]] = defaultdict(list)
        self._token_history: list[tuple[datetime, int]] = []

        self._initialize_database()

    def _initialize_database(self):
        """Initialize database tables for context management."""
        if not self.db_manager:
            return

        try:
            schema = """
            CREATE TABLE IF NOT EXISTS context_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                key_points TEXT,
                original_message_count INTEGER,
                summarized_message_count INTEGER,
                token_reduction REAL,
                compression_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS context_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                context_data TEXT NOT NULL,
                message_count INTEGER,
                token_count INTEGER,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS context_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                context_type TEXT NOT NULL,
                message_count INTEGER,
                token_count INTEGER,
                compression_ratio REAL,
                access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_context_summaries_session ON context_summaries(session_id);
            CREATE INDEX IF NOT EXISTS idx_context_cache_key ON context_cache(cache_key);
            CREATE INDEX IF NOT EXISTS idx_context_usage_session ON context_usage(session_id);
            """

            with self.db_manager.get_connection() as conn:
                conn.executescript(schema)

        except Exception as e:
            logger.error(f"Failed to initialize context database: {e}")

    def manage_context(self, messages: list[dict], max_tokens: int | None = None) -> list[dict]:
        """Manage context within token limits using sliding window.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens allowed (uses config default if None)

        Returns:
            Filtered messages within token limits
        """
        max_tokens = max_tokens or self.config.max_context_tokens

        if not messages:
            return []

        current_tokens = self.get_token_usage(messages)

        if current_tokens <= max_tokens:
            return messages

        logger.info(
            f"Context exceeds limit: {current_tokens} > {max_tokens} tokens, "
            f"applying sliding window"
        )

        scored_messages = self._score_messages(messages)
        sorted_messages = sorted(scored_messages, key=lambda x: x.final_score, reverse=True)

        selected_messages = []
        selected_tokens = 0

        for msg_score in sorted_messages:
            msg = messages[msg_score.message_index]
            msg_tokens = self.estimate_tokens(json.dumps(msg))

            if selected_tokens + msg_tokens > max_tokens:
                continue

            selected_messages.append(msg)
            selected_tokens += msg_tokens

        selected_messages.sort(key=lambda x: messages.index(x))

        logger.info(
            f"Reduced context from {len(messages)} to {len(selected_messages)} messages "
            f"({current_tokens} -> {selected_tokens} tokens)"
        )

        return selected_messages

    def calculate_importance(self, message: dict) -> MessageScore:
        """Calculate importance score for a message.

        Args:
            message: Message dictionary

        Returns:
            MessageScore with calculated scores
        """
        role = message.get("role", "")
        content = message.get("content", "")

        base_weight = self.config.importance_weights.get(
            f"{role}_message", self.config.importance_weights.get(role, 0.5)
        )

        factors = {"role_weight": base_weight}

        content_lower = content.lower()

        if role == "tool":
            if "error" in content_lower or "fail" in content_lower:
                base_weight *= 1.3
                factors["error_indicator"] = 0.3

            if "success" in content_lower:
                base_weight *= 1.1
                factors["success_indicator"] = 0.1

        if role == "user":
            if any(
                word in content_lower
                for word in ["important", "critical", "urgent", "must", "need"]
            ):
                base_weight *= 1.2
                factors["importance_indicator"] = 0.2

            if "?" in content:
                base_weight *= 1.1
                factors["question_indicator"] = 0.1

        if role == "assistant":
            if "final answer" in content_lower or "conclusion" in content_lower:
                base_weight *= 1.2
                factors["conclusion_indicator"] = 0.2

        content_length = len(content)
        if content_length > 1000:
            base_weight *= 1.1
            factors["length_indicator"] = 0.1
        elif content_length < 50:
            base_weight *= 0.8
            factors["short_indicator"] = -0.2

        importance_score = min(base_weight, 2.0)

        return MessageScore(
            message_index=0,
            importance_score=importance_score,
            recency_score=0.0,
            final_score=importance_score,
            factors=factors,
        )

    def _score_messages(self, messages: list[dict]) -> list[MessageScore]:
        """Score all messages with importance and recency.

        Args:
            messages: List of messages

        Returns:
            List of MessageScore objects
        """
        scored_messages = []
        total_messages = len(messages)

        for idx, message in enumerate(messages):
            msg_score = self.calculate_importance(message)
            msg_score.message_index = idx

            recency_position = idx / max(total_messages, 1)
            msg_score.recency_score = (
                1.0 - recency_position
            ) * self.config.recency_weight + recency_position * (1.0 - self.config.recency_weight)

            msg_score.final_score = (
                msg_score.importance_score * (1.0 - self.config.recency_weight)
                + msg_score.recency_score * self.config.recency_weight
            )

            scored_messages.append(msg_score)

        return scored_messages

    def summarize_messages(
        self, messages: list[dict], target_reduction: float = 0.65
    ) -> ContextSummary:
        """Summarize a group of messages to reduce token count.

        Args:
            messages: Messages to summarize
            target_reduction: Target reduction ratio (0.0-1.0)

        Returns:
            ContextSummary with summary and metadata
        """
        if not messages:
            return ContextSummary(
                original_messages=0,
                summarized_messages=0,
                token_reduction=0.0,
                summary_text="",
                key_points=[],
            )

        original_tokens = sum(self.estimate_tokens(json.dumps(msg)) for msg in messages)

        cache_key = self._generate_cache_key(messages)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        key_points = self._extract_key_points(messages)
        summary_text = self._create_summary_text(messages, key_points)

        summarized_tokens = self.estimate_tokens(summary_text)
        token_reduction = 1.0 - (summarized_tokens / max(original_tokens, 1))

        summary = ContextSummary(
            original_messages=len(messages),
            summarized_messages=1,
            token_reduction=token_reduction,
            summary_text=summary_text,
            key_points=key_points,
        )

        self._summary_cache[cache_key] = summary

        if self.db_manager:
            self._store_summary(cache_key, summary)

        logger.info(
            f"Summarized {len(messages)} messages: "
            f"{original_tokens} -> {summarized_tokens} tokens "
            f"({token_reduction:.1%} reduction)"
        )

        return summary

    def _extract_key_points(self, messages: list[dict]) -> list[str]:
        """Extract key points from messages.

        Args:
            messages: Messages to analyze

        Returns:
            List of key points
        """
        key_points = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                sentences = re.split(r"[.!?]", content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 10 and any(
                        word in sentence.lower()
                        for word in [
                            "need",
                            "want",
                            "should",
                            "must",
                            "important",
                            "please",
                            "help",
                        ]
                    ):
                        key_points.append(f"User: {sentence}")

            elif role == "tool":
                if "success" in content.lower():
                    key_points.append("Tool executed successfully")
                elif "error" in content.lower():
                    error_match = re.search(r"error[:\s]*(.*?)(?:\n|$)", content, re.IGNORECASE)
                    if error_match:
                        key_points.append(f"Tool error: {error_match.group(1).strip()}")

            elif role == "assistant":
                if "final answer" in content.lower():
                    answer_match = re.search(
                        r"final answer[:\s]*(.*?)(?:\n|$)", content, re.IGNORECASE
                    )
                    if answer_match:
                        key_points.append(f"Final answer: {answer_match.group(1).strip()[:100]}")

        return key_points[:10]

    def _create_summary_text(self, messages: list[dict], key_points: list[str]) -> str:
        """Create summary text from messages and key points.

        Args:
            messages: Original messages
            key_points: Extracted key points

        Returns:
            Summary text
        """
        if not key_points:
            return "[No significant content to summarize]"

        summary_parts = ["Conversation Summary:"]
        summary_parts.extend(f"- {point}" for point in key_points)

        return "\n".join(summary_parts)

    def compress_context(self, messages: list[dict]) -> list[dict]:
        """Compress context by removing redundancy and merging similar messages.

        Args:
            messages: Messages to compress

        Returns:
            Compressed message list
        """
        if not messages:
            return []

        original_count = len(messages)
        compressed = []

        redundancy_groups = self.detect_redundancy(messages)

        seen_hashes = set()

        for msg in messages:
            msg_hash = self._hash_message(msg)

            if msg_hash in seen_hashes:
                continue

            is_redundant = False
            for group in redundancy_groups:
                if msg_hash in group:
                    if len(group) > 1:
                        representative = messages[list(group)[0]]
                        if representative not in compressed:
                            compressed.append(representative)
                        is_redundant = True
                        break

            if not is_redundant:
                compressed.append(msg)

            seen_hashes.add(msg_hash)

        compressed = self.merge_similar_messages(compressed)

        logger.info(
            f"Compressed context: {original_count} -> {len(compressed)} messages "
            f"({(1 - len(compressed) / original_count):.1%} reduction)"
        )

        return compressed

    def retrieve_relevant_context(
        self, query: str, history: list[dict], max_results: int = 5
    ) -> list[dict]:
        """Retrieve relevant context from conversation history.

        Args:
            query: Query to match against
            history: Conversation history
            max_results: Maximum number of results

        Returns:
            List of relevant messages
        """
        if not history:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))

        scored_messages = []

        for idx, msg in enumerate(history):
            content = msg.get("content", "").lower()
            content_words = set(re.findall(r"\b\w+\b", content))

            intersection = query_words & content_words
            union = query_words | content_words

            jaccard_sim = len(intersection) / len(union) if union else 0.0

            if query_lower in content:
                jaccard_sim += 0.3

            if jaccard_sim > 0.1:
                scored_messages.append((jaccard_sim, idx, msg))

        scored_messages.sort(key=lambda x: x[0], reverse=True)

        return [msg for _, _, msg in scored_messages[:max_results]]

    def get_context(
        self,
        messages: list[dict],
        context_type: ContextType = ContextType.FULL,
        max_tokens: int | None = None,
    ) -> list[dict]:
        """Get context in specified format.

        Args:
            messages: Original messages
            context_type: Type of context to return
            max_tokens: Maximum tokens allowed

        Returns:
            Context messages in requested format
        """
        max_tokens = max_tokens or self.config.max_context_tokens

        if context_type == ContextType.FULL:
            return self.manage_context(messages, max_tokens)

        if context_type == ContextType.SUMMARY:
            current_tokens = self.get_token_usage(messages)

            if current_tokens > self.config.summary_threshold:
                summary = self.summarize_messages(messages)
                return [{"role": "system", "content": summary.summary_text}]

            return self.manage_context(messages, max_tokens)

        if context_type == ContextType.RELEVANT:
            if messages:
                last_user_msg = [m for m in reversed(messages) if m.get("role") == "user"]
                if last_user_msg:
                    query = last_user_msg[0].get("content", "")
                    return self.retrieve_relevant_context(query, messages)
            return messages[:5]

        if context_type == ContextType.MINIMAL:
            essential = self.get_essential_messages(messages)
            return self.manage_context(essential, max_tokens)

        return messages

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        words = len(text.split())
        chars = len(text)

        word_tokens = words
        char_tokens = chars / 4

        return int(max(word_tokens, char_tokens))

    def get_token_usage(self, messages: list[dict]) -> int:
        """Get current token usage for messages.

        Args:
            messages: List of messages

        Returns:
            Total token count
        """
        return sum(self.estimate_tokens(json.dumps(msg)) for msg in messages)

    def optimize_for_tokens(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Optimize context for token limits.

        Args:
            messages: Messages to optimize
            max_tokens: Maximum tokens allowed

        Returns:
            Optimized message list
        """
        current_tokens = self.get_token_usage(messages)

        if current_tokens <= max_tokens:
            return messages

        if current_tokens > max_tokens * 1.5:
            compressed = self.compress_context(messages)
            if self.get_token_usage(compressed) <= max_tokens:
                return compressed

        return self.manage_context(messages, max_tokens)

    def analyze_context(self, messages: list[dict]) -> ContextAnalysis:
        """Analyze context quality and potential improvements.

        Args:
            messages: Messages to analyze

        Returns:
            ContextAnalysis with findings
        """
        total_messages = len(messages)
        total_tokens = self.get_token_usage(messages)

        redundancy_groups = self.detect_redundancy(messages)
        redundant_messages = sum(len(g) - 1 for g in redundancy_groups)

        compression_potential = redundant_messages / max(total_messages, 1)

        quality_score = 1.0 - min(compression_potential, 0.5)

        gaps = []
        suggestions = []

        if total_messages > 20:
            suggestions.append("Consider summarizing older messages")

        if compression_potential > 0.2:
            suggestions.append(f"Found {redundant_messages} redundant messages that can be removed")

        if total_tokens > self.config.max_context_tokens:
            gaps.append("Context exceeds token limits")
            suggestions.append("Apply context window management")

        if not any(m.get("role") == "user" for m in messages):
            gaps.append("No user messages found")

        return ContextAnalysis(
            total_messages=total_messages,
            total_tokens=total_tokens,
            redundant_messages=redundant_messages,
            compression_potential=compression_potential,
            quality_score=quality_score,
            gaps=gaps,
            suggestions=suggestions,
        )

    def is_tool_result_important(self, result: dict) -> bool:
        """Check if tool result is important enough to keep.

        Args:
            result: Tool result dictionary

        Returns:
            True if result is important
        """
        if not result:
            return False

        if isinstance(result, str):
            result_str = result
        else:
            result_str = json.dumps(result, default=str)

        result_lower = result_str.lower()

        if any(keyword in result_lower for keyword in ["error", "fail", "exception", "timeout"]):
            return True

        if "success" in result_lower and len(result_str) < 500:
            return False

        if len(result_str) > 2000:
            return False

        return True

    def get_essential_messages(self, messages: list[dict]) -> list[dict]:
        """Get only essential messages from context.

        Args:
            messages: All messages

        Returns:
            Essential messages only
        """
        essential = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "user":
                essential.append(msg)

            elif role == "assistant":
                content = msg.get("content", "")
                if any(
                    keyword in content.lower()
                    for keyword in ["final answer", "conclusion", "result", "completed"]
                ):
                    essential.append(msg)

            elif role == "tool":
                if self.is_tool_result_important(msg.get("content", "")):
                    essential.append(msg)

        return essential

    def merge_similar_messages(self, messages: list[dict]) -> list[dict]:
        """Merge similar consecutive messages.

        Args:
            messages: Messages to merge

        Returns:
            Merged message list
        """
        if not messages:
            return []

        merged = []
        i = 0

        while i < len(messages):
            current = messages[i]

            if i + 1 < len(messages):
                next_msg = messages[i + 1]

                if current.get("role") == next_msg.get("role") and current.get("role") in [
                    "tool",
                    "assistant",
                ]:
                    merged_content = current.get("content", "") + "\n" + next_msg.get("content", "")
                    merged.append({"role": current.get("role"), "content": merged_content})
                    i += 2
                    continue

            merged.append(current)
            i += 1

        return merged

    def detect_redundancy(self, messages: list[dict]) -> list[set]:
        """Detect redundant messages.

        Args:
            messages: Messages to analyze

        Returns:
            List of sets containing indices of redundant messages
        """
        redundancy_groups = []
        seen_hashes = {}
        hash_groups = defaultdict(set)

        for idx, msg in enumerate(messages):
            msg_hash = self._hash_message(msg)
            hash_groups[msg_hash].add(idx)

        for msg_hash, indices in hash_groups.items():
            if len(indices) > 1:
                redundancy_groups.append(indices)

        return redundancy_groups

    def _hash_message(self, message: dict) -> str:
        """Generate hash for message deduplication.

        Args:
            message: Message to hash

        Returns:
            Hash string
        """
        content = message.get("content", "")
        role = message.get("role", "")

        normalized = re.sub(r"\s+", " ", content.lower().strip())
        hash_input = f"{role}:{normalized}"

        return hashlib.md5(hash_input.encode()).hexdigest()

    def _generate_cache_key(self, messages: list[dict]) -> str:
        """Generate cache key for messages.

        Args:
            messages: Messages to cache

        Returns:
            Cache key string
        """
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _store_summary(self, cache_key: str, summary: ContextSummary):
        """Store summary in database.

        Args:
            cache_key: Cache key for the summary
            summary: ContextSummary to store
        """
        if not self.db_manager:
            return

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO context_summaries
                    (cache_key, summary_text, key_points, original_message_count,
                     summarized_message_count, token_reduction, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        summary.summary_text,
                        json.dumps(summary.key_points),
                        summary.original_messages,
                        summary.summarized_messages,
                        summary.token_reduction,
                        self.config.summary_compression_ratio,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")

    def get_usage_statistics(self) -> dict[str, Any]:
        """Get context usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "cache_size": len(self._context_cache),
            "summary_cache_size": len(self._summary_cache),
            "total_usage_patterns": len(self._usage_patterns),
            "recent_token_history": self._token_history[-10:],
            "config": {
                "max_context_tokens": self.config.max_context_tokens,
                "summary_threshold": self.config.summary_threshold,
                "compression_strategy": self.config.compression_strategy.value,
            },
        }

    def clear_cache(self):
        """Clear all caches."""
        self._context_cache.clear()
        self._summary_cache.clear()
        self._usage_patterns.clear()
        self._token_history.clear()
        logger.info("Context manager cache cleared")
