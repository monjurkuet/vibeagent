# VibeAgent Comprehensive Implementation Plan
## Multi-Call LLM Improvements + SQLite Database Storage System

---

## Executive Summary

This plan integrates two critical initiatives:
1. **Multi-Call LLM Improvements**: Enhancing tool calling capabilities from 65% to 95% success rate
2. **SQLite Database Storage**: Comprehensive tracking of all interactions for analysis, evolution tracking, and future data utilization

**Total Timeline**: 12-16 weeks
**Total Effort**: ~90-110 person-days
**Expected Impact**:
- Success rate: 65% → 95%
- Complete interaction traceability
- Data-driven optimization capabilities
- Historical analysis and pattern recognition

---

## Part 1: Multi-Call LLM Improvements (Summary)

### Phase 1: Quick Wins (Weeks 1-2) - 5-7 days
**Impact**: +15% success rate (65% → 80%)

1. **Add System Prompt** (2 days)
   - Location: `skills/llm_skill.py:112-164`
   - Add comprehensive tool usage instructions
   - Include multi-step reasoning guidance
   - Add error handling protocols

2. **Implement Few-Shot Examples** (2 days)
   - Location: `skills/llm_skill.py`
   - Add 3-5 examples of tool calling
   - Include sequential and parallel examples
   - Show error recovery scenarios

3. **Enhance Error Feedback** (1 day)
   - Location: `core/tool_orchestrator.py:116-128`
   - Add detailed error context
   - Include retry suggestions
   - Format errors for LLM understanding

### Phase 2: Core Improvements (Weeks 3-6) - 15-20 days
**Impact**: +10% success rate (80% → 90%)

4. **Implement ReAct Loop** (5 days)
   - Location: `core/tool_orchestrator.py`
   - Add Thought/Action/Observation structure
   - Enable reasoning traces
   - Support plan revision

5. **Add Retry Logic** (3 days)
   - Location: `core/tool_orchestrator.py:213-249`
   - Configurable retry count
   - Exponential backoff
   - Retryable error detection

6. **Implement Parallel Execution** (5 days)
   - Location: `core/tool_orchestrator.py:106-128`
   - Async tool execution
   - Dependency detection
   - Result aggregation

7. **Add Self-Correction** (4 days)
   - Location: `core/tool_orchestrator.py`
   - Reflection prompts after errors
   - Plan revision capability
   - Alternative strategy exploration

### Phase 3: Advanced Features (Weeks 7-14) - 25-35 days
**Impact**: +5% success rate (90% → 95%)

8. **Tree of Thoughts** (10 days)
   - New file: `core/tot_orchestrator.py`
   - Multiple reasoning branches
   - Path evaluation and selection
   - Backtracking capability

9. **Plan-and-Execute** (10 days)
   - New file: `core/plan_execute_orchestrator.py`
   - Separate planning phase
   - Structured plan representation
   - Adaptive plan execution

10. **Context Management** (8 days)
    - Location: `core/tool_orchestrator.py`
    - Conversation summarization
    - Context windowing
    - Memory retrieval

11. **Model-Specific Configs** (5 days)
    - New file: `config/model_configs.py`
    - Per-model prompt templates
    - Temperature tuning
    - Iteration limits

*See MULTI_CALL_IMPROVEMENT_PLAN.md for detailed breakdown*

---

## Part 2: SQLite Database Storage System

### 2.1 Research: What Data to Store

Based on research of LLM agent systems and best practices for observability, the following data categories are critical:

#### Essential Data Categories:

1. **Conversations**
   - Full message history (user, assistant, system, tool)
   - Timestamps for each message
   - Message metadata (tokens, model, temperature)

2. **LLM Responses**
   - Raw API responses
   - Parsed tool calls
   - Reasoning traces (if available)
   - Token usage statistics
   - Latency metrics
   - Model version

3. **Tool Calls**
   - Tool name and parameters
   - Execution results (success/failure)
   - Execution time
   - Error details
   - Dependencies between calls

4. **Test Cases**
   - Test definitions
   - Expected outputs
   - Test metadata (name, category, complexity)

5. **Test Results**
   - Pass/fail status
   - Actual outputs vs expected
   - Performance metrics
   - Timestamp

6. **Judge Evaluations**
   - Judge model used
   - Evaluation criteria
   - Pass/fail with confidence
   - Reasoning/explanation
   - Detailed scoring

7. **Orchestration Metadata**
   - Iteration counts
   - Execution strategy used
   - Total execution time
   - Error recovery attempts

8. **Evolution Tracking**
   - Version history of prompts
   - Configuration changes
   - Performance trends over time
   - A/B test results

### 2.2 Database Schema Design

#### Core Tables

```sql
-- 1. Sessions: Top-level container for a complete interaction
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_type TEXT NOT NULL, -- 'test', 'production', 'development'
    orchestrator_type TEXT, -- 'basic', 'react', 'tot', 'plan_execute'
    model TEXT NOT NULL,
    total_iterations INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    total_duration_ms INTEGER,
    final_status TEXT, -- 'success', 'error', 'timeout', 'max_iterations'
    error_message TEXT,
    metadata JSON
);

-- 2. Messages: Individual messages in a conversation
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL, -- 'user', 'assistant', 'system', 'tool'
    content TEXT,
    raw_content TEXT, -- Original raw content
    tokens_input INTEGER,
    tokens_output INTEGER,
    model TEXT,
    temperature REAL,
    max_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_message_id INTEGER,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- 3. LLM Responses: Detailed LLM API responses
CREATE TABLE llm_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    message_id INTEGER,
    request_id TEXT UNIQUE,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    response_time_ms INTEGER,
    finish_reason TEXT,
    raw_response JSON,
    reasoning_content TEXT, -- If model provides reasoning
    tool_calls_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- 4. Tool Calls: Individual tool executions
CREATE TABLE tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    call_index INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_version TEXT,
    parameters JSON,
    execution_time_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    error_type TEXT, -- 'validation', 'execution', 'network', 'timeout', 'other'
    retry_count INTEGER DEFAULT 0,
    is_parallel BOOLEAN DEFAULT 0,
    parallel_batch_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- 5. Tool Results: Results from tool execution
CREATE TABLE tool_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_call_id INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    data JSON,
    error TEXT,
    result_size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id) ON DELETE CASCADE
);

-- 6. Test Cases: Definitions of test cases
CREATE TABLE test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT, -- 'simple', 'multi_call', 'complex', 'error_handling', 'parallel'
    description TEXT,
    messages JSON NOT NULL,
    tools JSON NOT NULL,
    expected_tools JSON,
    expected_parameters JSON,
    expect_no_tools BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT 1,
    metadata JSON
);

-- 7. Test Runs: Execution of test cases
CREATE TABLE test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_case_id INTEGER NOT NULL,
    session_id INTEGER,
    run_number INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    status TEXT, -- 'passed', 'failed', 'error', 'timeout'
    total_iterations INTEGER,
    total_tool_calls INTEGER,
    passed BOOLEAN,
    metadata JSON,
    FOREIGN KEY (test_case_id) REFERENCES test_cases(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- 8. Judge Evaluations: LLM judge results
CREATE TABLE judge_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id INTEGER NOT NULL,
    tool_call_id INTEGER,
    judge_model TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    evaluation_type TEXT, -- 'semantic', 'exact_match', 'custom'
    criteria JSON,
    details JSON,
    evaluation_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- 9. Reasoning Steps: For ReAct and ToT patterns
CREATE TABLE reasoning_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    step_type TEXT NOT NULL, -- 'thought', 'action', 'observation', 'reflection'
    content TEXT NOT NULL,
    tool_call_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- 10. Error Recovery: Track error recovery attempts
CREATE TABLE error_recovery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    tool_call_id INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    recovery_strategy TEXT NOT NULL, -- 'retry', 'modify_params', 'alternative_tool', 'skip'
    attempt_number INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    original_error TEXT,
    recovery_details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- 11. Configurations: Track configuration changes
CREATE TABLE configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_value JSON NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT 1,
    description TEXT,
    UNIQUE(config_name, config_key, version)
);

-- 12. Performance Metrics: Aggregated performance data
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- 13. Prompts: Track prompt versions and templates
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name TEXT NOT NULL,
    prompt_type TEXT NOT NULL, -- 'system', 'user', 'few_shot', 'error_recovery'
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    UNIQUE(prompt_name, prompt_type, version)
);

-- 14. A/B Tests: Track experimental variations
CREATE TABLE ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT NOT NULL UNIQUE,
    description TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT, -- 'running', 'completed', 'paused'
    variants JSON NOT NULL,
    success_metric TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 15. A/B Test Results: Results for each variant
CREATE TABLE ab_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ab_test_id INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    variant_name TEXT NOT NULL,
    metrics JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ab_test_id) REFERENCES ab_tests(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- 16. Tags: For categorization and filtering
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT UNIQUE NOT NULL,
    tag_type TEXT, -- 'category', 'priority', 'status', 'custom'
    color TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 17. Entity Tags: Many-to-many relationship for tagging
CREATE TABLE entity_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL, -- 'session', 'test_case', 'tool_call'
    entity_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    UNIQUE(entity_type, entity_id, tag_id)
);

-- 18. Insights: Auto-generated insights from data analysis
CREATE TABLE insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_type TEXT NOT NULL, -- 'pattern', 'anomaly', 'trend', 'recommendation'
    title TEXT NOT NULL,
    description TEXT,
    severity TEXT, -- 'info', 'warning', 'critical'
    data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_at TIMESTAMP
);

-- 19. Exports: Track data exports
CREATE TABLE exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    export_type TEXT NOT NULL, -- 'csv', 'json', 'parquet'
    filters JSON,
    record_count INTEGER,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
);

-- 20. System Events: Track system-level events
CREATE TABLE system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL, -- 'startup', 'shutdown', 'config_change', 'error'
    event_level TEXT, -- 'info', 'warning', 'error', 'critical'
    message TEXT NOT NULL,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 Indexes for Performance

```sql
-- Sessions
CREATE INDEX idx_sessions_created_at ON sessions(created_at);
CREATE INDEX idx_sessions_status ON sessions(final_status);
CREATE INDEX idx_sessions_model ON sessions(model);
CREATE INDEX idx_sessions_type ON sessions(session_type);

-- Messages
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Tool Calls
CREATE INDEX idx_tool_calls_session_id ON tool_calls(session_id);
CREATE INDEX idx_tool_calls_tool_name ON tool_calls(tool_name);
CREATE INDEX idx_tool_calls_success ON tool_calls(success);
CREATE INDEX idx_tool_calls_created_at ON tool_calls(created_at);

-- Test Runs
CREATE INDEX idx_test_runs_test_case_id ON test_runs(test_case_id);
CREATE INDEX idx_test_runs_status ON test_runs(status);
CREATE INDEX idx_test_runs_started_at ON test_runs(started_at);

-- Judge Evaluations
CREATE INDEX idx_judge_evaluations_test_run_id ON judge_evaluations(test_run_id);
CREATE INDEX idx_judge_evaluations_passed ON judge_evaluations(passed);

-- Performance Metrics
CREATE INDEX idx_performance_metrics_session_id ON performance_metrics(session_id);
CREATE INDEX idx_performance_metrics_name ON performance_metrics(metric_name);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
```

### 2.4 Views for Common Queries

```sql
-- Session Summary View
CREATE VIEW session_summary AS
SELECT
    s.id,
    s.session_id,
    s.created_at,
    s.session_type,
    s.model,
    s.total_iterations,
    s.total_tool_calls,
    s.total_duration_ms,
    s.final_status,
    COUNT(DISTINCT tc.id) as unique_tools_used,
    SUM(CASE WHEN tc.success = 0 THEN 1 ELSE 0 END) as failed_tool_calls,
    AVG(tc.execution_time_ms) as avg_tool_execution_time
FROM sessions s
LEFT JOIN tool_calls tc ON s.id = tc.session_id
GROUP BY s.id;

-- Test Performance View
CREATE VIEW test_performance AS
SELECT
    tc.name as test_name,
    tc.category,
    COUNT(tr.id) as total_runs,
    SUM(CASE WHEN tr.passed = 1 THEN 1 ELSE 0 END) as passed_runs,
    ROUND(CAST(SUM(CASE WHEN tr.passed = 1 THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(tr.id), 2) as pass_rate,
    AVG(tr.duration_ms) as avg_duration_ms,
    AVG(tr.total_iterations) as avg_iterations,
    MAX(tr.completed_at) as last_run
FROM test_cases tc
LEFT JOIN test_runs tr ON tc.id = tr.test_case_id
GROUP BY tc.id;

-- Tool Success Rate View
CREATE VIEW tool_success_rate AS
SELECT
    tool_name,
    COUNT(*) as total_calls,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
    ROUND(CAST(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as success_rate,
    AVG(execution_time_ms) as avg_execution_time_ms,
    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls
FROM tool_calls
GROUP BY tool_name
ORDER BY success_rate DESC;

-- Model Comparison View
CREATE VIEW model_comparison AS
SELECT
    model,
    COUNT(*) as total_sessions,
    SUM(CASE WHEN final_status = 'success' THEN 1 ELSE 0 END) as successful_sessions,
    ROUND(CAST(SUM(CASE WHEN final_status = 'success' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as success_rate,
    AVG(total_duration_ms) as avg_duration_ms,
    AVG(total_iterations) as avg_iterations,
    AVG(total_tool_calls) as avg_tool_calls
FROM sessions
GROUP BY model;

-- Error Analysis View
CREATE VIEW error_analysis AS
SELECT
    error_type,
    COUNT(*) as error_count,
    tool_name,
    ROUND(CAST(COUNT(*) AS FLOAT) * 100.0 / (SELECT COUNT(*) FROM tool_calls WHERE success = 0), 2) as error_percentage,
    AVG(retry_count) as avg_retry_count,
    SUM(CASE WHEN er.success = 1 THEN 1 ELSE 0 END) as recovered_count
FROM tool_calls tc
LEFT JOIN error_recovery er ON tc.id = er.tool_call_id
WHERE tc.success = 0
GROUP BY error_type, tool_name
ORDER BY error_count DESC;
```

---

## Part 3: Database Integration Architecture

### 3.1 Database Manager Implementation

**File**: `core/database_manager.py`

```python
"""Database manager for SQLite storage and retrieval."""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for VibeAgent."""

    def __init__(self, db_path: str = "data/vibeagent.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database schema if not exists."""
        with self.get_connection() as conn:
            # Read schema from file
            schema_file = Path(__file__).parent.parent / "config" / "schema.sql"
            if schema_file.exists():
                with open(schema_file) as f:
                    schema = f.read()
                conn.executescript(schema)
                logger.info("Database schema initialized")

    # Session Operations
    def create_session(
        self,
        session_id: str,
        session_type: str,
        model: str,
        orchestrator_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Create a new session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions
                (session_id, session_type, model, orchestrator_type, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, session_type, model, orchestrator_type,
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    def update_session(
        self,
        session_id: int,
        **kwargs
    ):
        """Update session fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = []
            values = []
            for key, value in kwargs.items():
                if key == 'metadata':
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value) if value else None)
                else:
                    fields.append(f"{key} = ?")
                    values.append(value)
            values.append(session_id)
            cursor.execute(
                f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?",
                values
            )

    # Message Operations
    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        message_index: int,
        raw_content: Optional[str] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_message_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a message to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages
                (session_id, role, content, raw_content, message_index,
                 tokens_input, tokens_output, model, temperature, max_tokens,
                 parent_message_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, raw_content, message_index,
                 tokens_input, tokens_output, model, temperature, max_tokens,
                 parent_message_id, json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # LLM Response Operations
    def add_llm_response(
        self,
        session_id: int,
        message_id: Optional[int],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        response_time_ms: int,
        finish_reason: Optional[str] = None,
        raw_response: Optional[Dict] = None,
        reasoning_content: Optional[str] = None,
        tool_calls_count: int = 0,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add an LLM response to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO llm_responses
                (session_id, message_id, model, prompt_tokens, completion_tokens,
                 total_tokens, response_time_ms, finish_reason, raw_response,
                 reasoning_content, tool_calls_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, message_id, model, prompt_tokens, completion_tokens,
                 total_tokens, response_time_ms, finish_reason,
                 json.dumps(raw_response) if raw_response else None,
                 reasoning_content, tool_calls_count,
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Tool Call Operations
    def add_tool_call(
        self,
        session_id: int,
        call_index: int,
        tool_name: str,
        parameters: Dict,
        execution_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        retry_count: int = 0,
        is_parallel: bool = False,
        parallel_batch_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a tool call to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_calls
                (session_id, call_index, tool_name, parameters, execution_time_ms,
                 success, error_message, error_type, retry_count, is_parallel,
                 parallel_batch_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, call_index, tool_name, json.dumps(parameters),
                 execution_time_ms, success, error_message, error_type,
                 retry_count, is_parallel, parallel_batch_id,
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Tool Result Operations
    def add_tool_result(
        self,
        tool_call_id: int,
        success: bool,
        data: Optional[Dict] = None,
        error: Optional[str] = None,
        result_size_bytes: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a tool result to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_results
                (tool_call_id, success, data, error, result_size_bytes, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tool_call_id, success, json.dumps(data) if data else None,
                 error, result_size_bytes, json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Test Case Operations
    def create_test_case(
        self,
        name: str,
        category: str,
        description: Optional[str],
        messages: List[Dict],
        tools: List[Dict],
        expected_tools: Optional[List[Dict]] = None,
        expected_parameters: Optional[Dict] = None,
        expect_no_tools: bool = False,
        metadata: Optional[Dict] = None
    ) -> int:
        """Create a test case."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_cases
                (name, category, description, messages, tools, expected_tools,
                 expected_parameters, expect_no_tools, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, category, description, json.dumps(messages),
                 json.dumps(tools), json.dumps(expected_tools) if expected_tools else None,
                 json.dumps(expected_parameters) if expected_parameters else None,
                 expect_no_tools, json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Test Run Operations
    def create_test_run(
        self,
        test_case_id: int,
        run_number: int,
        session_id: Optional[int] = None,
        status: str = "running"
    ) -> int:
        """Create a test run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_runs
                (test_case_id, session_id, run_number, status, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (test_case_id, session_id, run_number, status, datetime.now())
            )
            return cursor.lastrowid

    def update_test_run(
        self,
        test_run_id: int,
        **kwargs
    ):
        """Update test run fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = []
            values = []
            for key, value in kwargs.items():
                if key in ['metadata']:
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value) if value else None)
                else:
                    fields.append(f"{key} = ?")
                    values.append(value)
            values.append(test_run_id)
            cursor.execute(
                f"UPDATE test_runs SET {', '.join(fields)} WHERE id = ?",
                values
            )

    # Judge Evaluation Operations
    def add_judge_evaluation(
        self,
        test_run_id: int,
        judge_model: str,
        passed: bool,
        confidence: float,
        reasoning: str,
        evaluation_type: str = "semantic",
        criteria: Optional[Dict] = None,
        details: Optional[Dict] = None,
        evaluation_time_ms: Optional[int] = None,
        tool_call_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a judge evaluation to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO judge_evaluations
                (test_run_id, tool_call_id, judge_model, passed, confidence,
                 reasoning, evaluation_type, criteria, details, evaluation_time_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (test_run_id, tool_call_id, judge_model, passed, confidence,
                 reasoning, evaluation_type, json.dumps(criteria) if criteria else None,
                 json.dumps(details) if details else None, evaluation_time_ms,
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Reasoning Step Operations
    def add_reasoning_step(
        self,
        session_id: int,
        iteration: int,
        step_type: str,
        content: str,
        tool_call_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a reasoning step (for ReAct/ToT)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO reasoning_steps
                (session_id, iteration, step_type, content, tool_call_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, iteration, step_type, content, tool_call_id,
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Error Recovery Operations
    def add_error_recovery(
        self,
        session_id: int,
        tool_call_id: int,
        error_type: str,
        recovery_strategy: str,
        attempt_number: int,
        success: bool,
        original_error: str,
        recovery_details: Optional[Dict] = None
    ) -> int:
        """Add an error recovery attempt."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO error_recovery
                (session_id, tool_call_id, error_type, recovery_strategy,
                 attempt_number, success, original_error, recovery_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, tool_call_id, error_type, recovery_strategy,
                 attempt_number, success, original_error,
                 json.dumps(recovery_details) if recovery_details else None)
            )
            return cursor.lastrowid

    # Performance Metrics Operations
    def add_performance_metric(
        self,
        session_id: int,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a performance metric."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics
                (session_id, metric_name, metric_value, metric_unit, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, metric_name, metric_value, metric_unit,
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    # Query Operations
    def get_session(self, session_id: int) -> Optional[Dict]:
        """Get session by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session_messages(self, session_id: int) -> List[Dict]:
        """Get all messages for a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY message_index",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_test_performance(self, days: int = 30) -> List[Dict]:
        """Get test performance for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM test_performance
                WHERE last_run >= datetime('now', '-' || ? || ' days')
                ORDER BY last_run DESC
                """,
                (days,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_tool_success_rate(self) -> List[Dict]:
        """Get tool success rate statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tool_success_rate")
            return [dict(row) for row in cursor.fetchall()]

    def get_model_comparison(self) -> List[Dict]:
        """Get model comparison statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_comparison")
            return [dict(row) for row in cursor.fetchall()]

    # Export Operations
    def export_to_json(
        self,
        query: str,
        params: Optional[tuple] = None,
        file_path: Optional[str] = None
    ) -> str:
        """Export query results to JSON."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            rows = [dict(row) for row in cursor.fetchall()]

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(rows, f, indent=2)
            return file_path
        else:
            return json.dumps(rows, indent=2)
```

### 3.2 Integration Points

**Modified: `core/tool_orchestrator.py`**

```python
# Add database manager integration
def __init__(self, llm_skill, skills: Dict[str, BaseSkill], db_manager=None):
    self.llm_skill = llm_skill
    self.skills = skills
    self._tool_schemas = self._build_tool_schemas()
    self.db_manager = db_manager

def execute_with_tools(
    self, user_message: str, max_iterations: int = 10, session_type: str = "production"
) -> OrchestratorResult:
    import uuid
    from datetime import datetime

    session_id = str(uuid.uuid4())
    start_time = datetime.now()

    # Create session in database
    db_session_id = None
    if self.db_manager:
        db_session_id = self.db_manager.create_session(
            session_id=session_id,
            session_type=session_type,
            model=self.llm_skill.model,
            orchestrator_type="basic",
            metadata={"max_iterations": max_iterations}
        )

    # Store user message
    if self.db_manager and db_session_id:
        self.db_manager.add_message(
            session_id=db_session_id,
            role="user",
            content=user_message,
            message_index=0
        )

    # ... existing orchestration logic ...

    # Track each tool call
    for tool_call in tool_calls:
        tool_start = datetime.now()
        tool_result = self._execute_tool(tool_call)
        tool_time_ms = int((datetime.now() - tool_start).total_seconds() * 1000)

        if self.db_manager and db_session_id:
            tool_call_id = self.db_manager.add_tool_call(
                session_id=db_session_id,
                call_index=tool_calls_made,
                tool_name=function_name,
                parameters=arguments,
                execution_time_ms=tool_time_ms,
                success=tool_result.success,
                error_message=tool_result.error,
                metadata={}
            )

            self.db_manager.add_tool_result(
                tool_call_id=tool_call_id,
                success=tool_result.success,
                data=tool_result.data,
                error=tool_result.error
            )

        tool_results.append({
            "tool_call": tool_call,
            "result": tool_result,
        })

    # Update session on completion
    if self.db_manager and db_session_id:
        total_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        self.db_manager.update_session(
            db_session_id,
            total_iterations=iterations,
            total_tool_calls=tool_calls_made,
            total_duration_ms=total_duration_ms,
            final_status="success" if result.success else "error",
            error_message=result.error
        )

    return result
```

**Modified: `tests/llm_tool_calling_tester.py`**

```python
# Add database integration for test runs
def run_test_case(self, test_case: Dict) -> Dict:
    test_name = test_case.get("name", "Unknown")
    logger.info(f"\n{'='*60}")
    logger.info(f"Running test: {test_name}")
    logger.info(f"{'='*60}")

    # Create test run in database
    test_run_id = None
    if self.db_manager:
        # Get or create test case
        test_case_id = self._get_or_create_test_case(test_case)
        run_number = self._get_next_run_number(test_case_id)

        test_run_id = self.db_manager.create_test_run(
            test_case_id=test_case_id,
            run_number=run_number,
            status="running"
        )

    start_time = time.time()

    try:
        # Run the test
        result = self.orchestrator.execute_with_tools(
            messages=test_case.get("messages", []),
            session_type="test"
        )

        execution_time = time.time() - start_time

        # Get actual tool calls
        actual_tool_calls = []
        for tool_result in result.tool_results:
            tool_call = tool_result.get("tool_call", {})
            actual_tool_calls.append(tool_call)

        # Judge evaluation
        if self.judge:
            judge_result = self.judge.verify_tool_call(test_case, actual_tool_calls)

            # Store judge evaluation
            if self.db_manager and test_run_id:
                self.db_manager.add_judge_evaluation(
                    test_run_id=test_run_id,
                    judge_model=self.judge.judge_model,
                    passed=judge_result.get("passed", False),
                    confidence=judge_result.get("confidence", 0.0),
                    reasoning=judge_result.get("reasoning", ""),
                    evaluation_type="semantic",
                    criteria=test_case.get("expected_tools"),
                    details=judge_result.get("details", {}),
                    evaluation_time_ms=int(execution_time * 1000)
                )
        else:
            judge_result = None

        # Update test run
        if self.db_manager and test_run_id:
            self.db_manager.update_test_run(
                test_run_id,
                completed_at=datetime.now(),
                duration_ms=int(execution_time * 1000),
                status="passed" if result.success else "failed",
                total_iterations=result.iterations,
                total_tool_calls=result.tool_calls_made,
                passed=judge_result.get("passed", False) if judge_result else result.success,
                metadata={
                    "judge_result": judge_result,
                    "actual_tool_calls": actual_tool_calls
                }
            )

        return {
            "name": test_name,
            "success": result.success,
            "iterations": result.iterations,
            "tool_calls_made": result.tool_calls_made,
            "execution_time": execution_time,
            "judge_result": judge_result,
            "actual_tool_calls": actual_tool_calls
        }

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")

        if self.db_manager and test_run_id:
            self.db_manager.update_test_run(
                test_run_id,
                completed_at=datetime.now(),
                duration_ms=int((time.time() - start_time) * 1000),
                status="error",
                passed=False,
                metadata={"error": str(e)}
            )

        return {
            "name": test_name,
            "success": False,
            "error": str(e)
        }
```

---

## Part 4: Implementation Roadmap

### Phase 0: Database Foundation (Week 1) - 3-5 days

**Tasks:**
1. Create database schema file (`config/schema.sql`)
2. Implement `DatabaseManager` class (`core/database_manager.py`)
3. Create database initialization script
4. Add database configuration to config system
5. Create database migration system for future updates
6. Write unit tests for database operations

**Deliverables:**
- `config/schema.sql` - Complete schema with all tables
- `core/database_manager.py` - Database manager implementation
- `scripts/init_db.py` - Database initialization script
- `tests/test_database_manager.py` - Database tests

**Success Criteria:**
- All 20 tables created successfully
- All indexes created
- All views created
- Database manager can CRUD all entities
- Unit tests pass with 80%+ coverage

---

### Phase 1: Quick Wins + Database Integration (Weeks 2-3) - 8-10 days

**Tasks:**
1. **Add System Prompt** (2 days)
   - Create `prompts/react_prompt.py`
   - Integrate with `LLMSkill`
   - Add to database as versioned prompt

2. **Implement Few-Shot Examples** (2 days)
   - Create examples from test cases
   - Store in database
   - Dynamic loading based on test category

3. **Enhance Error Feedback** (1 day)
   - Improve error messages
   - Store error patterns in database
   - Enable error recovery tracking

4. **Database Integration - Tool Orchestrator** (2 days)
   - Add database manager to orchestrator
   - Track sessions, messages, tool calls
   - Store all execution data

5. **Database Integration - Test Runner** (2 days)
   - Track test cases and runs
   - Store judge evaluations
   - Track performance metrics

6. **Testing** (1 day)
   - Integration tests
   - Verify data integrity
   - Check performance impact

**Deliverables:**
- `prompts/react_prompt.py`
- `prompts/few_shot_examples.py`
- Updated `core/tool_orchestrator.py`
- Updated `tests/llm_tool_calling_tester.py`
- Integration tests

**Success Criteria:**
- All interactions stored in database
- Test results tracked with judge evaluations
- Database queries don't degrade performance (>100ms)
- Success rate improves to 80%

---

### Phase 2: Core Improvements (Weeks 4-7) - 15-20 days

**Tasks:**
1. **Implement ReAct Loop** (5 days)
   - Modify `ToolOrchestrator`
   - Track reasoning steps in database
   - Enable reasoning trace visualization

2. **Add Retry Logic** (3 days)
   - Implement retry mechanism
   - Track retry attempts in database
   - Analyze retry effectiveness

3. **Implement Parallel Execution** (5 days)
   - Create `ParallelExecutor`
   - Track parallel batches in database
   - Measure performance improvements

4. **Add Self-Correction** (4 days)
   - Implement reflection logic
   - Track error recovery in database
   - Learn from recovery patterns

5. **Database Analytics** (3 days)
   - Create analytics queries
   - Build dashboard views
   - Generate performance reports

**Deliverables:**
- Updated `core/tool_orchestrator.py`
- New `core/parallel_executor.py`
- New `core/self_corrector.py`
- Analytics queries and views
- Performance reports

**Success Criteria:**
- Success rate improves to 90%
- Parallel execution reduces time by 40%
- Self-correction recovers 60% of errors
- All reasoning steps tracked
- Analytics dashboard functional

---

### Phase 3: Advanced Features (Weeks 8-14) - 25-35 days

**Tasks:**
1. **Tree of Thoughts** (10 days)
   - Create `TreeOfThoughtsOrchestrator`
   - Track thought branches in database
   - Implement path evaluation

2. **Plan-and-Execute** (10 days)
   - Create `PlanExecuteOrchestrator`
   - Store plans in database
   - Track plan execution

3. **Context Management** (8 days)
   - Implement context summarization
   - Track context usage in database
   - Optimize token usage

4. **Model-Specific Configs** (5 days)
   - Create model configurations
   - Store in database
   - A/B testing framework

5. **Advanced Analytics** (5 days)
   - Trend analysis
   - Pattern detection
   - Automated insights

6. **Data Export & API** (3 days)
   - Export functionality
   - REST API for data access
   - Data visualization tools

**Deliverables:**
- `core/tot_orchestrator.py`
- `core/plan_execute_orchestrator.py`
- `core/context_manager.py`
- `config/model_configs.py`
- Analytics dashboard
- Export tools
- REST API

**Success Criteria:**
- Success rate improves to 95%
- Complex tasks handled effectively
- Token usage reduced by 30%
- All data accessible via API
- Automated insights generated

---

## Part 5: Database Query Examples

### 5.1 Performance Analysis Queries

```sql
-- Average success rate by model over time
SELECT
    DATE(created_at) as date,
    model,
    COUNT(*) as total_sessions,
    SUM(CASE WHEN final_status = 'success' THEN 1 ELSE 0 END) as successful,
    ROUND(CAST(SUM(CASE WHEN final_status = 'success' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as success_rate
FROM sessions
WHERE created_at >= datetime('now', '-30 days')
GROUP BY DATE(created_at), model
ORDER BY date DESC, model;

-- Tool execution time distribution
SELECT
    tool_name,
    MIN(execution_time_ms) as min_time,
    MAX(execution_time_ms) as max_time,
    AVG(execution_time_ms) as avg_time,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time_ms) as median_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_time
FROM tool_calls
WHERE success = 1
GROUP BY tool_name;

-- Error recovery success rate
SELECT
    error_type,
    recovery_strategy,
    COUNT(*) as total_attempts,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_recoveries,
    ROUND(CAST(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as recovery_rate
FROM error_recovery
GROUP BY error_type, recovery_strategy
ORDER BY recovery_rate DESC;

-- Test performance trends
SELECT
    tc.name,
    tr.run_number,
    tr.passed,
    tr.duration_ms,
    tr.total_iterations,
    tr.completed_at
FROM test_runs tr
JOIN test_cases tc ON tr.test_case_id = tc.id
WHERE tc.name = 'Simple tool call - ArXiv search'
ORDER BY tr.run_number;
```

### 5.2 Pattern Detection Queries

```sql
-- Find frequently failing tools
SELECT
    tool_name,
    COUNT(*) as total_calls,
    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures,
    ROUND(CAST(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as failure_rate
FROM tool_calls
WHERE created_at >= datetime('now', '-7 days')
GROUP BY tool_name
HAVING failures > 5
ORDER BY failure_rate DESC;

-- Detect performance degradation
SELECT
    DATE(created_at) as date,
    AVG(total_duration_ms) as avg_duration,
    AVG(total_iterations) as avg_iterations,
    COUNT(*) as session_count
FROM sessions
WHERE created_at >= datetime('now', '-30 days')
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Find successful patterns
SELECT
    s.model,
    t.tool_name,
    COUNT(*) as usage_count,
    ROUND(CAST(AVG(t.execution_time_ms) AS FLOAT), 2) as avg_time,
    ROUND(CAST(SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as success_rate
FROM sessions s
JOIN tool_calls t ON s.id = t.session_id
WHERE s.final_status = 'success' AND t.success = 1
GROUP BY s.model, t.tool_name
ORDER BY usage_count DESC;
```

### 5.3 Evolution Tracking Queries

```sql
-- Track prompt version effectiveness
SELECT
    p.prompt_name,
    p.version,
    COUNT(DISTINCT s.id) as sessions_used,
    ROUND(CAST(SUM(CASE WHEN s.final_status = 'success' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as success_rate,
    AVG(s.total_iterations) as avg_iterations
FROM prompts p
JOIN sessions s ON JSON_EXTRACT(s.metadata, '$.prompt_version') = p.version
WHERE p.prompt_type = 'system'
GROUP BY p.prompt_name, p.version
ORDER BY p.version DESC;

-- A/B test comparison
SELECT
    abt.test_name,
    abtr.variant_name,
    COUNT(*) as runs,
    JSON_EXTRACT(abtr.metrics, '$.success_rate') as success_rate,
    JSON_EXTRACT(abtr.metrics, '$.avg_duration_ms') as avg_duration,
    JSON_EXTRACT(abtr.metrics, '$.avg_iterations') as avg_iterations
FROM ab_test_results abtr
JOIN ab_tests abt ON abtr.ab_test_id = abt.id
WHERE abt.status = 'running'
GROUP BY abt.test_name, abtr.variant_name;
```

---

## Part 6: Data Analysis & Insights

### 6.1 Key Metrics to Track

1. **Success Metrics**
   - Overall success rate
   - Per-tool success rate
   - Per-model success rate
   - Test case pass rate

2. **Performance Metrics**
   - Average execution time
   - Tool execution time
   - LLM response time
   - Total iterations

3. **Quality Metrics**
   - Judge pass rate
   - Confidence scores
   - Error recovery rate
   - Self-correction success

4. **Resource Metrics**
   - Token usage
   - API call count
   - Database size
   - Memory usage

5. **Evolution Metrics**
   - Success rate over time
   - Performance trends
   - Error pattern evolution
   - Prompt effectiveness

### 6.2 Automated Insights Generation

**File**: `core/analytics_engine.py`

```python
"""Analytics engine for generating insights from database."""

from typing import List, Dict, Any
from datetime import datetime, timedelta


class AnalyticsEngine:
    """Generate insights from stored interaction data."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def generate_daily_insights(self) -> List[Dict]:
        """Generate daily insights."""
        insights = []

        # Check for success rate drop
        success_rate_trend = self._check_success_rate_trend()
        if success_rate_trend['significant']:
            insights.append({
                'insight_type': 'trend',
                'title': 'Success Rate Change',
                'description': success_rate_trend['description'],
                'severity': 'warning' if success_rate_trend['direction'] == 'down' else 'info',
                'data': success_rate_trend
            })

        # Find failing tools
        failing_tools = self._find_failing_tools()
        if failing_tools:
            insights.append({
                'insight_type': 'anomaly',
                'title': 'Tools with High Failure Rate',
                'description': f"{len(failing_tools)} tools have failure rate > 20%",
                'severity': 'warning',
                'data': {'tools': failing_tools}
            })

        # Detect performance degradation
        perf_degradation = self._detect_performance_degradation()
        if perf_degradation['significant']:
            insights.append({
                'insight_type': 'anomaly',
                'title': 'Performance Degradation',
                'description': perf_degradation['description'],
                'severity': 'critical',
                'data': perf_degradation
            })

        # Find successful patterns
        successful_patterns = self._find_successful_patterns()
        if successful_patterns:
            insights.append({
                'insight_type': 'pattern',
                'title': 'High-Performing Patterns',
                'description': f"Found {len(successful_patterns)} successful patterns",
                'severity': 'info',
                'data': {'patterns': successful_patterns}
            })

        # Store insights in database
        for insight in insights:
            self.db_manager.add_insight(**insight)

        return insights

    def _check_success_rate_trend(self) -> Dict:
        """Check if success rate is trending up or down."""
        # Implementation
        pass

    def _find_failing_tools(self) -> List[Dict]:
        """Find tools with high failure rates."""
        # Implementation
        pass

    def _detect_performance_degradation(self) -> Dict:
        """Detect performance issues."""
        # Implementation
        pass

    def _find_successful_patterns(self) -> List[Dict]:
        """Find patterns that lead to success."""
        # Implementation
        pass
```

---

## Part 7: Risk Assessment & Mitigation

### 7.1 Database-Related Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Database size grows too large | High | Medium | Implement data retention policy, archiving, compression |
| Database becomes bottleneck | Medium | High | Use connection pooling, async operations, caching |
| Data corruption | Low | Critical | Regular backups, integrity checks, transactions |
| Schema migration issues | Medium | Medium | Versioned migrations, rollback capability, testing |
| Query performance degradation | High | Medium | Index optimization, query analysis, partitioning |

### 7.2 Data Privacy & Security

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Sensitive data in logs | Medium | High | Data redaction, encryption at rest |
| Unauthorized access | Low | Critical | Access controls, authentication |
| Data leakage | Low | High | Encryption, secure transmission |
| Compliance issues | Low | Medium | Data retention policies, GDPR compliance |

---

## Part 8: Resource Requirements

### 8.1 Personnel

| Role | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Total |
|------|---------|---------|---------|---------|-------|
| Senior Engineer | 3 days | 5 days | 10 days | 15 days | 33 days |
| Backend Engineer | 2 days | 4 days | 8 days | 12 days | 26 days |
| Data Engineer | - | 2 days | 5 days | 8 days | 15 days |
| QA Engineer | 1 day | 2 days | 4 days | 6 days | 13 days |
| DevOps Engineer | 1 day | 1 day | 2 days | 3 days | 7 days |

**Total: ~94 person-days (~19 weeks)**

### 8.2 Infrastructure

**Storage Requirements:**
- Week 1-4: ~100 MB
- Week 5-8: ~500 MB
- Week 9-12: ~2 GB
- Week 13-16: ~5 GB

**Backup Requirements:**
- Daily incremental backups
- Weekly full backups
- 30-day retention

---

## Part 9: Success Metrics

### 9.1 Database Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Data completeness | 100% | All interactions stored |
| Query performance | <100ms | 95th percentile |
| Database uptime | 99.9% | Monitoring |
| Backup success rate | 100% | Daily checks |

### 9.2 Integration Metrics

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Data stored | 0% | 60% | 90% | 100% |
| Queries functional | 0% | 40% | 80% | 100% |
| Analytics available | 0% | 20% | 60% | 100% |
| Export capabilities | 0% | 0% | 50% | 100% |

### 9.3 Combined Success Criteria

- ✅ All multi-call improvements implemented (65% → 95% success rate)
- ✅ All interactions stored in database
- ✅ Real-time analytics dashboard functional
- ✅ Historical trend analysis available
- ✅ Automated insights generation working
- ✅ Data export functionality operational
- ✅ Performance impact <10%
- ✅ Database backup and recovery verified

---

## Part 10: Next Steps

### Immediate Actions (This Week)

1. **Review and approve** this comprehensive plan
2. **Set up database environment** (SQLite file location, backup system)
3. **Create Phase 0 task breakdown** with specific assignments
4. **Set up monitoring** for database performance
5. **Establish data retention policy**

### Phase 0 Kickoff (Week 1)

1. Create `config/schema.sql` with all 20 tables
2. Implement `DatabaseManager` class
3. Write unit tests for database operations
4. Create database initialization script
5. Test database performance with sample data

### Phase 1 Preparation

1. Review and approve system prompt templates
2. Prepare few-shot examples from test cases
3. Set up integration test environment
4. Create performance benchmark suite

---

## Conclusion

This comprehensive plan integrates multi-call LLM improvements with a robust SQLite database storage system. The database will capture all interactions, enabling:

1. **Complete Traceability**: Every conversation, tool call, and decision tracked
2. **Performance Analysis**: Identify bottlenecks and optimization opportunities
3. **Evolution Tracking**: Monitor success rate improvements over time
4. **Pattern Recognition**: Learn from successful and failed executions
5. **Data-Driven Optimization**: Use historical data to improve future performance

The phased approach ensures quick wins while building toward advanced capabilities. With proper implementation, VibeAgent will achieve state-of-the-art multi-call capabilities with comprehensive observability and analytics.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-24
**Author**: VibeAgent Engineering Team
**Review Status**: Pending Review